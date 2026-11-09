"""
DatasetEvaluator: batch evaluation of a ReasoningChain over an AbstractDataset.

Usage example::

    from mmar_carl import ReasoningChain, ReasoningContext, ReflectionOptions
    from mmar_carl.dataset_evaluator import DatasetEvaluator
    from mmar_carl.models.dataset import DataCase, SimpleDataset, TopKWorstStrategy

    dataset = SimpleDataset([
        DataCase(input="sample 1 text", label="s1"),
        DataCase(input="sample 2 text", label="s2"),
    ])

    evaluator = DatasetEvaluator(
        chain=chain,
        dataset=dataset,
        metric=my_metric,
        strategy=TopKWorstStrategy(k=3),
    )

    report = evaluator.evaluate(
        context_factory=lambda case: ReasoningContext(
            outer_context=case.input,
            api=llm_client,
        )
    )

    # Use in reflection
    reflection = chain.reflect(
        "Improve chain quality",
        options=ReflectionOptions(dataset_report=report),
    )
"""

import asyncio
from typing import Callable

from .logging_utils import log_info, log_warning
from .metrics import MetricBase, call_metric_async
from .models.context import ReasoningContext
from .models.dataset import (
    AbstractDataset,
    CaseEvaluationResult,
    DataCase,
    DatasetEvaluationReport,
    SelectionStrategy,
)

ContextFactory = Callable[[DataCase], ReasoningContext]


class DatasetEvaluator:
    """
    Runs a :class:`~mmar_carl.chain.ReasoningChain` on every case in an
    :class:`~mmar_carl.models.dataset.AbstractDataset`, scores each output with
    a :class:`~mmar_carl.metrics.MetricBase`, and applies a
    :data:`~mmar_carl.models.dataset.SelectionStrategy` to identify problem cases
    for reflection.

    Args:
        chain: The reasoning chain to evaluate.
        dataset: Dataset of :class:`~mmar_carl.models.dataset.DataCase` objects.
        metric: Metric used to score each chain output.
        strategy: Strategy that selects problem cases from all scored results.
    """

    def __init__(
        self,
        chain: "ReasoningChain",  # type: ignore[name-defined]  # noqa: F821  # avoid circular import
        dataset: AbstractDataset,
        metric: MetricBase,
        strategy: SelectionStrategy,
        *,
        max_concurrent: int = 1,
    ) -> None:
        self._chain = chain
        self._dataset = dataset
        self._metric = metric
        self._strategy = strategy
        #: Bounded parallel case evaluation. Default 1 = sequential (rate-limit
        #: safe). Raise for fast cascade-validation passes over a subsample.
        self._max_concurrent = max(1, int(max_concurrent))

    async def evaluate_async(
        self,
        context_factory: ContextFactory,
    ) -> DatasetEvaluationReport:
        """
        Run the dataset evaluation asynchronously.

        Cases are evaluated **sequentially** to avoid overwhelming downstream
        LLM API rate limits.  Parallel execution may be added in a future
        extension if ``max_concurrent`` is needed.

        Args:
            context_factory: Callable that converts a :class:`DataCase` into a
                :class:`~mmar_carl.models.context.ReasoningContext`.  Use this to
                control how ``case.input`` (and any other case fields) are wired
                into the chain's context.

        Returns:
            :class:`~mmar_carl.models.dataset.DatasetEvaluationReport` with all
            scored results and the selected problem cases.
        """
        cases = list(self._dataset)
        if self._max_concurrent > 1:
            sem = asyncio.Semaphore(self._max_concurrent)

            async def _bounded(i: int, c: DataCase) -> CaseEvaluationResult:
                async with sem:
                    return await self._evaluate_case(i, c, context_factory)

            all_results = list(
                await asyncio.gather(*(_bounded(i, c) for i, c in enumerate(cases)))
            )
        else:
            all_results = [
                await self._evaluate_case(i, c, context_factory) for i, c in enumerate(cases)
            ]

        selected = self._strategy.select(all_results)

        scores = [r.score for r in all_results]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0

        # Surface noteworthy situations so users can act on them
        if not all_results:
            log_info(
                f"DatasetEvaluator: dataset is empty, "
                f"no cases evaluated (metric='{self._metric.name}')"
            )
        elif not selected:
            log_info(
                f"DatasetEvaluator: no problem cases found — all scores pass the "
                f"selection criterion (metric='{self._metric.name}', "
                f"mean={mean_score:.3f})"
            )
        elif len(selected) == len(all_results):
            log_warning(
                f"DatasetEvaluator: every case ({len(all_results)}) was selected as "
                f"'problematic' — overfitting risk if reflection uses only these cases. "
                f"Consider reviewing the threshold or dataset coverage."
            )

        return DatasetEvaluationReport(
            metric_name=self._metric.name,
            strategy=self._strategy,
            all_results=all_results,
            selected_cases=selected,
            mean_score=mean_score,
            min_score=min_score,
            max_score=max_score,
        )

    async def _evaluate_case(
        self, idx: int, case: DataCase, context_factory: ContextFactory
    ) -> CaseEvaluationResult:
        """Run + score a single case. Never raises — a failed run becomes a
        ``success=False`` result with score 0.0 (so parallel gather stays intact)."""
        case_label = case.label or f"#{idx + 1}:{case.input[:30]}"
        context = context_factory(case)
        try:
            result = await self._chain.execute_async(context)
            output = result.get_final_output() if result.success else ""
            if result.success and output:
                try:
                    # Case-aware dispatch: metrics that declare a ``case``
                    # parameter receive the current DataCase; others fall
                    # back to the legacy ``compute_async(output)`` signature.
                    score = await call_metric_async(self._metric, result, case=case)
                except Exception as metric_exc:
                    log_warning(
                        f"DatasetEvaluator: metric '{self._metric.name}' failed "
                        f"for case '{case_label}': {metric_exc}"
                    )
                    score = 0.0
            else:
                score = 0.0

            # Per-step outcomes for the failure heatmap.
            step_outcomes: dict[int, str] = {}
            step_metrics: dict[int, dict[str, float]] = {}
            step_latencies_ms: dict[int, float] = {}
            for sr in result.step_results:
                if sr.skipped:
                    step_outcomes[sr.step_number] = "skipped"
                elif sr.success:
                    step_outcomes[sr.step_number] = "success"
                else:
                    step_outcomes[sr.step_number] = "failure"
                if sr.metrics:
                    step_metrics[sr.step_number] = dict(sr.metrics)
                if sr.execution_time is not None:
                    step_latencies_ms[sr.step_number] = float(sr.execution_time) * 1000.0

            return CaseEvaluationResult(
                case=case,
                score=score,
                chain_output=output,
                success=result.success,
                execution_time=result.total_execution_time,
                token_usage=dict(result.token_usage) if result.token_usage else {},
                llm_calls=len(result.token_usage_by_step),
                step_outcomes=step_outcomes,
                step_metrics=step_metrics,
                step_latencies_ms=step_latencies_ms,
            )
        except Exception as exc:
            log_warning(f"DatasetEvaluator: chain execution failed for case '{case_label}': {exc}")
            return CaseEvaluationResult(
                case=case, score=0.0, chain_output="", success=False, execution_time=None
            )

    def evaluate(
        self,
        context_factory: ContextFactory,
    ) -> DatasetEvaluationReport:
        """
        Synchronous wrapper around :meth:`evaluate_async`.

        Mirrors the pattern used by :meth:`~mmar_carl.chain.ReasoningChain.execute`
        and :meth:`~mmar_carl.chain.ReasoningChain.reflect`.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.evaluate_async(context_factory))
        else:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, self.evaluate_async(context_factory)
                )
                return future.result()
