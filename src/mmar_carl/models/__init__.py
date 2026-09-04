"""
Core data models for CARL reasoning system.

This module re-exports all models for backward compatibility.
For new code, prefer importing from the specific submodules:
- mmar_carl.models.enums: StepType, MemoryOperation, Language
- mmar_carl.models.base: SearchStrategy, SelfCriticDecision, SelfCriticEvaluatorBase
- mmar_carl.models.llm_client_base: SearchStrategy, SelfCriticDecision, SelfCriticEvaluatorBase
- mmar_carl.models.search: SubstringSearchStrategy, VectorSearchStrategy, ContextSearchConfig
- mmar_carl.models.config: ToolParameter, ToolStepConfig, MCPStepConfig, etc.
- mmar_carl.models.steps: StepDescriptionBase, LLMStepDescription, etc.
- mmar_carl.models.context: ReasoningContext
- mmar_carl.models.results: StepExecutionResult, ReasoningResult
- mmar_carl.models.prompts: PromptTemplate
"""

# Re-export all public symbols for backward compatibility
# flake8: noqa: F401

# Enums
from .enums import Language, MemoryOperation, StepType
from .artifacts import ArtifactInput, ArtifactOutput, ArtifactRecord

# Abstract base classes
from .base import SearchStrategy, SelfCriticDecision, SelfCriticEvaluatorBase
from .llm_client_base import ChatMessage, LLMClientBase
from .human_input import (
    HumanInputOutcome,
    HumanInputRequest,
    HumanInputResponse,
    HumanInputStatus,
)
from .chain_tool import ChainToolDefinition, ChainToolOutcome, ChainToolStatus
from ..code_execution import (
    CodeExecutionOutcome,
    CodeExecutionPolicy,
    CodeExecutionStatus,
    CodeRuntimeProfile,
    CodeSchemaError,
    CodeSourceError,
)

# Search strategies
from .search import ContextSearchConfig, SubstringSearchStrategy, VectorSearchStrategy

# Step configurations
from .config import (
    AgentHandoffStepConfig,
    AgentStepConfig,
    CodeStepConfig,
    ClaudeCodeStepConfig,
    CodexStepConfig,
    AfterWaitCondition,
    AnyOfWaitCondition,
    AtWaitCondition,
    CommandPlanStepConfig,
    CommandStepConfig,
    ShellSessionStepConfig,
    ConditionalBranch,
    ConditionalStepConfig,
    ContextQuery,
    EvalFailAction,
    EvaluationStepConfig,
    ExecutionMode,
    LLMStepConfig,
    LoopConfig,
    MCPServerConfig,
    MCPStepConfig,
    MCPResourceStepConfig,
    MemoryStepConfig,
    ParallelSamplingAggregation,
    ParallelSamplingStepConfig,
    StepCache,
    StepConfig,
    StepGroup,
    ToolErrorRecovery,
    ToolParameter,
    ToolStepConfig,
    TransformStepConfig,
    StructuredOutputStepConfig,
    SupervisorStepConfig,
    DebateStepConfig,
    CallableToolSource,
    DictToolSource,
    ModuleToolSource,
    HumanInputStepConfig,
    MapStepConfig,
    EventWaitCondition,
    ToolDiscoveryStepConfig,
    ToolSource,
    WaitCondition,
    WaitLeafCondition,
    WaitStepConfig,
)

# RE-PLAN models
from .replan import (
    LLMReplanCheckerConfig,
    RegisteredReplanCheckerConfig,
    ReplanAction,
    ReplanAggregationConfig,
    ReplanAggregationStrategy,
    ReplanBudgetConfig,
    ReplanCheckerBase,
    ReplanCheckerInput,
    ReplanCheckerSpec,
    ReplanPolicy,
    ReplanRollbackTarget,
    ReplanTargetType,
    ReplanTriggerConfig,
    ReplanVerdict,
    RuleBasedReplanCheckerConfig,
)

# AgentSkill models
from .agent_skill import (
    AgentSkillExecutionMode,
    AgentSkillSource,
    AgentSkillStepConfig,
    SkillManifest,
)

# Step descriptions
from .steps import (
    AgentHandoffStepDescription,
    AgentStepDescription,
    CodeStepDescription,
    ClaudeCodeStepDescription,
    CodexStepDescription,
    AgentSkillStepDescription,
    AnyStepDescription,
    CommandPlanStepDescription,
    CommandStepDescription,
    ShellSessionStepDescription,
    ConditionalStepDescription,
    EvaluationStepDescription,
    LLMStepDescription,
    MCPStepDescription,
    MCPResourceStepDescription,
    HumanInputStepDescription,
    MapStepDescription,
    WaitStepDescription,
    ToolDiscoveryStepDescription,
    MemoryStepDescription,
    ParallelSamplingStepDescription,
    StepDescription,
    StepDescriptionBase,
    ToolStepDescription,
    TransformStepDescription,
    StructuredOutputStepDescription,
    SupervisorStepDescription,
    DebateStepDescription,
    create_step,
)

# Context
from .context import ContextSnapshot, ReasoningContext
from ..command_policy import CommandApprovalRequest, CommandDecision, CommandPolicy

# Results
from .results import (
    ReasoningResult,
    ReplanAggregationOutcome,
    ReplanCheckerVote,
    ReplanEvent,
    StepExecutionResult,
)
from .result_data import (
    DebateTranscript,
    DebateTurn,
    ParallelSamples,
    MapItemOutcome,
    MapItemStatus,
    MapOutcome,
    SkillOutput,
    SupervisorDecision,
    WaitOutcome,
)

# Execution trace
from ..execution_trace import ExecutionTrace, TraceEvent

# Long-term memory
from ..ltm import InMemoryLTM, JsonFileLTM, LTMBase

# Prompts
from .prompts import PromptTemplate

# Dataset abstractions
from .dataset import (
    AbstractDataset,
    CaseEvaluationResult,
    DataCase,
    DataFrameDataset,
    DatasetEvaluationReport,
    SelectionStrategy,
    SimpleDataset,
    ThresholdStrategy,
    TopKWorstStrategy,
)

# CARE-namespace chain metadata
from .care_metadata import (
    CARE_METADATA_NAMESPACE,
    CareChainMetadata,
    CareContextFile,
)

# Preflight introspection
from .preflight import PreflightReport

# RunRecord — durable chain-execution snapshot
from .run_record import RunRecord

# Define __all__ for explicit public API
__all__ = [
    # Enums
    "StepType",
    "MemoryOperation",
    "Language",
    "HumanInputOutcome",
    "HumanInputRequest",
    "HumanInputResponse",
    "HumanInputStatus",
    "ChainToolDefinition",
    "ChainToolOutcome",
    "ChainToolStatus",
    "MapItemOutcome",
    "MapItemStatus",
    "MapOutcome",
    "CodeExecutionOutcome",
    "CodeExecutionPolicy",
    "CodeExecutionStatus",
    "CodeRuntimeProfile",
    "CodeSchemaError",
    "CodeSourceError",
    # Abstract base classes
    "ChatMessage",
    "LLMClientBase",
    "SearchStrategy",
    "SelfCriticDecision",
    "SelfCriticEvaluatorBase",
    # Search strategies
    "SubstringSearchStrategy",
    "VectorSearchStrategy",
    "ContextSearchConfig",
    # Step configurations
    "AgentHandoffStepConfig",
    "AgentStepConfig",
    "CodeStepConfig",
    "ClaudeCodeStepConfig",
    "CodexStepConfig",
    "AfterWaitCondition",
    "AtWaitCondition",
    "EventWaitCondition",
    "AnyOfWaitCondition",
    "WaitLeafCondition",
    "WaitCondition",
    "WaitStepConfig",
    "MapStepConfig",
    "ToolParameter",
    "ToolErrorRecovery",
    "ToolStepConfig",
    "MCPServerConfig",
    "MCPStepConfig",
    "MemoryStepConfig",
    "TransformStepConfig",
    "CommandPlanStepConfig",
    "CommandStepConfig",
    "ShellSessionStepConfig",
    "ArtifactInput",
    "ArtifactOutput",
    "ArtifactRecord",
    "StructuredOutputStepConfig",
    "SupervisorStepConfig",
    "DebateStepConfig",
    "ConditionalBranch",
    "ConditionalStepConfig",
    "EvalFailAction",
    "EvaluationStepConfig",
    "LoopConfig",
    "StepConfig",
    "StepGroup",
    "ContextQuery",
    "ExecutionMode",
    "LLMStepConfig",
    # RE-PLAN
    "ReplanAction",
    "ReplanTargetType",
    "ReplanRollbackTarget",
    "ReplanVerdict",
    "ReplanAggregationStrategy",
    "ReplanAggregationConfig",
    "ReplanTriggerConfig",
    "ReplanBudgetConfig",
    "RuleBasedReplanCheckerConfig",
    "LLMReplanCheckerConfig",
    "RegisteredReplanCheckerConfig",
    "ReplanCheckerSpec",
    "ReplanPolicy",
    "ReplanCheckerInput",
    "ReplanCheckerBase",
    # AgentSkill models
    "AgentSkillExecutionMode",
    "AgentSkillSource",
    "AgentSkillStepConfig",
    "SkillManifest",
    # Tool discovery
    "ModuleToolSource",
    "CallableToolSource",
    "DictToolSource",
    "ToolSource",
    "HumanInputStepConfig",
    "ToolDiscoveryStepConfig",
    # Step descriptions
    "AgentHandoffStepDescription",
    "AgentStepDescription",
    "MapStepDescription",
    "CodeStepDescription",
    "ClaudeCodeStepDescription",
    "CodexStepDescription",
    "WaitStepDescription",
    "StepDescriptionBase",
    "LLMStepDescription",
    "ToolStepDescription",
    "MCPStepDescription",
    "MCPResourceStepDescription",
    "MCPResourceStepConfig",
    "MemoryStepDescription",
    "TransformStepDescription",
    "CommandPlanStepDescription",
    "CommandStepDescription",
    "ShellSessionStepDescription",
    "StructuredOutputStepDescription",
    "SupervisorStepDescription",
    "DebateStepDescription",
    "ConditionalStepDescription",
    "EvaluationStepDescription",
    "AgentSkillStepDescription",
    "HumanInputStepDescription",
    "ToolDiscoveryStepDescription",
    "AnyStepDescription",
    "StepDescription",
    "create_step",
    # Context
    "ReasoningContext",
    "CommandApprovalRequest",
    "CommandDecision",
    "CommandPolicy",
    # Results
    "StepExecutionResult",
    "ReplanCheckerVote",
    "ReplanAggregationOutcome",
    "ReplanEvent",
    "ReasoningResult",
    "WaitOutcome",
    # Prompts
    "PromptTemplate",
    # Execution trace
    "ExecutionTrace",
    "TraceEvent",
    # Long-term memory
    "LTMBase",
    "InMemoryLTM",
    "JsonFileLTM",
    # Dataset abstractions
    "DataCase",
    "AbstractDataset",
    "SimpleDataset",
    "DataFrameDataset",
    "ThresholdStrategy",
    "TopKWorstStrategy",
    "SelectionStrategy",
    "CaseEvaluationResult",
    "DatasetEvaluationReport",
]
