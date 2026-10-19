"""
Agent Evaluation Harness - Flagship Core
A framework for running agents against task suites, recording complete trajectories,
evaluating deterministic and semantic criteria, auto-tagging failure taxonomies,
and calculating true reliability metrics (pass@k, pass^k consistency, task variance).
"""

from agent_eval_harness.models import (
    Task,
    Trajectory,
    TrajectoryStep,
    ToolCall,
    ToolObservation,
    StepType,
    ExecutionStatus,
    RunEvaluation,
    TaskReliabilityReport,
    SuiteEvaluationReport,
)
from agent_eval_harness.taxonomy import FailureType, FailureTagger
from agent_eval_harness.metrics import ReliabilityMetricsCalculator
from agent_eval_harness.runner import HarnessRunner
from agent_eval_harness.environment import ToolEnvironment, ToolDefinition

__all__ = [
    "Task",
    "Trajectory",
    "TrajectoryStep",
    "ToolCall",
    "ToolObservation",
    "StepType",
    "ExecutionStatus",
    "RunEvaluation",
    "TaskReliabilityReport",
    "SuiteEvaluationReport",
    "FailureType",
    "FailureTagger",
    "ReliabilityMetricsCalculator",
    "HarnessRunner",
    "ToolEnvironment",
    "ToolDefinition",
]
