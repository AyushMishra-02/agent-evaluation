"""
Core Data Models for Agent Trajectory Recording, Tasks, and Evaluation.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import time
from typing import Any, Dict, List, Optional


class StepType(str, Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    PLAN = "plan"
    CRITIQUE = "critique"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    TIMED_OUT = "timed_out"
    AGENT_CRASH = "agent_crash"


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    call_id: str
    raw_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolObservation:
    call_id: str
    tool_name: str
    output: Any
    is_error: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: TokenUsage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryStep:
    step_index: int
    step_type: StepType
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    observations: List[ToolObservation] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "step_type": self.step_type.value if isinstance(self.step_type, StepType) else self.step_type,
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "observations": [obs.to_dict() for obs in self.observations],
            "token_usage": self.token_usage.to_dict(),
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class Trajectory:
    task_id: str
    run_id: str
    agent_name: str
    status: ExecutionStatus = ExecutionStatus.FAILED
    steps: List[TrajectoryStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    total_latency_ms: float = 0.0
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: TrajectoryStep) -> None:
        self.steps.append(step)
        self.total_latency_ms += step.latency_ms
        self.total_tokens.add(step.token_usage)

    def all_tool_calls(self) -> List[ToolCall]:
        calls: List[ToolCall] = []
        for step in self.steps:
            calls.extend(step.tool_calls)
        return calls

    def all_observations(self) -> List[ToolObservation]:
        observations: List[ToolObservation] = []
        for step in self.steps:
            observations.extend(step.observations)
        return observations

    def all_errors(self) -> List[ToolObservation]:
        return [obs for obs in self.all_observations() if obs.is_error]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "status": self.status.value if isinstance(self.status, ExecutionStatus) else self.status,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens.to_dict(),
            "metadata": self.metadata,
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class Task:
    task_id: str
    instruction: str
    expected_output: Optional[str] = None
    golden_state: Optional[Dict[str, Any]] = None
    required_tools: List[str] = field(default_factory=list)
    forbidden_tools: List[str] = field(default_factory=list)
    available_tools: List[str] = field(default_factory=list)
    max_steps: int = 10
    timeout_seconds: float = 30.0
    evaluator_type: str = "exact_or_contains"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunEvaluation:
    task_id: str
    run_id: str
    agent_name: str
    is_success: bool
    score: float  # Normalized 0.0 to 1.0
    failure_tags: List[str] = field(default_factory=list)
    feedback: str = ""
    assertions: Dict[str, bool] = field(default_factory=dict)
    trajectory: Optional[Trajectory] = None

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "is_success": self.is_success,
            "score": self.score,
            "failure_tags": self.failure_tags,
            "feedback": self.feedback,
            "assertions": self.assertions,
        }
        if self.trajectory:
            res["trajectory"] = self.trajectory.to_dict()
        return res


@dataclass
class TaskReliabilityReport:
    task_id: str
    agent_name: str
    k: int
    pass_at_1: float
    pass_at_k: float
    pass_pow_k: float  # pass^k: 1.0 if all k runs passed, else 0.0
    mean_score: float
    variance: float
    std_dev: float
    flakiness: float  # 1.0 if not strictly 0.0 or 1.0 (flaky task)
    failure_tag_counts: Dict[str, int] = field(default_factory=dict)
    mean_latency_ms: float = 0.0
    mean_tokens: float = 0.0
    runs: List[RunEvaluation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "k": self.k,
            "pass_at_1": self.pass_at_1,
            "pass_at_k": self.pass_at_k,
            "pass_pow_k": self.pass_pow_k,
            "mean_score": self.mean_score,
            "variance": self.variance,
            "std_dev": self.std_dev,
            "flakiness": self.flakiness,
            "failure_tag_counts": self.failure_tag_counts,
            "mean_latency_ms": self.mean_latency_ms,
            "mean_tokens": self.mean_tokens,
            "runs": [r.to_dict() for r in self.runs],
        }


@dataclass
class SuiteEvaluationReport:
    suite_name: str
    agent_name: str
    k: int
    num_tasks: int
    total_runs: int
    overall_pass_at_1: float
    overall_pass_at_k: float
    overall_pass_pow_k: float  # Consistency metric across all tasks
    average_variance: float
    flaky_task_ratio: float  # % of tasks with non-deterministic outcome
    failure_tag_distribution: Dict[str, float]  # % of total failures
    mean_latency_ms: float
    mean_tokens: float
    task_reports: Dict[str, TaskReliabilityReport] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "agent_name": self.agent_name,
            "k": self.k,
            "num_tasks": self.num_tasks,
            "total_runs": self.total_runs,
            "overall_pass_at_1": self.overall_pass_at_1,
            "overall_pass_at_k": self.overall_pass_at_k,
            "overall_pass_pow_k": self.overall_pass_pow_k,
            "average_variance": self.average_variance,
            "flaky_task_ratio": self.flaky_task_ratio,
            "failure_tag_distribution": self.failure_tag_distribution,
            "mean_latency_ms": self.mean_latency_ms,
            "mean_tokens": self.mean_tokens,
            "task_reports": {tid: rep.to_dict() for tid, rep in self.task_reports.items()},
        }
