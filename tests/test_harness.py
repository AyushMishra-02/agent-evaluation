"""
Unit and Integration Tests for Agent Evaluation Harness.
Verifies trajectory recording, failure taxonomy auto-tagging,
and reliability metrics (pass@1, pass@k, pass^k, variance).
"""

import pytest
from agent_eval_harness.environment import ToolEnvironment, ToolDefinition
from agent_eval_harness.evaluators import CompositeEvaluator, ExactOrContainsEvaluator
from agent_eval_harness.metrics import ReliabilityMetricsCalculator, compute_pass_at_m
from agent_eval_harness.models import (
    ExecutionStatus,
    RunEvaluation,
    StepType,
    Task,
    TokenUsage,
    ToolCall,
    ToolObservation,
    Trajectory,
    TrajectoryStep,
)
from agent_eval_harness.taxonomy import FailureTagger, FailureType
from agent_eval_harness.agents.react import ReActAgent
from agent_eval_harness.agents.plan_and_execute import PlanAndExecuteAgent
from agent_eval_harness.agents.reflection import ReflectionAgent
from agent_eval_harness.tasks.sample_suite import build_benchmark_environment, get_benchmark_tasks
from agent_eval_harness.runner import HarnessRunner


def test_metrics_pass_pow_k_strictness():
    """Verify pass^k is 1.0 iff all k runs succeed, and 0.0 if any run fails (flakiness detection)."""
    # 5 runs: 4 pass, 1 fail
    runs = [
        RunEvaluation("T1", "r1", "Agent", True, 1.0),
        RunEvaluation("T1", "r2", "Agent", True, 1.0),
        RunEvaluation("T1", "r3", "Agent", True, 1.0),
        RunEvaluation("T1", "r4", "Agent", True, 1.0),
        RunEvaluation("T1", "r5", "Agent", False, 0.0),
    ]
    report = ReliabilityMetricsCalculator.calculate_task_report("T1", "Agent", runs)
    assert report.pass_at_1 == 0.8  # 80% pass@1
    assert report.pass_pow_k == 0.0  # 0% consistency because of 1 failed run!
    assert report.flakiness == 1.0
    assert report.variance > 0.0

    # 5 runs: 5 pass
    all_pass = [RunEvaluation("T2", f"r{i}", "Agent", True, 1.0) for i in range(5)]
    rep_pass = ReliabilityMetricsCalculator.calculate_task_report("T2", "Agent", all_pass)
    assert rep_pass.pass_at_1 == 1.0
    assert rep_pass.pass_pow_k == 1.0
    assert rep_pass.flakiness == 0.0
    assert rep_pass.variance == 0.0


def test_taxonomy_infinite_loop_detection():
    """Verify that repeated consecutive calls trigger INFINITE_LOOP."""
    tagger = FailureTagger(loop_repetition_threshold=2)
    task = Task("T_LOOP", "Query data", required_tools=["tool_a"])
    traj = Trajectory("T_LOOP", "r1", "TestAgent", status=ExecutionStatus.FAILED)

    step1 = TrajectoryStep(1, StepType.ACTION, tool_calls=[ToolCall("tool_a", {"id": 1}, "c1")])
    step2 = TrajectoryStep(2, StepType.ACTION, tool_calls=[ToolCall("tool_a", {"id": 1}, "c2")])
    traj.add_step(step1)
    traj.add_step(step2)

    tags = tagger.tag_trajectory(task, traj, is_success=False)
    tag_types = [t[0] for t in tags]
    assert FailureType.INFINITE_LOOP in tag_types


def test_taxonomy_bad_arguments_detection():
    """Verify that schema validation failures trigger BAD_ARGUMENTS."""
    tagger = FailureTagger()
    task = Task("T_ARGS", "Query", required_tools=["tool_a"])
    traj = Trajectory("T_ARGS", "r1", "TestAgent", status=ExecutionStatus.FAILED)

    obs = ToolObservation("c1", "tool_a", "Validation Error: Missing required parameter 'query'", is_error=True)
    step = TrajectoryStep(1, StepType.ACTION, tool_calls=[ToolCall("tool_a", {}, "c1")], observations=[obs])
    traj.add_step(step)

    tags = tagger.tag_trajectory(task, traj, is_success=False)
    tag_types = [t[0] for t in tags]
    assert FailureType.BAD_ARGUMENTS in tag_types


def test_taxonomy_wrong_tool_detection():
    """Verify that calling forbidden or unlisted tools triggers WRONG_TOOL."""
    tagger = FailureTagger()
    task = Task("T_WRONG", "Query", available_tools=["tool_a"], forbidden_tools=["forbidden_tool"])
    traj = Trajectory("T_WRONG", "r1", "TestAgent", status=ExecutionStatus.FAILED)

    step = TrajectoryStep(1, StepType.ACTION, tool_calls=[ToolCall("forbidden_tool", {}, "c1")])
    traj.add_step(step)

    tags = tagger.tag_trajectory(task, traj, is_success=False)
    tag_types = [t[0] for t in tags]
    assert FailureType.WRONG_TOOL in tag_types


def test_end_to_end_runner(tmp_path):
    """Verify full end-to-end runner on mini suite."""
    runner = HarnessRunner(output_dir=str(tmp_path))
    tasks = get_benchmark_tasks()[:2]
    agent = ReflectionAgent(name="Reflection-Test", max_iterations=4)

    report = runner.run_suite(
        suite_name="test_suite",
        agent=agent,
        tasks=tasks,
        env_factory=build_benchmark_environment,
        k=2,
        verbose=False,
    )

    assert report.total_runs == 4
    assert report.suite_name == "test_suite"
    assert "TASK_01_DB_QUERY" in report.task_reports
