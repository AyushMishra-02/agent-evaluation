"""
Harness Runner: Orchestrates Agent Trajectory Execution, Repeated Trials,
Automatic Failure Tagging, and Reliability Report Generation.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional

from agent_eval_harness.agents.base import BaseAgent
from agent_eval_harness.environment import ToolEnvironment
from agent_eval_harness.evaluators import BaseEvaluator, EvaluatorRegistry
from agent_eval_harness.metrics import ReliabilityMetricsCalculator
from agent_eval_harness.models import (
    RunEvaluation,
    SuiteEvaluationReport,
    Task,
    TaskReliabilityReport,
    Trajectory,
)
from agent_eval_harness.taxonomy import FailureTagger, FailureType


class HarnessRunner:
    """
    Core Evaluation Engine that runs agent suites across k repeated trials,
    records comprehensive trajectories, auto-tags failure taxonomy modes,
    and aggregates consistency & variance metrics.
    """

    def __init__(
        self,
        output_dir: str = "eval_results",
        tagger: Optional[FailureTagger] = None,
        evaluator: Optional[BaseEvaluator] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tagger = tagger or FailureTagger()
        self.evaluator = evaluator or EvaluatorRegistry.get("composite")

    def run_suite(
        self,
        suite_name: str,
        agent: BaseAgent,
        tasks: List[Task],
        env_factory: Callable[[], ToolEnvironment],
        k: int = 5,
        base_seed: int = 1000,
        verbose: bool = True,
    ) -> SuiteEvaluationReport:
        """
        Runs the full evaluation benchmark for an agent across all tasks,
        executing each task k times to measure pass@1, pass@k, pass^k, and variance.
        """
        agent_dir = self.output_dir / suite_name / agent.name
        agent_dir.mkdir(parents=True, exist_ok=True)
        trajectories_file = agent_dir / "trajectories.jsonl"

        task_reports: Dict[str, TaskReliabilityReport] = {}

        if verbose:
            print(f"\n=======================================================")
            print(f"[*] Evaluating Agent: {agent.name}")
            print(f"[-] Suite: {suite_name} | Tasks: {len(tasks)} | Iterations (k): {k}")
            print(f"=======================================================")

        with open(trajectories_file, "a", encoding="utf-8") as traj_out:
            for task_idx, task in enumerate(tasks, 1):
                if verbose:
                    print(f"\n[{task_idx}/{len(tasks)}] Task: {task.task_id} ('{task.instruction[:60]}...')")

                runs_for_task: List[RunEvaluation] = []

                for trial in range(1, k + 1):
                    run_id = f"{task.task_id}_r{trial}_{int(time.time())}"
                    run_seed = base_seed + (task_idx * 100) + trial

                    # Initialize isolated fresh environment sandbox
                    env = env_factory()

                    # Execute Agent and capture full trajectory
                    trajectory: Trajectory = agent.run(task, env, run_id=run_id, seed=run_seed)

                    # Score Trajectory
                    is_success, score, assertions, feedback = self.evaluator.evaluate(task, trajectory, env)

                    # Automated Failure Taxonomy Classification
                    tagged_failures = self.tagger.tag_trajectory(task, trajectory, is_success)
                    failure_tags = [ft.value for ft, _ in tagged_failures]

                    run_eval = RunEvaluation(
                        task_id=task.task_id,
                        run_id=run_id,
                        agent_name=agent.name,
                        is_success=is_success,
                        score=score,
                        failure_tags=failure_tags,
                        feedback=feedback,
                        assertions=assertions,
                        trajectory=trajectory,
                    )
                    runs_for_task.append(run_eval)

                    # Stream trajectory record to JSONL disk
                    traj_record = {
                        "eval": {
                            "is_success": is_success,
                            "score": score,
                            "failure_tags": failure_tags,
                            "feedback": feedback,
                        },
                        "trajectory": trajectory.to_dict(),
                    }
                    traj_out.write(json.dumps(traj_record) + "\n")
                    traj_out.flush()

                    if verbose:
                        status_glyph = "[PASS]" if is_success else f"[FAIL] ({','.join(failure_tags) or 'incorrect'})"
                        print(f"  * Run {trial}/{k}: {status_glyph} | Score: {score:.2f} | Latency: {trajectory.total_latency_ms:.0f}ms | Tokens: {trajectory.total_tokens.total_tokens}")

                # Calculate task-level reliability report across k runs
                task_report = ReliabilityMetricsCalculator.calculate_task_report(
                    task_id=task.task_id,
                    agent_name=agent.name,
                    runs=runs_for_task,
                )
                task_reports[task.task_id] = task_report

                if verbose:
                    print(f"  [+] Task Summary: pass@1={task_report.pass_at_1*100:.0f}% | pass^{k}={task_report.pass_pow_k*100:.0f}% | Var={task_report.variance:.3f} | Flaky={task_report.flakiness > 0}")

        # Aggregate across all tasks into suite report
        suite_report = ReliabilityMetricsCalculator.calculate_suite_report(
            suite_name=suite_name,
            agent_name=agent.name,
            task_reports=task_reports,
        )

        # Save summary report to JSON
        summary_path = agent_dir / "summary_report.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(suite_report.to_dict(), f, indent=2)

        if verbose:
            print(f"\nSaved trajectories to: {trajectories_file}")
            print(f"Saved suite report to: {summary_path}")

        return suite_report
