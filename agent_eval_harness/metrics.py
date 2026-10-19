"""
Reliability Metrics Engine.
Computes pass@1, pass@k, pass^k (consistency across all k runs),
task-level sample variance, flakiness index, and failure taxonomy distributions.
"""

from __future__ import annotations
import math
import statistics
from typing import Dict, List, Optional, Tuple

from agent_eval_harness.models import (
    RunEvaluation,
    SuiteEvaluationReport,
    TaskReliabilityReport,
)


def compute_pass_at_m(n: int, c: int, m: int) -> float:
    """
    Unbiased estimator for pass@m from Chen et al. (HumanEval):
    pass@m = 1 - binom(n - c, m) / binom(n, m)
    where:
      n = total runs
      c = correct runs
      m = sample size to estimate (m <= n)
    """
    if n - c < m:
        return 1.0
    if c == 0:
        return 0.0
    # 1 - prod_{j=0}^{m-1} (n - c - j) / (n - j)
    prob_all_wrong = 1.0
    for j in range(m):
        prob_all_wrong *= (n - c - j) / (n - j)
    return 1.0 - prob_all_wrong


class ReliabilityMetricsCalculator:
    """
    Aggregates repeated evaluations of tasks and computes statistical reliability metrics:
    - pass@1: Average pass rate on any single run.
    - pass@k: Probability of at least 1 success in k attempts.
    - pass^k: Strict consistency across all k runs (pass^k = 1.0 iff all k runs pass).
    - Variance & Standard Deviation across repeated attempts.
    - Flakiness Ratio: proportion of tasks with non-deterministic outcomes (0 < pass < 1).
    - Failure Taxonomy Distribution: breakdown of root causes for all failed runs.
    """

    @classmethod
    def calculate_task_report(
        cls,
        task_id: str,
        agent_name: str,
        runs: List[RunEvaluation],
    ) -> TaskReliabilityReport:
        k = len(runs)
        if k == 0:
            raise ValueError(f"Cannot calculate metrics for task {task_id} with 0 runs.")

        scores = [1.0 if r.is_success else 0.0 for r in runs]
        success_count = sum(1 for r in runs if r.is_success)

        mean_score = sum(scores) / k
        pass_at_1 = mean_score
        pass_at_k = compute_pass_at_m(n=k, c=success_count, m=k) if k > 0 else 0.0

        # pass^k: Strict consistency (all k runs must succeed)
        pass_pow_k = 1.0 if success_count == k else 0.0

        # Sample variance: s^2 = sum(x - mean)^2 / (k - 1)
        if k > 1:
            variance = statistics.variance(scores)
            std_dev = statistics.stdev(scores)
        else:
            variance = 0.0
            std_dev = 0.0

        # Flakiness flag: 1.0 if task has inconsistent results, 0.0 if deterministic
        flakiness = 1.0 if (0 < success_count < k) else 0.0

        # Failure tag frequency counts
        tag_counts: Dict[str, int] = {}
        for r in runs:
            if not r.is_success:
                for tag in r.failure_tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Resource consumption
        latencies = [r.trajectory.total_latency_ms for r in runs if r.trajectory]
        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

        tokens = [r.trajectory.total_tokens.total_tokens for r in runs if r.trajectory]
        mean_tokens = sum(tokens) / len(tokens) if tokens else 0.0

        return TaskReliabilityReport(
            task_id=task_id,
            agent_name=agent_name,
            k=k,
            pass_at_1=pass_at_1,
            pass_at_k=pass_at_k,
            pass_pow_k=pass_pow_k,
            mean_score=mean_score,
            variance=variance,
            std_dev=std_dev,
            flakiness=flakiness,
            failure_tag_counts=tag_counts,
            mean_latency_ms=mean_latency,
            mean_tokens=mean_tokens,
            runs=runs,
        )

    @classmethod
    def calculate_suite_report(
        cls,
        suite_name: str,
        agent_name: str,
        task_reports: Dict[str, TaskReliabilityReport],
    ) -> SuiteEvaluationReport:
        num_tasks = len(task_reports)
        if num_tasks == 0:
            raise ValueError(f"Cannot calculate suite report with 0 tasks.")

        total_runs = sum(rep.k for rep in task_reports.values())
        k = next(iter(task_reports.values())).k

        overall_pass_at_1 = sum(rep.pass_at_1 for rep in task_reports.values()) / num_tasks
        overall_pass_at_k = sum(rep.pass_at_k for rep in task_reports.values()) / num_tasks

        # Flagship pass^k: Fraction of tasks that succeeded consistently across EVERY run
        overall_pass_pow_k = sum(rep.pass_pow_k for rep in task_reports.values()) / num_tasks

        average_variance = sum(rep.variance for rep in task_reports.values()) / num_tasks

        flaky_tasks = sum(1 for rep in task_reports.values() if rep.flakiness > 0.0)
        flaky_task_ratio = flaky_tasks / num_tasks

        # Overall failure distribution
        aggregated_tag_counts: Dict[str, int] = {}
        total_failures = 0
        for rep in task_reports.values():
            for r in rep.runs:
                if not r.is_success:
                    total_failures += 1
            for tag, count in rep.failure_tag_counts.items():
                aggregated_tag_counts[tag] = aggregated_tag_counts.get(tag, 0) + count

        failure_distribution: Dict[str, float] = {}
        if total_failures > 0:
            for tag, count in aggregated_tag_counts.items():
                failure_distribution[tag] = round((count / total_failures) * 100, 2)

        mean_latency = sum(rep.mean_latency_ms for rep in task_reports.values()) / num_tasks
        mean_tokens = sum(rep.mean_tokens for rep in task_reports.values()) / num_tasks

        return SuiteEvaluationReport(
            suite_name=suite_name,
            agent_name=agent_name,
            k=k,
            num_tasks=num_tasks,
            total_runs=total_runs,
            overall_pass_at_1=round(overall_pass_at_1, 4),
            overall_pass_at_k=round(overall_pass_at_k, 4),
            overall_pass_pow_k=round(overall_pass_pow_k, 4),
            average_variance=round(average_variance, 4),
            flaky_task_ratio=round(flaky_task_ratio, 4),
            failure_tag_distribution=failure_distribution,
            mean_latency_ms=round(mean_latency, 2),
            mean_tokens=round(mean_tokens, 2),
            task_reports=task_reports,
        )
