"""
Scoring Evaluators and Deterministic Assertion Engines.
Evaluates final answers, tool call patterns, and environment state.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from agent_eval_harness.environment import ToolEnvironment
from agent_eval_harness.models import Task, Trajectory


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        task: Task,
        trajectory: Trajectory,
        env: Optional[ToolEnvironment] = None,
    ) -> Tuple[bool, float, Dict[str, bool], str]:
        """
        Returns:
            is_success (bool): Whether the run satisfied the success criteria.
            score (float): Normalized score between 0.0 and 1.0.
            assertions (Dict[str, bool]): Individual named assertion outcomes.
            feedback (str): Diagnostic notes or error description.
        """
        pass


class ExactOrContainsEvaluator(BaseEvaluator):
    """
    Checks if the final answer matches or contains the expected string or pattern.
    """

    def evaluate(
        self,
        task: Task,
        trajectory: Trajectory,
        env: Optional[ToolEnvironment] = None,
    ) -> Tuple[bool, float, Dict[str, bool], str]:
        if not trajectory.final_answer:
            return False, 0.0, {"has_final_answer": False}, "No final answer was provided by agent."

        assertions: Dict[str, bool] = {"has_final_answer": True}
        expected = task.expected_output

        if expected is None:
            # If no expected output defined, passing depends on non-empty answer and no fatal crashes
            return True, 1.0, assertions, "Completed without specific answer assertions."

        ans_norm = trajectory.final_answer.strip().lower()
        exp_norm = expected.strip().lower()

        is_exact = ans_norm == exp_norm
        is_contains = exp_norm in ans_norm

        assertions["matches_expected"] = is_exact or is_contains

        if assertions["matches_expected"]:
            score = 1.0 if is_exact else 0.95
            return True, score, assertions, "Final answer successfully satisfied expected output."
        else:
            return (
                False,
                0.0,
                assertions,
                f"Expected '{expected}' but agent produced: '{trajectory.final_answer[:150]}'",
            )


class ToolTrajectoryEvaluator(BaseEvaluator):
    """
    Validates that:
    1. All task.required_tools were called at least once.
    2. Zero task.forbidden_tools were called.
    3. Tool calls did not exit with uncaught fatal errors.
    """

    def evaluate(
        self,
        task: Task,
        trajectory: Trajectory,
        env: Optional[ToolEnvironment] = None,
    ) -> Tuple[bool, float, Dict[str, bool], str]:
        all_calls = trajectory.all_tool_calls()
        called_names = {c.name for c in all_calls}

        assertions: Dict[str, bool] = {}

        # 1. Check required tools invoked
        missing_required = set(task.required_tools) - called_names
        assertions["required_tools_invoked"] = len(missing_required) == 0

        # Check required tools succeeded (at least one non-error observation per required tool)
        successful_tools = {obs.tool_name for obs in trajectory.all_observations() if not obs.is_error}
        unsuccessful_required = set(task.required_tools) - successful_tools
        assertions["required_tools_succeeded"] = len(unsuccessful_required) == 0

        # 2. Check forbidden tools
        used_forbidden = set(task.forbidden_tools).intersection(called_names)
        assertions["forbidden_tools_avoided"] = len(used_forbidden) == 0

        # 3. Check for successful observations
        errors = trajectory.all_errors()
        assertions["zero_fatal_tool_errors"] = len(errors) == 0

        is_success = (
            assertions["required_tools_invoked"]
            and assertions["required_tools_succeeded"]
            and assertions["forbidden_tools_avoided"]
        )
        score = 1.0 if is_success else 0.0

        feedback = []
        if missing_required:
            feedback.append(f"Missing required tool calls: {list(missing_required)}")
        if unsuccessful_required:
            feedback.append(f"Required tool(s) failed without recovery: {list(unsuccessful_required)}")
        if used_forbidden:
            feedback.append(f"Called forbidden tool(s): {list(used_forbidden)}")
        if errors:
            feedback.append(f"Encountered {len(errors)} tool error(s).")

        return is_success, score, assertions, "; ".join(feedback) if feedback else "Tool constraints met."


class StateAssertionEvaluator(BaseEvaluator):
    """
    Validates that the final sandbox environment state satisfies task.golden_state.
    """

    def evaluate(
        self,
        task: Task,
        trajectory: Trajectory,
        env: Optional[ToolEnvironment] = None,
    ) -> Tuple[bool, float, Dict[str, bool], str]:
        if not task.golden_state:
            return True, 1.0, {"no_state_constraints": True}, "No golden state specified."

        if env is None:
            return False, 0.0, {"env_present": False}, "ToolEnvironment missing for state evaluation."

        assertions: Dict[str, bool] = {}
        mismatches: List[str] = []

        for key, expected_val in task.golden_state.items():
            actual_val = env.state.get(key)
            match = actual_val == expected_val
            assertions[f"state_{key}"] = match
            if not match:
                mismatches.append(f"state['{key}'] expected {expected_val!r}, got {actual_val!r}")

        is_success = len(mismatches) == 0
        score = 1.0 if is_success else max(0.0, (len(assertions) - len(mismatches)) / len(assertions))
        feedback = "; ".join(mismatches) if mismatches else "Environment state matches golden state."
        return is_success, score, assertions, feedback


class CompositeEvaluator(BaseEvaluator):
    """
    Combines output, trajectory, and state evaluators into a unified score.
    """

    def __init__(self, evaluators: Optional[List[BaseEvaluator]] = None):
        self.evaluators = evaluators or [
            ExactOrContainsEvaluator(),
            ToolTrajectoryEvaluator(),
            StateAssertionEvaluator(),
        ]

    def evaluate(
        self,
        task: Task,
        trajectory: Trajectory,
        env: Optional[ToolEnvironment] = None,
    ) -> Tuple[bool, float, Dict[str, bool], str]:
        all_passed = True
        scores: List[float] = []
        all_assertions: Dict[str, bool] = {}
        all_feedback: List[str] = []

        for ev in self.evaluators:
            passed, score, assertions, fb = ev.evaluate(task, trajectory, env)
            all_passed = all_passed and passed
            scores.append(score)
            all_assertions.update(assertions)
            if fb:
                all_feedback.append(fb)

        mean_score = sum(scores) / len(scores) if scores else 0.0
        return all_passed, mean_score, all_assertions, " | ".join(all_feedback)


class EvaluatorRegistry:
    @staticmethod
    def get(evaluator_type: str) -> BaseEvaluator:
        if evaluator_type == "tool_trajectory":
            return ToolTrajectoryEvaluator()
        elif evaluator_type == "state":
            return StateAssertionEvaluator()
        elif evaluator_type == "composite":
            return CompositeEvaluator()
        else:
            return CompositeEvaluator()
