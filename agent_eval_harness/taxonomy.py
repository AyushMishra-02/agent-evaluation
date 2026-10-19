"""
Failure Taxonomy and Automated Trajectory Tagger.
Detects and tags failure modes deterministically across agent runs:
- WRONG_TOOL
- BAD_ARGUMENTS
- INFINITE_LOOP
- HALLUCINATED_TOOL_OUTPUT
- PREMATURE_TERMINATION
- IGNORING_OBSERVATIONS
- STEP_BUDGET_EXCEEDED
"""

from __future__ import annotations
from enum import Enum
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from agent_eval_harness.models import (
    ExecutionStatus,
    StepType,
    Task,
    ToolCall,
    ToolObservation,
    Trajectory,
)


class FailureType(str, Enum):
    WRONG_TOOL = "wrong_tool"
    BAD_ARGUMENTS = "bad_arguments"
    INFINITE_LOOP = "infinite_loop"
    HALLUCINATED_TOOL_OUTPUT = "hallucinated_tool_output"
    PREMATURE_TERMINATION = "premature_termination"
    IGNORING_OBSERVATIONS = "ignoring_observations"
    STEP_BUDGET_EXCEEDED = "step_budget_exceeded"
    AGENT_CRASH = "agent_crash"


class FailureTagger:
    """
    Deterministic rule-based analyzer that examines agent trajectories
    and classifies the root cause(s) of failure according to the failure taxonomy.
    """

    def __init__(
        self,
        loop_repetition_threshold: int = 2,
        tool_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Args:
            loop_repetition_threshold: Number of identical consecutive tool calls
                                      or 2-cycle repetitions to consider an infinite loop.
            tool_schemas: Optional dictionary mapping tool names to schema definitions
                          (e.g., {'required_args': ['query', 'limit'], 'arg_types': {...}}).
        """
        self.loop_repetition_threshold = loop_repetition_threshold
        self.tool_schemas = tool_schemas or {}

    def tag_trajectory(
        self,
        task: Task,
        trajectory: Trajectory,
        is_success: bool,
    ) -> List[Tuple[FailureType, str]]:
        """
        Analyze a trajectory and return a list of (FailureType, diagnostic_message).
        If the run was successful and no severe behavioral violation occurred,
        returns an empty list.
        """
        if is_success:
            return []

        tags: List[Tuple[FailureType, str]] = []
        applied_types: Set[FailureType] = set()

        def add_tag(ft: FailureType, reason: str):
            if ft not in applied_types:
                tags.append((ft, reason))
                applied_types.add(ft)

        # 1. Agent Crash / Unhandled Exception
        if trajectory.status == ExecutionStatus.AGENT_CRASH:
            add_tag(FailureType.AGENT_CRASH, "Agent runtime terminated unexpectedly with an unhandled exception.")

        # 2. Step Budget Exceeded
        if trajectory.status == ExecutionStatus.MAX_STEPS_EXCEEDED or len(trajectory.steps) >= task.max_steps:
            add_tag(
                FailureType.STEP_BUDGET_EXCEEDED,
                f"Trajectory reached or exceeded max step budget ({task.max_steps} steps) without reaching completion.",
            )

        all_calls = trajectory.all_tool_calls()
        available_tools = set(task.available_tools)
        forbidden_tools = set(task.forbidden_tools)

        # 3. WRONG_TOOL Detection
        # Check if agent called tools not in available_tools, forbidden tools, or made up tool names
        for tc in all_calls:
            if forbidden_tools and tc.name in forbidden_tools:
                add_tag(
                    FailureType.WRONG_TOOL,
                    f"Agent invoked forbidden tool '{tc.name}' which is explicitly disallowed for task {task.task_id}.",
                )
            elif available_tools and tc.name not in available_tools:
                add_tag(
                    FailureType.WRONG_TOOL,
                    f"Agent invoked unregistered or unavailable tool '{tc.name}'. Available: {list(available_tools)}",
                )

        # 4. BAD_ARGUMENTS Detection
        # Check tool observations for argument schema errors, missing fields, or validation failures
        for step in trajectory.steps:
            for obs in step.observations:
                if obs.is_error:
                    err_msg = str(obs.output).lower()
                    if any(kw in err_msg for kw in ["missing argument", "invalid argument", "schema validation", "unexpected keyword", "type error", "validation error", "required key"]):
                        add_tag(
                            FailureType.BAD_ARGUMENTS,
                            f"Tool '{obs.tool_name}' failed with argument validation error: {obs.output}",
                        )

            # Also check against provided tool schemas if available
            for tc in step.tool_calls:
                if tc.name in self.tool_schemas:
                    schema = self.tool_schemas[tc.name]
                    required_args = schema.get("required_args", [])
                    missing = [arg for arg in required_args if arg not in tc.args]
                    if missing:
                        add_tag(
                            FailureType.BAD_ARGUMENTS,
                            f"Tool call '{tc.name}' missing required parameters: {missing}. Passed: {list(tc.args.keys())}",
                        )

        # 5. INFINITE_LOOP Detection
        # Check for:
        # a) Repeated identical consecutive tool calls (name + exact args) >= threshold
        # b) Alternating 2-cycle loops (A -> B -> A -> B)
        call_signatures = [self._call_signature(c) for c in all_calls]
        if self._detect_consecutive_loop(call_signatures, self.loop_repetition_threshold):
            add_tag(
                FailureType.INFINITE_LOOP,
                f"Agent called identical tool signature consecutively {self.loop_repetition_threshold}+ times without progress.",
            )
        elif self._detect_alternating_loop(call_signatures):
            add_tag(
                FailureType.INFINITE_LOOP,
                "Agent entered an alternating cyclic loop (A -> B -> A -> B) repeating identical calls.",
            )

        # 6. IGNORING_OBSERVATIONS Detection
        # Check if step i returned an error observation, and step i+1 invoked the exact same tool with identical args
        for i in range(len(trajectory.steps) - 1):
            curr_step = trajectory.steps[i]
            next_step = trajectory.steps[i + 1]

            error_obs = [obs for obs in curr_step.observations if obs.is_error]
            if error_obs and curr_step.tool_calls and next_step.tool_calls:
                curr_sig = self._call_signature(curr_step.tool_calls[-1])
                next_sig = self._call_signature(next_step.tool_calls[0])
                if curr_sig == next_sig:
                    add_tag(
                        FailureType.IGNORING_OBSERVATIONS,
                        f"Step {i+1} received error '{error_obs[0].output}' from tool '{curr_step.tool_calls[-1].name}', "
                        f"but step {i+2} repeated the identical call without modifying arguments.",
                    )

        # 7. PREMATURE_TERMINATION Detection
        # Check if the agent finished with a final answer, but:
        # a) Required tools were never executed
        # b) Output states "I cannot", "As an AI", "I am unable" despite tools being available
        # c) Agent finished in step 0 or 1 without attempting necessary interactions
        called_tool_names = {c.name for c in all_calls}
        missing_required = set(task.required_tools) - called_tool_names
        if missing_required:
            add_tag(
                FailureType.PREMATURE_TERMINATION,
                f"Agent concluded without calling mandatory tool(s): {list(missing_required)}.",
            )

        if trajectory.final_answer:
            ans_lower = trajectory.final_answer.lower()
            refusal_patterns = [
                "i cannot assist with that",
                "i am unable to perform",
                "i don't have access to tools",
                "as an ai model, i cannot",
                "i apologize, but i cannot execute",
                "output is unknown",
                "finished partially",
                "unable to complete",
            ]
            if any(p in ans_lower for p in refusal_patterns):
                add_tag(
                    FailureType.PREMATURE_TERMINATION,
                    f"Agent prematurely terminated or declared output unknown despite available resources: '{trajectory.final_answer}'",
                )

        # 8. Contradicting or Ignoring Observation Payload in Final Answer
        # If the observation contained the expected answer/token, but the final answer missed or contradicted it
        if task.expected_output and trajectory.final_answer:
            exp_clean = task.expected_output.lower()
            obs_texts = [str(obs.output).lower() for obs in trajectory.all_observations() if not obs.is_error]
            obs_had_expected = any(exp_clean in o for o in obs_texts)
            ans_has_expected = exp_clean in trajectory.final_answer.lower()
            if obs_had_expected and not ans_has_expected:
                add_tag(
                    FailureType.IGNORING_OBSERVATIONS,
                    f"Tool observation contained the correct target '{task.expected_output}', but the agent's final answer failed to incorporate it.",
                )

        # 9. HALLUCINATED_TOOL_OUTPUT Detection
        # Check if agent fabricates tool returns in thought text or claims tool data
        # without actually running the tool, or invents observation formats.
        all_obs_text = " ".join([str(obs.output) for obs in trajectory.all_observations()]).lower()
        if trajectory.final_answer and len(all_calls) == 0 and task.required_tools:
            # If agent answered claiming concrete retrieved facts without running any tools
            add_tag(
                FailureType.HALLUCINATED_TOOL_OUTPUT,
                "Agent provided a factual final answer without calling required tools, hallucinating data source results.",
            )
        else:
            # Check thoughts for fake observation markers like "Observation: ..." generated by the agent itself
            for step in trajectory.steps:
                if step.step_type in (StepType.THOUGHT, StepType.PLAN) and step.content:
                    if re.search(r"(?:observation|tool result|returned):\s*\{.*?\}", step.content, re.IGNORECASE):
                        if not step.observations:
                            add_tag(
                                FailureType.HALLUCINATED_TOOL_OUTPUT,
                                f"Step {step.step_index} generated hallucinated inline observation block without real tool execution.",
                            )

        # Fallback tag for any failed run where specific heuristics didn't trigger
        if not tags and not is_success:
            if any(obs.is_error for obs in trajectory.all_observations()):
                add_tag(
                    FailureType.IGNORING_OBSERVATIONS,
                    "Agent encountered tool execution errors and failed to recover, terminating with an incorrect state.",
                )
            else:
                add_tag(
                    FailureType.PREMATURE_TERMINATION,
                    f"Agent concluded with an incorrect or incomplete result without satisfying task criteria: '{trajectory.final_answer}'",
                )

        return tags

    @staticmethod
    def _call_signature(tc: ToolCall) -> str:
        """Create a deterministic string representation of a tool call and its arguments."""
        try:
            sorted_args = json.dumps(tc.args, sort_keys=True)
        except Exception:
            sorted_args = str(tc.args)
        return f"{tc.name}::{sorted_args}"

    @staticmethod
    def _detect_consecutive_loop(signatures: List[str], threshold: int) -> bool:
        if len(signatures) < threshold:
            return False
        count = 1
        for i in range(1, len(signatures)):
            if signatures[i] == signatures[i - 1]:
                count += 1
                if count >= threshold:
                    return True
            else:
                count = 1
        return False

    @staticmethod
    def _detect_alternating_loop(signatures: List[str]) -> bool:
        # Detects patterns like [A, B, A, B]
        if len(signatures) < 4:
            return False
        for i in range(len(signatures) - 3):
            if signatures[i] == signatures[i + 2] and signatures[i + 1] == signatures[i + 3] and signatures[i] != signatures[i + 1]:
                return True
        return False
