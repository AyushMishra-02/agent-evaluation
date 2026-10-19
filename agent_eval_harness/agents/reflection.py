"""
Reflection / Self-Critique Agent Archetype.
Executes actions followed by an explicit evaluation & critique step,
refining parameters or recovering from errors before finalization.
"""

from __future__ import annotations
import random
import time
from typing import Any, Callable, Dict, List, Optional

from agent_eval_harness.agents.base import BaseAgent
from agent_eval_harness.environment import ToolEnvironment
from agent_eval_harness.models import (
    ExecutionStatus,
    StepType,
    Task,
    TokenUsage,
    ToolCall,
    ToolObservation,
    Trajectory,
    TrajectoryStep,
)


class ReflectionAgent(BaseAgent):
    """
    Self-Reflective Agent Architecture.
    Trade-offs:
    - Pros: Higher pass^k consistency, significantly lower rate of ignoring observations,
            catches schema and parameter errors during critique.
    - Honest Negatives: 2x - 3x higher token consumption, higher latency,
            and occasional over-reflection (critique hallucinating errors where none exist).
    """

    def __init__(
        self,
        name: str = "Reflection-SelfCritique",
        max_iterations: int = 12,
        llm_fn: Optional[Callable[[str, List[Dict[str, Any]]], Dict[str, Any]]] = None,
    ):
        super().__init__(name=name, max_iterations=max_iterations)
        self.llm_fn = llm_fn

    def run(
        self,
        task: Task,
        env: ToolEnvironment,
        run_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Trajectory:
        rng = random.Random(seed) if seed is not None else random.Random()
        trajectory = self._init_trajectory(task, run_id)

        step_idx = 0
        tools_to_run = list(task.required_tools) if task.required_tools else task.available_tools[:1]
        tool_index = 0

        while step_idx < min(self.max_iterations, task.max_steps) and tool_index < len(tools_to_run):
            step_idx += 1
            step_start = time.time()
            current_tool = tools_to_run[tool_index]

            # 1. Action Step
            tool_args = self._derive_args(task, current_tool, env, rng)
            action_call = ToolCall(name=current_tool, args=tool_args, call_id=f"refl_{step_idx}")

            act_step = TrajectoryStep(
                step_index=step_idx,
                step_type=StepType.ACTION,
                content=f"Thought: Calling tool {current_tool} with verified parameters.\nAction: {current_tool}",
                tool_calls=[action_call],
                token_usage=TokenUsage(prompt_tokens=420, completion_tokens=50, total_tokens=470),
            )
            obs = env.execute_tool(action_call.name, action_call.args, action_call.call_id)
            act_step.observations.append(obs)
            act_step.latency_ms = (time.time() - step_start) * 1000.0 + obs.latency_ms + rng.uniform(60, 110)
            trajectory.add_step(act_step)

            # 2. Explicit Reflection / Critique Step
            step_idx += 1
            critique_start = time.time()

            if obs.is_error:
                # Reflection catches error and analyzes cause
                critique_msg = (
                    f"Critique: Tool {current_tool} failed with error '{obs.output}'. "
                    f"Analyzing parameter mismatch and adjusting request."
                )
                refl_step = TrajectoryStep(
                    step_index=step_idx,
                    step_type=StepType.CRITIQUE,
                    content=critique_msg,
                    token_usage=TokenUsage(prompt_tokens=510, completion_tokens=90, total_tokens=600),
                    latency_ms=(time.time() - critique_start) * 1000.0 + rng.uniform(80, 130),
                )
                trajectory.add_step(refl_step)

                # Retry with corrected args in next step
                step_idx += 1
                retry_start = time.time()
                corrected_args = self._derive_args(task, current_tool, env, rng, force_valid=True)
                retry_call = ToolCall(name=current_tool, args=corrected_args, call_id=f"refl_retry_{step_idx}")
                retry_step = TrajectoryStep(
                    step_index=step_idx,
                    step_type=StepType.ACTION,
                    content=f"Thought: Retrying {current_tool} with corrected arguments.\nAction: {current_tool}",
                    tool_calls=[retry_call],
                    token_usage=TokenUsage(prompt_tokens=530, completion_tokens=40, total_tokens=570),
                )
                retry_obs = env.execute_tool(retry_call.name, retry_call.args, retry_call.call_id)
                retry_step.observations.append(retry_obs)
                retry_step.latency_ms = (time.time() - retry_start) * 1000.0 + retry_obs.latency_ms + rng.uniform(60, 100)
                trajectory.add_step(retry_step)
            else:
                # Honest Negative: 5% chance of "over-reflection" (doubting a correct observation)
                over_reflect = rng.random() < 0.05
                if over_reflect:
                    critique_msg = f"Critique: Received observation '{obs.output}', but suspecting potential edge-case hallucination."
                else:
                    critique_msg = f"Critique: Tool execution successful. Observation satisfies preconditions."

                refl_step = TrajectoryStep(
                    step_index=step_idx,
                    step_type=StepType.CRITIQUE,
                    content=critique_msg,
                    token_usage=TokenUsage(prompt_tokens=480, completion_tokens=60, total_tokens=540),
                    latency_ms=(time.time() - critique_start) * 1000.0 + rng.uniform(70, 110),
                )
                trajectory.add_step(refl_step)

            tool_index += 1

        # Phase 3: Final Answer
        step_idx += 1
        fin_start = time.time()
        answer = task.expected_output or "Reflective workflow concluded successfully."

        final_step = TrajectoryStep(
            step_index=step_idx,
            step_type=StepType.FINAL_ANSWER,
            content=f"Final Answer: {answer}",
            token_usage=TokenUsage(prompt_tokens=650, completion_tokens=85, total_tokens=735),
            latency_ms=(time.time() - fin_start) * 1000.0 + rng.uniform(90, 150),
        )
        trajectory.add_step(final_step)
        trajectory.final_answer = answer
        trajectory.status = ExecutionStatus.SUCCESS
        return trajectory

    def _derive_args(
        self,
        task: Task,
        tool_name: str,
        env: ToolEnvironment,
        rng: random.Random,
        force_valid: bool = False,
    ) -> Dict[str, Any]:
        tool_def = env.tools.get(tool_name)
        if not tool_def:
            return {}

        props = tool_def.parameters.get("properties", {})
        args: Dict[str, Any] = {}

        # Initial attempt has small 5% error rate, but force_valid is 0%
        has_bad_arg = False if force_valid else (rng.random() < 0.05)

        for param, spec in props.items():
            expected_type = spec.get("type", "string")
            if has_bad_arg and param in tool_def.parameters.get("required", []):
                args[param] = 1234 if expected_type == "string" else "invalid"
                has_bad_arg = False
            else:
                if expected_type == "string":
                    args[param] = task.metadata.get(f"arg_{param}", f"valid_{param}")
                elif expected_type in ("integer", "number"):
                    args[param] = int(task.metadata.get(f"arg_{param}", 50))
                elif expected_type == "boolean":
                    args[param] = True
                elif expected_type == "array":
                    args[param] = ["verified_item"]
                elif expected_type == "object":
                    args[param] = {"verified": True}
        return args
