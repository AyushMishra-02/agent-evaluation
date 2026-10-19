"""
ReAct (Reasoning + Acting) Agent Archetype.
Executes an interleaved Thought -> Action -> Observation loop.
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


class ReActAgent(BaseAgent):
    """
    Classic ReAct Agent architecture.
    Known real-world failure modes:
    - Susceptible to infinite loops if tool fails or returns ambiguous results.
    - Tends to ignore observations when error messages are subtle.
    - Lower token footprint per step, but prone to drifting off-course over long horizons.
    """

    def __init__(
        self,
        name: str = "ReAct-Baseline",
        max_iterations: int = 8,
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
        consecutive_tool_failures = 0
        last_call_sig = None

        while step_idx < min(self.max_iterations, task.max_steps):
            step_idx += 1
            start_step_time = time.time()

            if self.llm_fn:
                # Custom LLM dispatch if provided
                pass  # LLM implementation hook

            # Simulated / Benchmark Archetype Logic:
            # Emulates the ReAct loop across available tools
            available = [t for t in task.available_tools if t in env.tools]

            # ReAct reasoning step
            thought = f"Step {step_idx}: Assessing goal '{task.instruction}'. Checking available tools: {available}."

            # Determine next action
            # Known pathology: ReAct has ~15-20% chance of repeating an identical tool call when stuck
            if step_idx == 1 and available:
                # ReAct pathology: 20% chance of calling forbidden or irrelevant tool if present
                if task.forbidden_tools and rng.random() < 0.20:
                    selected_tool = rng.choice(task.forbidden_tools)
                else:
                    selected_tool = available[0]

                tool_args = self._derive_args(task, selected_tool, env, rng)
                action_call = ToolCall(name=selected_tool, args=tool_args, call_id=f"call_{step_idx}")

                step = TrajectoryStep(
                    step_index=step_idx,
                    step_type=StepType.ACTION,
                    content=f"Thought: {thought}\nAction: Calling {selected_tool}",
                    tool_calls=[action_call],
                    token_usage=TokenUsage(prompt_tokens=180 + step_idx * 40, completion_tokens=45, total_tokens=225 + step_idx * 40),
                )
                obs = env.execute_tool(action_call.name, action_call.args, action_call.call_id)
                step.observations.append(obs)
                step.latency_ms = (time.time() - start_step_time) * 1000.0 + obs.latency_ms + rng.uniform(40, 90)
                trajectory.add_step(step)

                if obs.is_error:
                    consecutive_tool_failures += 1
                last_call_sig = (action_call.name, str(sorted(action_call.args.items())))

            elif step_idx > 1 and available:
                # Check if previous observation had errors
                last_step = trajectory.steps[-1]
                has_error = any(o.is_error for o in last_step.observations)

                # ReAct pathology: 35% chance of ignoring observation and repeating identical call consecutively
                repeat_bug = has_error and rng.random() < 0.35

                if repeat_bug and last_step.tool_calls:
                    prev_call = last_step.tool_calls[0]
                    # Loop consecutively
                    action_call = ToolCall(name=prev_call.name, args=prev_call.args, call_id=f"call_{step_idx}")
                    step = TrajectoryStep(
                        step_index=step_idx,
                        step_type=StepType.ACTION,
                        content=f"Thought: Retrying call despite error without changing arguments.\nAction: Calling {prev_call.name}",
                        tool_calls=[action_call],
                        token_usage=TokenUsage(prompt_tokens=240, completion_tokens=30, total_tokens=270),
                    )
                    obs = env.execute_tool(action_call.name, action_call.args, action_call.call_id)
                    step.observations.append(obs)
                    step.latency_ms = (time.time() - start_step_time) * 1000.0 + obs.latency_ms + rng.uniform(40, 80)
                    trajectory.add_step(step)
                elif len(available) > 1 and task.required_tools and not all(t in [c.name for c in trajectory.all_tool_calls()] for t in task.required_tools):
                    # Call next required tool
                    remaining = [t for t in task.required_tools if t not in [c.name for c in trajectory.all_tool_calls()]]
                    next_tool = remaining[0] if remaining else available[-1]
                    tool_args = self._derive_args(task, next_tool, env, rng)
                    action_call = ToolCall(name=next_tool, args=tool_args, call_id=f"call_{step_idx}")
                    step = TrajectoryStep(
                        step_index=step_idx,
                        step_type=StepType.ACTION,
                        content=f"Thought: Calling required tool {next_tool}.\nAction: Calling {next_tool}",
                        tool_calls=[action_call],
                        token_usage=TokenUsage(prompt_tokens=260, completion_tokens=50, total_tokens=310),
                    )
                    obs = env.execute_tool(action_call.name, action_call.args, action_call.call_id)
                    step.observations.append(obs)
                    step.latency_ms = (time.time() - start_step_time) * 1000.0 + obs.latency_ms + rng.uniform(40, 80)
                    trajectory.add_step(step)
                else:
                    # Final answer phase
                    # ReAct pathology: Premature termination on complex constraints (15% chance of misinterpreting observations)
                    flawed_answer = rng.random() < 0.15
                    if flawed_answer and task.expected_output:
                        answer = f"The process finished partially: output is unknown."
                    else:
                        answer = task.expected_output or "Task finished successfully."

                    step = TrajectoryStep(
                        step_index=step_idx,
                        step_type=StepType.FINAL_ANSWER,
                        content=f"Thought: Finalizing answer based on observations.\nFinal Answer: {answer}",
                        token_usage=TokenUsage(prompt_tokens=310, completion_tokens=60, total_tokens=370),
                    )
                    step.latency_ms = (time.time() - start_step_time) * 1000.0 + rng.uniform(50, 100)
                    trajectory.add_step(step)
                    trajectory.final_answer = answer
                    trajectory.status = ExecutionStatus.SUCCESS
                    return trajectory
            else:
                # No tools available or direct answer
                answer = task.expected_output or "Task completed."
                step = TrajectoryStep(
                    step_index=step_idx,
                    step_type=StepType.FINAL_ANSWER,
                    content=f"Final Answer: {answer}",
                    token_usage=TokenUsage(prompt_tokens=120, completion_tokens=30, total_tokens=150),
                )
                step.latency_ms = (time.time() - start_step_time) * 1000.0 + rng.uniform(30, 60)
                trajectory.add_step(step)
                trajectory.final_answer = answer
                trajectory.status = ExecutionStatus.SUCCESS
                return trajectory

        # If loop finished without final answer
        trajectory.status = ExecutionStatus.MAX_STEPS_EXCEEDED
        return trajectory

    def _derive_args(self, task: Task, tool_name: str, env: ToolEnvironment, rng: random.Random) -> Dict[str, Any]:
        """Derives arguments, occasionally testing schema resilience."""
        tool_def = env.tools.get(tool_name)
        if not tool_def:
            return {}

        props = tool_def.parameters.get("properties", {})
        args: Dict[str, Any] = {}

        # 10% chance for ReAct agent to produce a bad argument if parameter is complex
        has_bad_arg = rng.random() < 0.10

        for param, spec in props.items():
            expected_type = spec.get("type", "string")
            if has_bad_arg and param in tool_def.parameters.get("required", []):
                # Inject invalid type
                args[param] = 99999 if expected_type == "string" else "invalid_string"
                has_bad_arg = False  # only corrupt one
            else:
                if expected_type == "string":
                    args[param] = task.metadata.get(f"arg_{param}", f"sample_{param}")
                elif expected_type in ("integer", "number"):
                    args[param] = int(task.metadata.get(f"arg_{param}", 10))
                elif expected_type == "boolean":
                    args[param] = True
                elif expected_type == "array":
                    args[param] = ["item_1", "item_2"]
                elif expected_type == "object":
                    args[param] = {"key": "val"}
        return args
