"""
Plan-and-Execute Agent Archetype.
First produces an explicit multi-step plan, then dispatches subtasks sequentially,
and finally synthesizes the overall output.
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


class PlanAndExecuteAgent(BaseAgent):
    """
    Plan-and-Execute Agent architecture.
    Known real-world failure modes:
    - High plan brittleness: fails to adapt if intermediate tool returns an unexpected error.
    - Premature termination: if step 1 fails, executor sometimes declares completion without retrying.
    - Hallucinated synthesis: when tools fail to return values, final synthesis may invent missing facts.
    """

    def __init__(
        self,
        name: str = "Plan-and-Execute-Baseline",
        max_iterations: int = 10,
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

        step_idx = 1
        start_time = time.time()

        # Phase 1: Planning Step
        plan_items = []
        if task.required_tools:
            for t in task.required_tools:
                plan_items.append(f"Execute {t} to gather information.")
        elif task.available_tools:
            plan_items.append(f"Call {task.available_tools[0]} to inspect state.")
        plan_items.append("Synthesize results and return final answer.")

        plan_content = "Generated Plan:\n" + "\n".join([f"{i+1}. {p}" for i, p in enumerate(plan_items)])
        plan_step = TrajectoryStep(
            step_index=step_idx,
            step_type=StepType.PLAN,
            content=plan_content,
            token_usage=TokenUsage(prompt_tokens=320, completion_tokens=90, total_tokens=410),
            latency_ms=(time.time() - start_time) * 1000.0 + rng.uniform(80, 140),
        )
        trajectory.add_step(plan_step)

        # Phase 2: Sequential Execution
        tools_to_run = task.required_tools if task.required_tools else task.available_tools[:1]
        intermediate_failed = False

        for tool_name in tools_to_run:
            step_idx += 1
            if step_idx > min(self.max_iterations, task.max_steps):
                break

            step_start = time.time()
            tool_args = self._derive_args(task, tool_name, env, rng)
            action_call = ToolCall(name=tool_name, args=tool_args, call_id=f"plan_exec_{step_idx}")

            step = TrajectoryStep(
                step_index=step_idx,
                step_type=StepType.ACTION,
                content=f"Executing subtask: Calling {tool_name}",
                tool_calls=[action_call],
                token_usage=TokenUsage(prompt_tokens=280, completion_tokens=40, total_tokens=320),
            )
            obs = env.execute_tool(action_call.name, action_call.args, action_call.call_id)
            step.observations.append(obs)
            step.latency_ms = (time.time() - step_start) * 1000.0 + obs.latency_ms + rng.uniform(50, 90)
            trajectory.add_step(step)

            if obs.is_error:
                intermediate_failed = True
                # Plan-and-execute pathology: 30% chance of premature termination on error
                if rng.random() < 0.30:
                    break

        # Phase 3: Final Synthesis
        step_idx += 1
        synth_start = time.time()

        if intermediate_failed:
            # Plan-and-execute pathology: 40% chance of hallucinating tool output
            if rng.random() < 0.40:
                answer = task.expected_output or "Data successfully extracted via tools."
                # Hallucination tag will trigger if tool failed or was not called
            else:
                answer = "Failed to complete plan due to upstream tool errors."
        else:
            answer = task.expected_output or "Task concluded following plan."

        final_step = TrajectoryStep(
            step_index=step_idx,
            step_type=StepType.FINAL_ANSWER,
            content=f"Final Answer: {answer}",
            token_usage=TokenUsage(prompt_tokens=450, completion_tokens=70, total_tokens=520),
            latency_ms=(time.time() - synth_start) * 1000.0 + rng.uniform(70, 120),
        )
        trajectory.add_step(final_step)
        trajectory.final_answer = answer
        trajectory.status = ExecutionStatus.SUCCESS if not intermediate_failed else ExecutionStatus.FAILED
        return trajectory

    def _derive_args(self, task: Task, tool_name: str, env: ToolEnvironment, rng: random.Random) -> Dict[str, Any]:
        tool_def = env.tools.get(tool_name)
        if not tool_def:
            return {}

        props = tool_def.parameters.get("properties", {})
        args: Dict[str, Any] = {}

        # 8% chance of bad argument
        has_bad_arg = rng.random() < 0.08

        for param, spec in props.items():
            expected_type = spec.get("type", "string")
            if has_bad_arg and param in tool_def.parameters.get("required", []):
                args[param] = None  # Missing / null argument error
                has_bad_arg = False
            else:
                if expected_type == "string":
                    args[param] = task.metadata.get(f"arg_{param}", f"valid_{param}")
                elif expected_type in ("integer", "number"):
                    args[param] = int(task.metadata.get(f"arg_{param}", 42))
                elif expected_type == "boolean":
                    args[param] = True
                elif expected_type == "array":
                    args[param] = ["item_a"]
                elif expected_type == "object":
                    args[param] = {"status": "ok"}
        return args
