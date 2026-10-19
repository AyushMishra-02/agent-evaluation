"""
Agent Archetype Interfaces and Base Classes.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import time
from typing import Any, Callable, Dict, List, Optional
import uuid

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


class BaseAgent(ABC):
    def __init__(self, name: str, max_iterations: int = 10):
        self.name = name
        self.max_iterations = max_iterations

    @abstractmethod
    def run(
        self,
        task: Task,
        env: ToolEnvironment,
        run_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Trajectory:
        """
        Execute the agent on the given task and environment, recording every step into a Trajectory.
        """
        pass

    def _init_trajectory(self, task: Task, run_id: Optional[str]) -> Trajectory:
        run_id = run_id or str(uuid.uuid4())[:8]
        return Trajectory(
            task_id=task.task_id,
            run_id=run_id,
            agent_name=self.name,
            metadata={"max_iterations": self.max_iterations},
        )
