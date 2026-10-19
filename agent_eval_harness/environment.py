"""
Tool Environment and Execution Sandbox.
Provides tool registry, schema validation, simulated latency,
and error handling for reproducible agent evaluations.
"""

from __future__ import annotations
import inspect
import time
from typing import Any, Callable, Dict, List, Optional
import uuid

from agent_eval_harness.models import ToolCall, ToolObservation


class ToolDefinition:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Any],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        """Validate input arguments against parameter specification."""
        required = self.parameters.get("required", [])
        for req in required:
            if req not in args:
                return f"Missing required parameter '{req}' for tool '{self.name}'."

        properties = self.parameters.get("properties", {})
        for k, val in args.items():
            if k not in properties and not self.parameters.get("additionalProperties", True):
                return f"Unexpected parameter '{k}' provided to tool '{self.name}'."

            expected_type = properties.get(k, {}).get("type")
            if expected_type == "integer" and not isinstance(val, int):
                return f"Parameter '{k}' must be an integer, got {type(val).__name__}."
            elif expected_type == "number" and not isinstance(val, (int, float)):
                return f"Parameter '{k}' must be a number, got {type(val).__name__}."
            elif expected_type == "string" and not isinstance(val, str):
                return f"Parameter '{k}' must be a string, got {type(val).__name__}."
            elif expected_type == "boolean" and not isinstance(val, bool):
                return f"Parameter '{k}' must be a boolean, got {type(val).__name__}."
            elif expected_type == "array" and not isinstance(val, list):
                return f"Parameter '{k}' must be a list/array, got {type(val).__name__}."
            elif expected_type == "object" and not isinstance(val, dict):
                return f"Parameter '{k}' must be an object/dict, got {type(val).__name__}."
        return None

    def to_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolEnvironment:
    """
    Execution sandbox managing available tools and state for a single evaluation run.
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        self.tools: Dict[str, ToolDefinition] = {}
        self.state: Dict[str, Any] = state or {}
        self.history: List[ToolObservation] = []

    def register_tool(self, tool: ToolDefinition) -> None:
        self.tools[tool.name] = tool

    def get_tool_schemas(self, allowed_tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        schemas = []
        for name, tool in self.tools.items():
            if allowed_tools is None or name in allowed_tools:
                schemas.append(tool.to_schema())
        return schemas

    def execute_tool(
        self,
        name: str,
        args: Dict[str, Any],
        call_id: Optional[str] = None,
    ) -> ToolObservation:
        call_id = call_id or str(uuid.uuid4())[:8]
        start_time = time.time()

        if name not in self.tools:
            latency = (time.time() - start_time) * 1000.0
            obs = ToolObservation(
                call_id=call_id,
                tool_name=name,
                output=f"Error: Tool '{name}' does not exist in available tools.",
                is_error=True,
                latency_ms=latency,
            )
            self.history.append(obs)
            return obs

        tool = self.tools[name]
        validation_err = tool.validate_args(args)
        if validation_err:
            latency = (time.time() - start_time) * 1000.0
            obs = ToolObservation(
                call_id=call_id,
                tool_name=name,
                output=f"Validation Error: {validation_err}",
                is_error=True,
                latency_ms=latency,
            )
            self.history.append(obs)
            return obs

        try:
            # Check if handler expects environment state
            sig = inspect.signature(tool.handler)
            if "env" in sig.parameters:
                result = tool.handler(env=self, **args)
            elif "state" in sig.parameters:
                result = tool.handler(state=self.state, **args)
            else:
                result = tool.handler(**args)

            latency = (time.time() - start_time) * 1000.0
            obs = ToolObservation(
                call_id=call_id,
                tool_name=name,
                output=result,
                is_error=False,
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.time() - start_time) * 1000.0
            obs = ToolObservation(
                call_id=call_id,
                tool_name=name,
                output=f"Execution Error in {name}: {type(e).__name__}: {str(e)}",
                is_error=True,
                latency_ms=latency,
            )

        self.history.append(obs)
        return obs
