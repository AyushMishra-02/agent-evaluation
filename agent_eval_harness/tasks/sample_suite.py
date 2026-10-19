"""
Benchmark Task Suite and Mock Environment Setup.
Provides realistic agent tasks designed to expose:
- Infinite loops
- Bad arguments / schema violations
- Ignoring observations
- Premature terminations
- Hallucinated tool outputs
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple

from agent_eval_harness.environment import ToolDefinition, ToolEnvironment
from agent_eval_harness.models import Task


def build_benchmark_environment() -> ToolEnvironment:
    """Builds a sandboxed ToolEnvironment with standard mocked tools."""
    env = ToolEnvironment(state={"inventory": {"A-901": 25}, "manifest": [], "vault": "SECRET_KEY_9921_X"})

    # 1. Database Query Tool
    def handle_db(customer_id: int, table: str = "customers") -> Dict[str, Any]:
        if customer_id == 1042:
            return {"customer_id": 1042, "name": "Alice Smith", "balance": "$4,250.00", "status": "active"}
        return {"error": f"Customer {customer_id} not found."}

    env.register_tool(
        ToolDefinition(
            name="query_database",
            description="Query the SQL database for customer records.",
            parameters={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer", "description": "Customer ID number."},
                    "table": {"type": "string", "description": "Table name."},
                },
                "required": ["customer_id"],
            },
            handler=handle_db,
        )
    )

    # 2. Forex API with simulated flakiness
    call_counts: Dict[str, int] = {"forex": 0}

    def handle_forex(pair: str) -> Dict[str, Any]:
        call_counts["forex"] += 1
        if call_counts["forex"] % 2 == 1:
            # Simulated transient error to test if agent can adapt or loops/ignores
            raise RuntimeError("API Rate Limit: 429 Too Many Requests. Please back off and retry.")
        return {"pair": pair, "rate": 1.085, "timestamp": "2026-10-19T10:00:00Z"}

    env.register_tool(
        ToolDefinition(
            name="fetch_forex_rate",
            description="Fetch current forex exchange rate for a currency pair.",
            parameters={
                "type": "object",
                "properties": {"pair": {"type": "string", "description": "Currency pair symbol like EUR/USD."}},
                "required": ["pair"],
            },
            handler=handle_forex,
        )
    )

    # 3. Inventory & Manifest Tools
    def handle_inventory(sku: str, env: ToolEnvironment) -> Dict[str, Any]:
        count = env.state.get("inventory", {}).get(sku, 0)
        return {"sku": sku, "in_stock": count > 0, "quantity": count}

    env.register_tool(
        ToolDefinition(
            name="check_inventory",
            description="Check warehouse inventory level for a given SKU.",
            parameters={
                "type": "object",
                "properties": {"sku": {"type": "string", "description": "Stock Keeping Unit code."}},
                "required": ["sku"],
            },
            handler=handle_inventory,
        )
    )

    def handle_manifest(sku: str, quantity: int, env: ToolEnvironment) -> Dict[str, Any]:
        env.state["manifest"].append({"sku": sku, "quantity": quantity})
        return {"status": "success", "manifest_id": "MNF-8812"}

    env.register_tool(
        ToolDefinition(
            name="update_manifest",
            description="Add an item to the shipping manifest.",
            parameters={
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "SKU to ship."},
                    "quantity": {"type": "integer", "description": "Quantity to ship."},
                },
                "required": ["sku", "quantity"],
            },
            handler=handle_manifest,
        )
    )

    # 4. Strict Deployment Tool
    def handle_deploy(service_name: str, memory_limit_mb: int, replicas: int, env_vars: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "deployment_id": f"dep_{service_name}",
            "status": "deployed",
            "allocated_ram": memory_limit_mb,
            "replicas": replicas,
        }

    env.register_tool(
        ToolDefinition(
            name="deploy_service",
            description="Deploy a container service with strict configuration parameters.",
            parameters={
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                    "memory_limit_mb": {"type": "integer"},
                    "replicas": {"type": "integer"},
                    "env_vars": {"type": "object"},
                },
                "required": ["service_name", "memory_limit_mb", "replicas", "env_vars"],
            },
            handler=handle_deploy,
        )
    )

    # 5. Vault Tool
    def handle_vault(secret_name: str, env: ToolEnvironment) -> Dict[str, Any]:
        if secret_name == "auth_token":
            return {"secret": env.state.get("vault")}
        return {"error": "Secret not found"}

    env.register_tool(
        ToolDefinition(
            name="read_vault",
            description="Read decrypted credentials from secure vault.",
            parameters={
                "type": "object",
                "properties": {"secret_name": {"type": "string"}},
                "required": ["secret_name"],
            },
            handler=handle_vault,
        )
    )

    return env


def get_benchmark_tasks() -> List[Task]:
    """Returns standard suite of benchmark tasks."""
    return [
        Task(
            task_id="TASK_01_DB_QUERY",
            instruction="Query the database for customer ID 1042 and output their balance.",
            expected_output="$4,250.00",
            required_tools=["query_database"],
            available_tools=["query_database"],
            metadata={"arg_customer_id": 1042, "arg_table": "customers"},
        ),
        Task(
            task_id="TASK_02_API_RETRY",
            instruction="Fetch latest forex exchange rate for EUR/USD. Handle any rate-limiting gracefully.",
            expected_output="1.085",
            required_tools=["fetch_forex_rate"],
            available_tools=["fetch_forex_rate"],
            metadata={"arg_pair": "EUR/USD"},
        ),
        Task(
            task_id="TASK_03_MULTI_STEP",
            instruction="Check inventory for SKU 'A-901' and add 1 unit to the shipping manifest.",
            expected_output="MNF-8812",
            required_tools=["check_inventory", "update_manifest"],
            available_tools=["check_inventory", "update_manifest"],
            metadata={"arg_sku": "A-901", "arg_quantity": 1},
        ),
        Task(
            task_id="TASK_04_SCHEMA_STRICTNESS",
            instruction="Deploy service 'auth-api' with memory_limit_mb=1024, replicas=3, env_vars={'ENV': 'prod'}.",
            expected_output="dep_auth-api",
            required_tools=["deploy_service"],
            available_tools=["deploy_service"],
            metadata={
                "arg_service_name": "auth-api",
                "arg_memory_limit_mb": 1024,
                "arg_replicas": 3,
                "arg_env_vars": {"ENV": "prod"},
            },
        ),
        Task(
            task_id="TASK_05_VAULT_RETRIEVAL",
            instruction="Retrieve auth_token from the vault. Output the raw token value.",
            expected_output="SECRET_KEY_9921_X",
            required_tools=["read_vault"],
            forbidden_tools=["query_database"],
            available_tools=["read_vault"],
            metadata={"arg_secret_name": "auth_token"},
        ),
    ]
