"""
Prompt Builder
Generates state specific prompts for the LLM
"""

from typing import Dict, Any, List
from agent2.states import AgentState, ActionType, get_available_workers
from agent2.memory import AgentMemory

class PromptBuilder:
    EXAMPLE_OUTPUT = """
{"worker": "get_db_schema", "params": {}, "reasoning": "Need schema first", "confidence": 0.95}
{"worker": "few_shot_select", "params": {"strategy": "jaccard", "k": 5}, "reasoning": "Similar questions found", "confidence": 0.8}
{"worker": "generate_sql", "params": {}, "reasoning": "Schema loaded, ready to generate", "confidence": 0.9}
    """
    ACTION_DESCRIPTIONS = {
            ActionType.GET_DB_SCHEMA: "Get database tables and columns",
            ActionType.VALIDATE_SQL: "Validate SQL syntax",
            ActionType.GENERATE_SQL: "Generate SQL query",
            ActionType.CHECK_SEMANTIC: "Compare the question and SQL query",
            ActionType.FEW_SHOT_SELECT: "Select a few-shot example strategy",
        }

    def __init__(self, config):
        self.config = config

    def build_prompt(
        self,
        state: AgentState,
        memory: AgentMemory
    ) -> str:
        if state == AgentState.GET_DB_SCHEMA:
            return self._build_get_db_prompt(state, memory)
        elif state == AgentState.VALIDATE_SQL:
            return self._build_validate_sql_prompt(state, memory)
        elif state == AgentState.GENERATE_SQL:
            return self._build_generate_sql_prompt(state, memory)
        elif state == AgentState.CHECK_SEMANTIC:
            return self._build_check_semantic_prompt(state, memory)
        elif state == ActionType.FEW_SHOT_SELECT:
            return self._build_few_shot_prompt(state, memory)
        else:
            return ""

    def build_decision_prompt(self, state: AgentState, memory:AgentMemory) -> str:
        sql = memory.get_last_sql()
        prev_ex_result = memory.get_last_execution_result()
        schema_summary = memory.schema_summary
        last_action = memory.get_last_action()
        available_workers = get_available_workers(state)
        workers_formatted = self._format_workers(available_workers)

        return f"""You are an Agent Controller for an NL2SQL system.
Your job is to decide the NEXT worker to call.

[Question]
{memory.question}
[Current SQL]
{sql if sql else "No SQL yet"}

[Execution Result]
{prev_ex_result if prev_ex_result else "No execution result yet"}

[DB Schema Summary]
{schema_summary if schema_summary else "No DB schema loaded yet"}

[Previous Actions]
{last_action if last_action else "No action done yet"}

[Instructions]
- Choose exactly ONE next worker.
- Available workers:
{workers_formatted}

- If you choose Few-shot Selector, specify:
- strategy: Random | Intent cluster | Jaccard
- k: number of examples

- Respond ONLY in valid JSON.
- Do NOT include explanations outside JSON.

[JSON Output Format]
{{
"worker": "<worker_name>",
"params": {{}},
"reasoning": "...",
"confidence": 0.0-1.0
}}

[Example Outputs]
{self.EXAMPLE_OUTPUT}
    """

    def _format_workers(self, actions: List[ActionType]) -> str:
        formatted = []
        for i, action in enumerate(actions):
            description = self.ACTION_DESCRIPTIONS.get(action, "No description")
            formatted.append(f"{i}. {action.value}: {description}")
        
        return "\n".join(formatted)