from typing import Dict, Any, List, Optional
import json


class AgentTools:

    def __init__(self, model_type: str = "qwen", db_id: str = None, db_path: str = None):
        self.model_type = model_type
        self.db_id = db_id
        self.db_path = db_path
        self.model = self._load_model()        

    def _load_model(self):
        if self.model_type == "qwen":
            from models import get_llm
            return get_llm("qwen")
        
        if self.model_type == "sonnet":
            from claude_integration import get_claude_client
            return get_claude_client()
        
    def get_db_schema(self):
        from utils.RAG_setup import summarize_schema, get_schema_safe

        schema = get_schema_safe(self.db_id)
        summary_schema = summarize_schema(schema)

        return {
            "structured": schema,
            "summary": summary_schema
        }
    
    def search_similar_examples(self, question: str, k: int = 3):
        from utils.jaccard import retrieve_jaccard_examples
        return retrieve_jaccard_examples(question, k)
    
    def execute_sql(self, sql: str):
        from models import run_db
        try:
            result = run_db(sql, f"sqlite:///{self.db_path}")

            return {
                "result": result,
                "success": True
            }

        except Exception as e:
            from agent.states import classify_error

            error_msg = str(e)
            error_type = classify_error(error_msg)

            return {
                "error": error_msg,
                "error_type": error_type.value,
                "success": False
            }