from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from database import DATABASE_URL
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

from pydantic import BaseModel
from pathlib import Path
import re

from utils.fixed_examples import create_fixed_examples
from utils.RAG_examples import retrieve_RAG_examples
from utils.intent_clustering import retrieve_intent_based_examples
from utils.jaccard import retrieve_jaccard_examples
from utils.random_examples import create_random_examples
from utils.RAG_setup import summarize_schema

EXAMPLE_PATH = Path(__file__).parent / "utils" / "examples.txt"
top_k = 5
K = 5

# NOLIMIT_PREFIX = """You are a SQLite expert. Given an input question and database schema, create a syntactically correct SQLite query.
# Schema: {table_info}
# Critical Rules:
# 1. If a table/column is not in the schema above, you CANNOT use it
# 2. Check spelling carefully (case-sensitive)
# 3. Do NOT use common sense - use ONLY what's in the schema
# 4. Return ONLY the SQL query

# Study these examples ONLY for the SQL query:
# (top_k : {top_k} for reference only, do not add LIMIT unless question specifies)
# Examples:"""

K0_PREFIX = """You are a SQLite expert."""
K0_SUFFIX = """Schema: {table_info}
Question: {input}"""

NOLIMIT_PREFIX = """You are a SQLite expert. Learn these natural languages to SQL examples.
Examples:"""
NOLIMIT_SUFFIX = """
Now, given the following information, generate the correct SQL query:

Schema: {table_info}

Critical Rules:
1. If a table/column is not in the schema above, you CANNOT use it
2. Check spelling carefully (case-sensitive)
3. Do NOT use common sense - use ONLY what's in the schema
4. Return ONLY the SQL query

Question: {input}
"""

LIMIT_PREFIX = """You are a SQLite expert. Given an input question and database schema, create a syntactically correct SQLite query.
Critical Rules:
1. Return ONLY the SQL query - no explanations, no markdown blocks
2. Use ONLY the columns that exist in the schema below
3. Do NOT use schemas in examples
4. Pay attention to column types and foreign key relationships
5. Do NOT add LIMIT unless the question specifically asks for it
6. Add "LIMIT {top_k}" at the end unless COUNT/SUM/AVG/MIN/MAX is used
Examples:"""

def create_examples(question: str, schema: str, args):
    # print(f"[DEBUG] Creating Examples ... ")
    if args.strategy == "random": # baseline - 랜덤한 K 개의 예제
        # print(f"[DEBUG] Random Examples ... ")
        return create_random_examples(args.k_examples)

    if args.strategy == "fixed":
        return create_fixed_examples(args.k_examples)

    if args.strategy == "rag":
        # print(f"[DEBUG] Retrieving RAG examples...")
        return retrieve_RAG_examples(question, schema, args.k_examples)        
    
    if args.strategy == "ic":
        return retrieve_intent_based_examples(question, args.k_examples, args.cluster)

    if args.strategy == 'jacc':
        return retrieve_jaccard_examples(question, args.k_examples)


psql_prompt = PromptTemplate(
    input_variables = ["input","query"],
    template = "Question: {input}\nSQL:{query}"
)

def create_prompt(question: str, schema_summary:str, args):
    examples = create_examples(question, schema_summary, args)
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=psql_prompt,
        prefix=NOLIMIT_PREFIX if args.k_examples > 0 else K0_PREFIX,
        suffix=NOLIMIT_SUFFIX if args.k_examples > 0 else K0_SUFFIX,
        input_variables=["input","top_k","table_info"]
    )
    return prompt

def get_llm(model: str):

    if model == "mistral":
        return OllamaLLM(
            # model="mistral:7b-instruct-q4_0",
            model="mistral:7b-instruct-q5_K_M",
            temperature=0,
            streaming=False,
            verbose=True)
            # stop=['\n\n', 'Question:'])
    if model == "qwen":
        return OllamaLLM(model="qwen2.5-coder:7b",
                         temperature=0, 
                         streaming=False, 
                         verbose=True)
                        #  stop=['\n\n', 'Question:'])       
        

class QueryRequest(BaseModel):
    question: str
   
def generate_sql(question: str, schema: str, args, db_uri: str) -> tuple[str, str]:    
    # print(f"Schema: \n{schema}")
    # print(f"[DEBUG] Creating LLM...")
    llm = get_llm(args.model)
    # print(f"[DEBUG] Connecting to DB: {db_uri}")

    # print(f"[DEBUG] Creating prompt with example_type: {args.strategy}")
    prompt = create_prompt(question, schema, args)

    # print(f"[DEBUG] Creating chain...")
    schema_summary = summarize_schema(schema)
    filled_prompt = prompt.format(
        input=question,
        top_k=args.k_examples,
        table_info=schema_summary
    )

    # print("\n" + "="*80)
    # print("FULL PROMPT")
    # print("="*80)
    # print(filled_prompt)
    # print("="*80 + "\n")
    
    # print(f"[DEBUG] Invoking chain...")
    try:
        response = llm.invoke(filled_prompt)
    except Exception as e:
        print(f"Chain error: {e}")
        # Fallback SQL (LLM 재호출 안 함)
        return "SELECT * LIMIT 1"

    sql = extract_sql(response.strip())

    return sql

def run_db(sql: str, db_uri: str):
    db = SQLDatabase.from_uri(db_uri)
    return db.run(sql)

# SQL 블록 제거
def extract_sql(text: str) -> str:
    match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        sql = match.group(1).strip()
    else:
        sql = text.strip()
    
    if ';' in sql:
        sql = sql.split(';')[0].strip()
    
    sql = re.sub(r'\\(.)', r'\1', sql)
    
    return ' '.join(sql.split())