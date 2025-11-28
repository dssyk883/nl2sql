from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains.sql_database.query import create_sql_query_chain
from database import DATABASE_URL
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

from pydantic import BaseModel
from pathlib import Path
import re

from utils.fixed_examples import create_fixed_examples
from utils.RAG_examples import retrieve_RAG_examples
from utils.random_examples import create_random_examples

EXAMPLE_PATH = Path(__file__).parent / "utils" / "examples.txt"
top_k = 5
K = 5
_static_prompt_cache = {}

def create_examples(question: str, example_type: str = ""):
    if not example_type: # baseline - 랜덤한 K 개의 예제
        return create_random_examples(K)

    if example_type == "fixed":
        return create_fixed_examples(K)

    if example_type == "rag":
        return retrieve_RAG_examples(question, K)

NOLIMIT_PREFIX = """Given an input quesiton, create a syntatically correct PostgreSQL query.
Critical Rules:
1. Return ONLY the sql auery, no explanations, no other lines.
2. Do NOT wrap in '''sql''' blocks
3. Only use columns from: {table_info}
(top_k : {top_k} for reference only, do not add LIMIT unless question specifies)
Examples:"""

LIMIT_PREFIX = """Given an input quesiton, create a syntatically correct PostgreSQL query.
Critical Rules:
1. Return ONLY the sql auery, no explanations, no other lines.
2. Do NOT wrap in '''sql''' blocks
3. Add "LIMIT {top_k}" at the end unless COUNT/SUM/AVG is used
4. Only use columns from: {table_info}
Examples:"""

psql_prompt = PromptTemplate(
    input_variables = ["input","query"],
    template = "Question: {input}\nSQL:{query}"
)

def create_prompt(question: str, example_type: str, use_limit: bool):
    cache_key = (example_type, use_limit)

    if cache_key in _static_prompt_cache:
        print(f"*** cache found: Using cache ... ")
        return _static_prompt_cache[cache_key]
    
    print(f"*** create_prompt: Creating examples & prompt ... ")

    examples = create_examples(question, example_type)
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=psql_prompt,
        prefix=LIMIT_PREFIX if use_limit else NOLIMIT_PREFIX,
        suffix="Question:{input}\nSQL:",
        input_variables=["input","top_k","table_info"]
    )

    if example_type != "rag":
        _static_prompt_cache[cache_key] = prompt
    
    return prompt

def get_llm(model: str):

    if model == "Ollama":
        return OllamaLLM(model="llama3.2", temperature=0)
    
    # if model == "Gpt-3.5":
    #     return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

class QueryRequest(BaseModel):
    question: str
   

def generate_sql(question: str, example_type: str, db_uri: str, use_limit: bool = True) -> tuple[str, str]:    
    llm = get_llm("Ollama")
    db = SQLDatabase.from_uri(db_uri)

    prompt = create_prompt(question, example_type, use_limit)

    chain = create_sql_query_chain(
    llm=llm,
    db=db,
    k=K,
    prompt=prompt
    )
    
    response = chain.invoke({"question": question})
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
    
    return sql
