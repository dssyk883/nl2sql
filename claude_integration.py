from anthropic import Anthropic
from dotenv import load_dotenv
import os

from utils.RAG_setup import summarize_schema
from models import create_examples, extract_sql

load_dotenv()

def get_claude_client():
    return Anthropic(api_key=os.getenv("ANTHROPIC_API"))

claude_client = get_claude_client()


top_k = 5
K = 5

NOLIMIT_PREFIX="You are a SQLite expert. Learn these natural languages to SQL examples."
K0_PREFIX="You are a SQLite expert."
SUFFIX="""
Now, given the following question, generate the correct SQL query:

Critical Rules:
1. If a table/column is not in the schema above, you CANNOT use it
2. Check spelling carefully (case-sensitive)
3. Do NOT use common sense - use ONLY what's in the schema
4. Return ONLY the SQL query

Question: 
"""

def generate_sql_claude(question: str, schema: str, args):
    """
    Generate sql using Claude API
    
    :param question: NL question
    :type question: str
    :param schema: DB schema info
    :type schema: str
 
    :return: SQL Query string
    :rtype: str
    """
    schema_summary = summarize_schema(schema)
    examples = create_examples(question, schema_summary, args)
    examples = format_claude_examples(examples)
    prompt = create_prompt(question, schema_summary, args, examples)

    try:
        message = claude_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2048,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        sql_response = message.content[0].text.strip()
        sql = extract_sql(sql_response)

        return sql
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        raise



def format_claude_examples(examples):
    """
    Format claude few shot examples with a default one
    
    :param examples: List of dictionaries: [{input: , query: }]
    :return formatted: Formatted examples as: Example #, Question:, SQL:
    """
    formatted = "Few-shot Examples:\n\n"
    
    for i, example in enumerate(examples, 1):
        formatted += f"Example {i}:\n"
        formatted += f"Question: {example['input']}\n"
        formatted += f"SQL: {example['query']}\n\n"
    
    return formatted

def create_prompt(question: str, schema: str, args, examples):
    if args.k_examples > 0:
        prefix = NOLIMIT_PREFIX
    else:
        prefix = K0_PREFIX
    suffix = SUFFIX
    prompt = f"""{prefix}

    {examples}

    Database schema:
    {schema}

    {suffix}    
    {question}
    SQL Query:"""

    return prompt