import json
from pathlib import Path
from models import generate_sql, run_db
from claude_integration import generate_sql_claude
from utils.classifier import classify_level
from utils.RAG_setup import summarize_schema, get_schema_safe
import time
import random

text2sql_path = Path(__file__).parent.parent
spider_db_dir_path = text2sql_path.parent / "spider" / "database"
spider_dir_path = text2sql_path.parent / "spider"


def run_spider_benchmark(args):
    print(f"example_type: {args.strategy}")
    examples_path = spider_dir_path / "evaluation_examples" / "examples"
    dev_json_path = examples_path  / "dev.json"
    tables_json_path = examples_path / "tables.json"

    if not dev_json_path.exists():
        raise FileNotFoundError(f"Spider dev.json not found at {dev_json_path}")
    if not tables_json_path.exists():
        raise FileNotFoundError(f"Spider tables.json not found at {tables_json_path}")
    
    with open(dev_json_path, "r") as f:
        dev_data = json.load(f)

    predictions = []
    results = []

    print(f"Starting Spider benchmark on {args.batch} examples .... ")
    start_time = time.time()
    random.seed(88)
    batch = random.sample(dev_data, args.batch)
    # RELOAD_COUNT = 108
    for idx, example in enumerate(batch, 1):
        question = example["question"]
        db_id = example["db_id"]
        gold_sql = example["query"]
        db_path = spider_db_dir_path / db_id / f"{db_id}.sqlite"

        if not db_path.exists():
            print(f"Warning: DB db_id = {db_id} not found")
            continue
        schema = get_schema_safe(db_id)
        summarized_schema = summarize_schema(schema)
        if args.model == 'sonnet':
            predicted_sql = generate_sql_claude(question,
                                                schema,
                                                args)
        else:# Generate SQL 
            predicted_sql = generate_sql(question,
                                     schema,
                                     args,
                                     f"sqlite:///{db_path}")
        
        # print(f"[{idx}] Generated: {predicted_sql}")
        level, counts = classify_level(gold_sql)
        # print(f"*** predicted sql: {predicted_sql}")
        # print(f"[{idx}] Running DB...")
        try:
            predicted_result = run_db(predicted_sql, f"sqlite:///{db_path}")
            # print(f"[{idx}] Result: {predicted_result}")

            predictions.append(predicted_sql)

            results.append({
                "question": question,
                "schema": summarized_schema,
                "predicted_sql": predicted_sql,
                "predicted_result": predicted_result,
                "gold_sql": gold_sql,
                "level": level,
                "db_id": db_id,
                "success": True
            })
            print(f"Success: {idx} / {args.batch}")
        
        except Exception as e:
            # print(f"Error on example {idx}: {str(e)}")
            # print(f"Error on {idx}: ({type(e).__name__})")
            predictions.append(predicted_sql) # fallback
            results.append({
                "question": question,
                "schema": summarized_schema,
                "predicted_sql": predicted_sql,
                "predicted_result": None,
                "gold_sql": gold_sql,
                "level": None,
                "db_id": db_id,
                "success": False,
                "error": str(e)
            })
            print(f"Failed: {idx} / {args.batch}")        
        
        if idx % 10 == 0:
            print(f"Progress: {idx} / {args.batch}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    output_dir = Path(__file__).parent.parent / "output" / f"{args.model}_{args.batch}_k-{args.k_examples}"
    pred_file = output_dir / f"pred-{args.strategy}.sql"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_file, "w") as f:
        f.write("\n".join(predictions))
    
    results_file = output_dir / f"predictions-{args.strategy}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark complete!")
    print(f"Predictions saved to {pred_file}")
    print(f"Detailed results saved to {results_file}")
    
    # 단순 sql 실행 성공률 (정확성 XX)
    # 정확도는 여기서 만들어진 sql 문으로
    success_count = sum(1 for r in results if r["success"])
    print(f"Success rate: {success_count}/{args.batch} ({success_count/args.batch*100:.1f}%)")
    print(f"Total Execution Time: {int(elapsed_time//60)}분 {elapsed_time%60:.2f}초")
    
    return {
        "total": args.batch,
        "success": success_count,
        "failed": args.batch - success_count,
        "results": results
    }
