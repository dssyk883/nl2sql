import json
from pathlib import Path
from models import generate_sql, run_db
from utils.classifier import classify_level
import time

spider_db_dir_path = Path('/home/sykwon/spider/database')
spider_dir_path = Path('/home/sykwon/spider')

def run_spider_benchmark(batch_size:int = 100, example_type:str = ""):
    print(f"example_type: {example_type}")
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

    print(f"Starting Spider benchmark on {batch_size} examples .... ")
    start_time = time.time()
    for idx, example in enumerate(dev_data[:batch_size], 1):
        question = example["question"]
        db_id = example["db_id"]
        gold_sql = example["query"]
        db_path = spider_db_dir_path / db_id / f"{db_id}.sqlite"

        if not db_path:
            print(f"Warning: DB db_id = {db_id} not found")
            continue
        
        # print(f"Trying to access: {db_path}")
        # print(f"File exists: {db_path.exists()}")

        predicted_sql = generate_sql(question, example_type, f"sqlite:///{db_path}", use_limit=False)
        level = classify_level(gold_sql)

        try:
            predicted_result = run_db(predicted_sql, f"sqlite:///{db_path}")

            predictions.append(predicted_sql)

            results.append({
                "question": question,
                "predicted_sql": predicted_sql,
                "predicted_result": predicted_result,
                "gold_sql": gold_sql,
                "level": level,
                "db_id": db_id,
                "success": True
            })

            if idx % 10 == 0:
                print(f"Progress: {idx} / {batch_size}")
        
        except Exception as e:
            print(f"Error on example {idx}: {str(e)}")
            print(f"({type(e).__name__})")
            predictions.append("SELECT * LIMIT 1") # fallback
            results.append({
                "question": question,
                "predicted_sql": predicted_sql,
                "predicted_result": None,
                "gold_sql": gold_sql,
                "level": level,
                "db_id": db_id,
                "success": False,
                "error": str(e)
            })
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    output_dir = Path(__file__).parent
    pred_file = output_dir / "pred.sql"

    with open(pred_file, "w") as f:
        f.write("\n".join(predictions))
    
    results_file = output_dir/"predictions.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark complete!")
    print(f"Predictions saved to {pred_file}")
    print(f"Detailed results saved to {results_file}")
    
    # 단순 sql 실행 성공률 (정확성 XX)
    # 정확도는 여기서 만들어진 sql 문으로
    success_count = sum(1 for r in results if r["success"])
    print(f"Success rate: {success_count}/{batch_size} ({success_count/batch_size*100:.1f}%)")
    print(f"Total Execution Time: {int(elapsed_time//60)}분 {elapsed_time%60:.2f}초")
    
    return {
        "total": batch_size,
        "success": success_count,
        "failed": batch_size - success_count,
        "results": results
    }