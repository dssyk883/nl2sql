import json
from pathlib import Path
import random


TRAIN_PATH = Path(__file__).parent / ".." /".." / "spider" / "evaluation_examples" / "examples" / "train_spider.json"


def create_random_examples(k: int = 3):
    with open(TRAIN_PATH, 'r') as f:
        train_data = json.load(f)
    
    samples = random.sample(train_data, k)

    return [{"input": s["question"], "query": s["query"]} for s in samples]
    