import json
from pathlib import Path
from utils.classifier import classify_level


TRAIN_PATH = Path(__file__).parent / ".." /".." / "spider" / "evaluation_examples" / "examples" / "train_spider.json"

LEVEL_RATIO = {
    "easy": 0.1,
    "medium": 0.2,
    "hard": 0.35,
    "extra": 0.35
}


def create_fixed_examples(k: int):
    """
    Fixed Few Shot 
    k 개의 예제 생성
    각 난이도별 개수는 비율은 전역 변수 LEVEL_RATIO를 따름

    Args:
        k: 예제 개수
    """
    with open(TRAIN_PATH, "r") as f:
        train_data = json.load(f)

    targets = {level: int(k * ratio) for level, ratio in LEVEL_RATIO.items()}
    targets["extra"] += k - sum(targets.values())
    
    counts = {level: 0 for level in targets}
    examples = []

    for data in train_data:
        if len(examples) >= k:
            break
        
        lvl, _ = classify_level(data["query"])
        
        if counts[lvl] < targets[lvl]:
            examples.append({"input": data["question"], "query": data["query"]})
            counts[lvl] += 1        

    return examples