import json
from pathlib import Path
from classifier import classify_hardness


train_path = Path(__file__).parent / ".." /".." / "spider" / "evaluation_examples" / "examples" / "train_spider.json"
example_path = Path(__file__).parent / "examples.txt"

with open(train_path, "r") as f:
    train_data = json.load(f)

# 2 3 3 2 
# 2 3 5 5 
e, m, h, ex = 3, 4, 6, 6
total = 15
examples_num = {"easy": 0, "medium": 0, "hard": 0, "extra": 0}
examples = []

for idx, data in enumerate(train_data, 1):
    if len(examples) >= total:
        break
    
    question, query = data["question"], data["query"]
    hardness, counts = classify_hardness(query)
    
    if ((hardness == "easy" and examples_num[hardness] < e)
        or (hardness == "medium" and examples_num[hardness] < m)
        or (hardness == "hard" and examples_num[hardness] < h)
        or (hardness == "extra" and examples_num[hardness] < ex)):
        examples.append((question, query))
        examples_num[hardness] += 1

    

with open(example_path, "w") as f:
    for question, query in examples:
        f.write(f'Question: {question}\nSQL: {query}\n')