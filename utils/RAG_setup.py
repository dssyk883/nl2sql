import json
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

PRJ_ROOT = Path(__file__).parent.parent
DATA_DIR = PRJ_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"

train_path = DATA_DIR / "train_spider.json"

embedding_file = INDEX_DIR / "embeddings.npy"
questions_file = INDEX_DIR / "questions.pkl"
sqls_file = INDEX_DIR / "sqls.pkl"
faiss_index_file = INDEX_DIR / "faiss.index"

def build_save_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with open(train_path, "r") as f:
        train_data = json.load(f)

    questions = [item['question'] for item in train_data]
    sqls = [item['query'] for item in train_data]

    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(
        questions,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)

    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)

    np.save(embedding_file, embeddings)
    with open(questions_file, "wb") as f:
        pickle.dump(questions, f)
    with open(sqls_file, "wb") as f:
        pickle.dump(sqls, f)
    faiss.write_index(index, str(faiss_index_file))

build_save_index()