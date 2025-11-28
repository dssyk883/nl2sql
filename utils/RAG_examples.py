import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

PRJ_ROOT = Path(__file__).parent.parent
DATA_DIR = PRJ_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"

embeddings_file = INDEX_DIR / "embeddings.npy"
questions_file = INDEX_DIR / "questions.pkl"
sqls_file = INDEX_DIR / "sqls.pkl"
faiss_index_file = INDEX_DIR / "faiss.index"

embedder = None
train_questions = []
train_sqls = []
faiss_index = None

def load_index():
    global embedder, train_questions, train_sqls, faiss_index

    print("*** Loading embedder...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    print("*** Loading questions...")
    with open(questions_file, 'rb') as f:
        train_questions = pickle.load(f)

    print("*** Loading sqls...")
    with open(sqls_file, 'rb') as f:
        train_sqls = pickle.load(f)

    print("*** Loading FAISS index...")
    faiss_index = faiss.read_index(str(faiss_index_file))

    print(f"*** Load {len(train_questions)} vectors on CPU")



def retrieve_RAG_examples(question: str, k: int = 3) -> list:
    """
    RAG retrieve
    질문과 유사한 예제 k 개의 검색

    Args:
        question: 사용자 질문
        k: 반환할 예제 개수
    """

    if faiss_index is None:
        load_index()
    
    query_embedding = embedder.encode([question], convert_to_numpy=True)
    # SentenceTransformer - float64 사용, 
    # faiss - float32 사용
    query_embedding = query_embedding.astype('float32')
    faiss.normalize_L2(query_embedding)

    distances, indices = faiss_index.search(query_embedding, k)

    examples = []
    for i, idx in enumerate(indices[0]):
        # print(f"{i+1}. 거리: {distances[0][i]:.3f}")
        # print(f"   Q: {train_questions[idx]}")
        # print(f"   SQL: {train_sqls[idx][:80]}...\n")
        examples.append({
            "input": train_questions[idx],
            "query": train_sqls[idx]
        })

    return examples