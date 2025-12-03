"""
Intent-based clustering for NL2SQL Few-Shot Learning

1. 질문의 의도(intent)만으로 clustering (스키마 독립적)
2. SQL 패턴 유사도를 기반으로 cluster 형성
3. 새 질문 → 같은 intent cluster의 예제 선택
"""

import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.cluster import KMeans
from paths import INDEX_DIR

# 기존 파일들
questions_file = INDEX_DIR / "questions.pkl"
sqls_file = INDEX_DIR / "sqls.pkl"
embeddings_file = INDEX_DIR / "embeddings.npy"

# 새로운 clustering 관련 파일들
clusters_file = INDEX_DIR / "clusters.pkl"
cluster_centers_file = INDEX_DIR / "cluster_centers.npy"

embeddings, embedder, cluster_centers, cluster_labels = None, None, None, None
questions, sqls = [], []

# SQL 패턴 추출 함수
def extract_sql_pattern(sql: str) -> str:
    """
    SQL에서 구조적 패턴만 추출 (테이블/컬럼명 제거)
    
    예: "SELECT name FROM users WHERE age > 30"
    → "SELECT * FROM * WHERE * > *"
    """
    import re
    
    # 기본 정규화
    sql = sql.upper().strip()
    
    # 문자열/숫자 리터럴 제거
    sql = re.sub(r"'[^']*'", "*", sql)
    sql = re.sub(r"\d+", "*", sql)
    
    # 테이블명/컬럼명을 *로 대체 (키워드는 유지)
    keywords = {
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
        'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'DISTINCT',
        'COUNT', 'SUM', 'AVG', 'MIN', 'MAX',
        'AND', 'OR', 'NOT', 'IN', 'BETWEEN', 'LIKE',
        'UNION', 'INTERSECT', 'EXCEPT', 'AS'
    }
    
    # 간단한 토큰화 후 키워드가 아닌 것은 *로
    tokens = re.findall(r'\w+|[^\w\s]', sql)
    pattern_tokens = []
    for token in tokens:
        if token.upper() in keywords or token == '*':
            pattern_tokens.append(token)
        elif token in ['(', ')', ',', '=', '>', '<', '>=', '<=', '!=']:
            pattern_tokens.append(token)
        else:
            pattern_tokens.append('*')
    
    return ' '.join(pattern_tokens)


def build_intent_clusters(n_clusters: int = 50):
    """
    질문을 intent 기반으로 clustering
    
    Args:
        n_clusters: 클러스터 개수 (기본 50)
    """
    
    print("*** Loading embeddings and questions...")
    embeddings = np.load(embeddings_file)
    
    with open(questions_file, 'rb') as f:
        questions = pickle.load(f)
    
    with open(sqls_file, 'rb') as f:
        sqls = pickle.load(f)
    
    print(f"*** Clustering {len(questions)} questions into {n_clusters} intent clusters...")
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # 각 클러스터별 예제 분석
    cluster_info = {}
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        cluster_sqls = [sqls[j] for j in range(len(sqls)) if cluster_mask[j]]
        
        # 이 클러스터의 대표 SQL 패턴들
        patterns = [extract_sql_pattern(sql) for sql in cluster_sqls[:5]]
        
        cluster_info[i] = {
            'size': int(np.sum(cluster_mask)),
            'sample_patterns': patterns[:3]
        }
    
    # 저장
    with open(clusters_file, 'wb') as f:
        pickle.dump({
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'info': cluster_info
        }, f)
    
    np.save(cluster_centers_file, kmeans.cluster_centers_)
    
    print(f"*** Clustering complete!")
    print(f"*** Cluster info saved to {clusters_file}")
    
    # 클러스터 통계 출력
    print("\n=== Cluster Statistics ===")
    for i in range(min(5, n_clusters)):
        info = cluster_info[i]
        print(f"\nCluster {i}: {info['size']} examples")
        print(f"Sample patterns: {info['sample_patterns'][0]}")
    
    return cluster_labels, kmeans.cluster_centers_


def load_clusters():
    global embeddings, embedder, embeddings, cluster_centers, cluster_labels, questions, sqls

    # Load resources
    embedder = SentenceTransformer('BAAI/bge-base-en-v1.5')

    embeddings = np.load(embeddings_file)
    cluster_centers = np.load(cluster_centers_file)
    
    with open(questions_file, 'rb') as f:
        questions = pickle.load(f)
    
    with open(sqls_file, 'rb') as f:
        sqls = pickle.load(f)
    
    with open(clusters_file, 'rb') as f:
        cluster_data = pickle.load(f)
        cluster_labels = cluster_data['labels']


def retrieve_intent_based_examples(question: str, k: int = 5, k_clusters: int = 3) -> list:
    """
    Intent clustering 기반 예제 검색
    
    Args:
        question: 입력 질문
        k: 반환할 예제 개수
        k_clusters: 검색할 클러스터 개수 (다양성 확보)
    
    Returns:
        list of examples
    """    

    if embedder is None:
        load_clusters()
    
    # 1. 질문 embedding
    question_embedding = embedder.encode([question], convert_to_numpy=True)
    question_embedding = question_embedding.astype('float32')
    faiss.normalize_L2(question_embedding)
    
    # 2. 가장 가까운 k_clusters개의 클러스터 찾기
    centers_normalized = cluster_centers.astype('float32')
    faiss.normalize_L2(centers_normalized)
    
    # FAISS로 nearest clusters 검색
    index = faiss.IndexFlatIP(centers_normalized.shape[1])  # Inner Product (cosine similarity)
    index.add(centers_normalized)
    
    _, nearest_clusters = index.search(question_embedding, k_clusters)
    nearest_clusters = nearest_clusters[0]
    
    # 3. 각 클러스터에서 가장 유사한 예제들 수집
    candidate_indices = []
    for cluster_id in nearest_clusters:
        # 이 클러스터에 속한 예제들의 인덱스
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        # 클러스터 내에서 질문과 가장 유사한 예제들 찾기
        cluster_embeddings = embeddings[cluster_indices]
        
        # Cosine similarity 계산
        similarities = np.dot(cluster_embeddings, question_embedding.T).flatten()
        
        # 상위 예제들 선택 (클러스터당 k//k_clusters개)
        n_per_cluster = max(1, k // k_clusters)
        top_indices_in_cluster = np.argsort(similarities)[-n_per_cluster:][::-1]
        
        candidate_indices.extend(cluster_indices[top_indices_in_cluster])
    
    # 4. 최종 k개 선택 (중복 제거 후)
    candidate_indices = list(set(candidate_indices))[:k]
    
    examples = []
    for idx in candidate_indices:
        examples.append({
            "input": questions[idx],
            "query": sqls[idx]
        })
    
    return examples

if __name__ == "__main__":
    build_intent_clusters()
    # # 테스트
    # examples = retrieve_intent_based_examples(
    #     "What is the average salary?", 
    #     k=5
    # )

    # for ex in examples:
    #     print(ex['input'])
    #     print(ex['query'])
    #     print("---")