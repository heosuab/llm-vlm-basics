# RAG (Retrieval-Augmented Generation)

## RAG 기본 구조

```
사용자 질문
    ↓
Query Encoding → Vector DB 검색 (ANN)
    ↓
관련 문서 검색 (Top-K)
    ↓
Re-ranking (Cross-encoder)
    ↓
[System Prompt + Retrieved Docs + Question] → LLM
    ↓
답변 생성 (Grounded in Retrieved Docs)
```

**왜 RAG인가?**
- LLM 지식 컷오프 해결 (최신 정보)
- Hallucination 감소 (근거 문서 제공)
- 도메인 특화 지식 추가 (Fine-tuning 없이)
- 출처 추적 가능 (transparency)
- 비용 효율: Fine-tuning보다 저렴하게 도메인 특화

**RAG vs Long Context:**
```
RAG:    많은 문서 중 관련된 것만 선택 → 효율적
LC:     모든 문서를 컨텍스트에 넣기 → 비용, 속도, lost-in-middle

RAG 선택: 문서 수 많음, 검색 가능한 패턴
LC 선택: 문서 적음 (<200K), 전체 맥락 중요
```

---

## 1. 문서 처리 (Indexing)

### 청크 전략 (Chunking Strategies)

**Fixed-size Chunking:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # 토큰 수
    chunk_overlap=50,     # 연속성 유지
    separators=["\n\n", "\n", ".", " "]  # 우선순위 구분자
)
chunks = splitter.split_text(document)
```

**Semantic Chunking (의미 기반):**
```python
# 문장 임베딩 유사도 기반 경계 탐지
from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_chunk(text: str, threshold: float = 0.8):
    """유사도 급감 지점에서 청크 분리"""
    sentences = text.split(". ")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    embeddings = model.encode(sentences)

    # 인접 문장 간 코사인 유사도
    similarities = []
    for i in range(len(embeddings) - 1):
        cos_sim = np.dot(embeddings[i], embeddings[i+1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
        )
        similarities.append(cos_sim)

    # 유사도가 임계값 아래로 떨어지는 지점에서 분리
    boundaries = [i+1 for i, sim in enumerate(similarities) if sim < threshold]

    chunks = []
    prev = 0
    for boundary in boundaries:
        chunks.append(". ".join(sentences[prev:boundary]))
        prev = boundary
    chunks.append(". ".join(sentences[prev:]))
    return chunks
```

**Late Chunking (jina-embeddings-v3):**
```
기존: 청크 → 임베딩 (청크 내 컨텍스트만)
Late Chunking: 전체 문서 → 토큰 임베딩 → 청크별 평균 풀링
효과: 청크가 전체 문서 컨텍스트를 보존
```

**Parent-Child Chunking:**
```python
# 작은 청크로 검색 → 큰 청크로 답변 생성
# 정확한 검색 + 풍부한 컨텍스트

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

parent_docs = parent_splitter.split_text(doc)
child_docs = []
for i, parent in enumerate(parent_docs):
    children = child_splitter.split_text(parent)
    for child in children:
        child_docs.append({"text": child, "parent_id": i})

# 검색: child_docs (정확도)
# 반환: parent_docs[parent_id] (맥락)
```

---

## 2. 임베딩 모델

### 임베딩 아키텍처

**Bi-encoder (Dense Retrieval):**
```
Query → Encoder → q_vec ∈ R^d
Doc   → Encoder → d_vec ∈ R^d
Score = q_vec · d_vec (내적, 코사인 유사도)

장점: 빠름 (사전 계산 가능)
단점: 표현력 한계 (독립 인코딩)
```

**ColBERT (Late Interaction):**
```python
# MaxSim: 각 쿼리 토큰이 가장 유사한 문서 토큰과 매칭
# score(q, d) = Σᵢ max_j (qᵢ · dⱼ)

# 장점: Bi-encoder보다 정확, Cross-encoder보다 빠름
# 트레이드오프: 더 많은 스토리지 (토큰별 벡터 저장)

# RAGatouille 라이브러리
from ragatouille import RAGPretrainedModel
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
RAG.index(collection=docs, index_name="my_index")
results = RAG.search(query="...", k=5)
```

**임베딩 모델 비교:**

| 모델 | 차원 | 특징 | MTEB 순위 |
|------|------|------|---------|
| text-embedding-3-large | 3072 | OpenAI, 최강 영어 | ~SOTA |
| BAAI/bge-large-en-v1.5 | 1024 | 오픈소스 영어 최강 | Top-5 |
| BAAI/bge-m3 | 1024 | 다국어, Dense+Sparse+ColBERT | Top-10 |
| Qwen/Qwen2.5-Embeddings | 7B | 최신 LLM-based | SOTA |
| intfloat/multilingual-e5-large | 1024 | 다국어, E5 방법론 | Top-10 |
| jina-embeddings-v3 | 1024 | Late Chunking, Task-specific LoRA | Top-10 |
| nomic-embed-text-v1.5 | 768 | Apache 2.0, 긴 컨텍스트 | Top-15 |

**BGE-M3 (Multi-functionality):**
```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

# Dense 검색 (표준)
dense_scores = model.encode(queries)["dense_vecs"]

# Sparse 검색 (BM25-like, 토큰 가중치)
sparse_scores = model.encode(queries)["lexical_weights"]

# ColBERT (Multi-vector)
colbert_scores = model.encode(queries)["colbert_vecs"]

# Hybrid 점수 (DBFS)
final_score = (
    dense_weight * dense_score +
    sparse_weight * sparse_score +
    colbert_weight * colbert_score
)
```

---

## 3. Vector Database & 검색

### HNSW (Hierarchical Navigable Small World)

```
계층적 그래프 인덱스:
  상위 레이어: 드문 노드, 빠른 글로벌 탐색
  하위 레이어: 밀집 노드, 정밀한 로컬 탐색

검색 과정:
  1. 상위 레이어에서 목표에 가까운 진입점 찾기
  2. 아래 레이어로 이동하며 탐욕적 탐색
  3. 최하위 레이어에서 최근접 이웃 반환

파라미터:
  M: 각 노드의 최대 연결 수 (정확도↑ vs 메모리↑)
  ef_construction: 인덱스 빌드 시 탐색 범위
  ef_search: 쿼리 시 탐색 범위

근사 (ANN): 정확도 99%+ 달성 가능
```

**FAISS 사용법:**
```python
import faiss
import numpy as np

d = 1024  # 임베딩 차원
embeddings = np.random.randn(10000, d).astype("float32")

# HNSW 인덱스 (빠른 검색)
index = faiss.IndexHNSWFlat(d, 32)  # M=32
index.add(embeddings)

# 또는 IVF + PQ (대규모)
# IVF: Inverted File Index (클러스터링 기반)
# PQ: Product Quantization (압축)
nlist = 100  # 클러스터 수
m = 8        # PQ 서브벡터 수
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
index.train(embeddings[:5000])
index.add(embeddings)

# 검색
query = np.random.randn(1, d).astype("float32")
distances, indices = index.search(query, k=5)
```

**Vector DB 비교:**

| DB | 인덱스 | 특징 | 최적 사용 |
|----|--------|------|---------|
| Chroma | HNSW | 로컬, 개발용, Python 네이티브 | 프로토타입 |
| FAISS | HNSW, IVF | Meta, 빠른 검색, 인메모리 | 오프라인 처리 |
| Pinecone | 독자 | 클라우드, 관리형, 필터링 | 빠른 프로덕션 |
| Weaviate | HNSW | 하이브리드 검색, GraphQL | 하이브리드 |
| Qdrant | HNSW | 고성능, 페이로드 필터링 | 프로덕션 |
| Milvus | HNSW, IVF | 대규모, 클러스터, 분산 | 억 단위 |
| pgvector | HNSW, IVF | PostgreSQL 확장, ACID | 기존 PG |

---

## 4. 하이브리드 검색 (Hybrid Search)

```
Dense (시맨틱): 의미적 유사도, OOV 처리
Sparse (BM25): 정확한 키워드 매칭, 해석 가능

결합 방식: Reciprocal Rank Fusion (RRF)
  score(d) = Σₖ 1 / (k + rank_k(d))
  k=60 (상수), rank_k = 각 검색기에서의 순위

효과: 시맨틱 + 키워드 둘 다 커버
```

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, docs, embedding_model, alpha=0.5):
        """
        alpha: dense 가중치 (1-alpha: sparse 가중치)
        """
        self.docs = docs
        self.embedder = embedding_model
        self.alpha = alpha

        # BM25 인덱스
        tokenized = [d.split() for d in docs]
        self.bm25 = BM25Okapi(tokenized)

        # Dense 인덱스
        self.embeddings = embedding_model.encode(docs)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        # Sparse scores (BM25)
        sparse_scores = self.bm25.get_scores(query.split())

        # Dense scores (cosine)
        query_emb = self.embedder.encode([query])[0]
        dense_scores = self.embeddings @ query_emb / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )

        # RRF 결합
        sparse_ranks = np.argsort(-sparse_scores)
        dense_ranks = np.argsort(-dense_scores)

        rrf_scores = np.zeros(len(self.docs))
        for rank, doc_idx in enumerate(sparse_ranks):
            rrf_scores[doc_idx] += (1 - self.alpha) / (60 + rank + 1)
        for rank, doc_idx in enumerate(dense_ranks):
            rrf_scores[doc_idx] += self.alpha / (60 + rank + 1)

        top_k = np.argsort(-rrf_scores)[:k]
        return [self.docs[i] for i in top_k]
```

---

## 5. Re-ranking

```
Bi-encoder: 빠른 근사 검색 (Top-100)
Cross-encoder: 정확한 재순위 (Top-100 → Top-5)

Cross-encoder: 쿼리+문서를 함께 인코딩
  score = BERT([CLS] query [SEP] doc [SEP])
  정확하지만 느림 → Top-K 후처리로만 사용
```

```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3")

def rerank(query: str, candidates: list[str], top_k: int = 5) -> list[str]:
    """Cross-encoder로 재순위 매기기"""
    pairs = [[query, doc] for doc in candidates]
    scores = cross_encoder.predict(pairs)  # (N,) 배열

    # 점수 기준 정렬
    ranked = sorted(zip(scores, candidates), reverse=True)
    return [doc for _, doc in ranked[:top_k]]

# 파이프라인
dense_results = retriever.retrieve(query, k=100)  # Bi-encoder
final_results = rerank(query, dense_results, top_k=5)  # Cross-encoder
```

**LLM Re-ranking (Pointwise/Listwise):**
```python
# LLM으로 관련도 점수 매기기 (느리지만 최고 품질)
def llm_rerank(query: str, docs: list[str], llm) -> list[str]:
    scores = []
    for doc in docs:
        prompt = f"""Query: {query}
Document: {doc}

Rate the relevance of the document to the query from 0 to 10.
Output only the number."""
        score = float(llm.generate(prompt).strip())
        scores.append(score)

    ranked = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in ranked]
```

---

## 6. 고급 RAG 기법

### HyDE (Hypothetical Document Embeddings)

```
문제: 쿼리 임베딩 ≠ 답변 임베딩 분포
해결: 쿼리 → 가상 답변 생성 → 가상 답변으로 검색

Query: "Transformer attention 복잡도는?"
  ↓ LLM으로 가상 답변 생성
Hypo: "Transformer self-attention은 O(n²d) 복잡도를 가집니다.
       시퀀스 길이가 n이고 차원이 d일 때..."
  ↓ 가상 답변 임베딩으로 실제 문서 검색
Result: 관련 문서 검색 (임베딩 공간 격차 감소)

효과: 특히 팩추얼 질문에서 검색 품질 향상
```

```python
def hyde_retrieve(query: str, llm, retriever, k: int = 5) -> list[str]:
    # 가상 답변 생성
    hypo_prompt = f"Write a detailed answer to: {query}"
    hypothetical_doc = llm.generate(hypo_prompt)

    # 가상 답변으로 검색
    return retriever.retrieve(hypothetical_doc, k=k)
```

### Multi-Query Retrieval

```python
def multi_query_retrieve(query: str, llm, retriever, k: int = 5) -> list[str]:
    """여러 쿼리 변형으로 다각도 검색"""
    rewrite_prompt = f"""Generate 3 different versions of this query for retrieval:
Original: {query}
Output format: 1. ...\n2. ...\n3. ..."""

    variants_text = llm.generate(rewrite_prompt)
    variants = [query] + parse_variants(variants_text)  # 원본 + 3개 변형

    # 각 변형으로 검색, 중복 제거
    seen, all_results = set(), []
    for q in variants:
        for doc in retriever.retrieve(q, k=k):
            if doc not in seen:
                seen.add(doc)
                all_results.append(doc)

    return all_results[:k * 2]  # 통합 결과
```

### Contextual RAG (Anthropic, 2024)

```python
def add_context_to_chunk(document: str, chunk: str, llm) -> str:
    """각 청크에 전체 문서 맥락 추가"""
    prompt = f"""<document>
{document}
</document>

<chunk>
{chunk}
</chunk>

Provide a brief context (2-3 sentences) for this chunk within the full document.
Focus on what section this belongs to and what topic it covers."""

    context = llm.generate(prompt)
    return f"{context}\n\n{chunk}"

# 효과: 검색 성공률 49% 향상 (Anthropic 발표)
# 비용: 청크당 LLM 호출 → Prompt Caching으로 최적화
# document를 cache_control로 캐싱하면 비용 대폭 감소
```

### Self-RAG (선택적 검색)

```
기존 RAG: 모든 쿼리에 검색 → 불필요한 경우도 검색
Self-RAG: 검색 필요 여부 스스로 결정

특수 토큰:
  [Retrieve]: 검색 필요
  [NoRetrieve]: 검색 불필요
  [Relevant]: 검색 결과 관련 있음
  [Irrelevant]: 검색 결과 관련 없음
  [Supported]: 생성 내용이 문서에 지지됨
  [Contradicted]: 생성 내용이 문서와 모순

학습: Reflection Token을 예측하도록 SFT
```

### Corrective RAG (CRAG)

```
문제: 검색 결과 품질 낮을 수 있음
해결:
  1. 검색 결과 품질 평가 (평가 모델)
  2. 품질 낮으면 → Web Search로 보완
  3. 두 결과 통합 → 답변 생성

품질 평가:
  High: 직접 사용
  Low: 웹 검색으로 대체
  Medium: 필터링 후 웹 검색 보완
```

### RAPTOR (Recursive Abstractive Processing)

```python
# 계층적 문서 표현
def build_raptor_tree(chunks: list[str], llm, embedder, max_levels: int = 3):
    """
    bottom-up: 청크 → 클러스터 요약 → 더 높은 요약
    """
    current_level = chunks
    all_levels = [chunks]

    for level in range(max_levels):
        # GMM 클러스터링으로 관련 청크 그룹화
        embeddings = embedder.encode(current_level)
        clusters = gaussian_mixture_cluster(embeddings, n_components=max(1, len(current_level)//5))

        # 각 클러스터 요약
        summaries = []
        for cluster_docs in clusters:
            summary_prompt = f"Summarize:\n" + "\n".join(cluster_docs)
            summaries.append(llm.generate(summary_prompt))

        all_levels.append(summaries)
        current_level = summaries

        if len(current_level) <= 1:
            break

    return all_levels

# 검색: 모든 레벨 통합 검색, 또는 질문 복잡도에 따른 레벨 선택
```

---

## 7. GraphRAG (Microsoft, 2024)

```
단순 벡터 검색의 한계:
  - 여러 문서에 걸친 정보 통합 어려움
  - 전체적 요약 질문 처리 어려움
  - 엔티티 간 관계 파악 불가

GraphRAG 파이프라인:
  1. 문서 → LLM으로 엔티티/관계 추출
  2. 지식 그래프 구축 (노드: 엔티티, 엣지: 관계)
  3. Leiden 알고리즘으로 커뮤니티 탐지
  4. 커뮤니티별 계층적 요약 생성
  5. 요약의 임베딩 인덱싱

질의 종류:
  Local Search: 엔티티 중심 검색 (사실 질문)
    → 관련 엔티티 + 관계 + 청크 검색
  Global Search: 커뮤니티 요약 Map-Reduce
    → 모든 커뮤니티 요약에서 답변 수집 → 통합

적합한 태스크:
  - "이 데이터셋의 주요 테마는?" (전체 요약)
  - "A와 B의 관계는?" (관계 추론)
  - 일반 팩추얼 Q&A는 표준 RAG가 더 빠름
```

---

## 8. Agentic RAG

```python
import anthropic

class AgenticRAG:
    """검색을 도구로 사용하는 에이전트형 RAG"""

    def __init__(self, retriever, client: anthropic.Anthropic):
        self.retriever = retriever
        self.client = client

        self.tools = [
            {
                "name": "search_documents",
                "description": "Search the knowledge base for relevant information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "k": {"type": "integer", "description": "Number of results", "default": 5}
                    },
                    "required": ["query"]
                }
            }
        ]

    def run(self, user_query: str) -> str:
        messages = [{"role": "user", "content": user_query}]

        while True:
            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=2048,
                tools=self.tools,
                messages=messages
            )

            if response.stop_reason == "end_turn":
                return response.content[-1].text

            # 도구 호출 처리
            tool_results = []
            for content_block in response.content:
                if content_block.type == "tool_use":
                    if content_block.name == "search_documents":
                        docs = self.retriever.retrieve(
                            content_block.input["query"],
                            k=content_block.input.get("k", 5)
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": "\n\n".join(docs)
                        })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
```

---

## 9. RAG 평가 (RAGAS)

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # 생성 답변이 검색 문서에 근거하는가
    answer_relevancy,   # 답변이 질문과 관련있는가
    context_precision,  # 검색 문서 중 관련 있는 비율
    context_recall,     # 정답에 필요한 정보가 검색되었는가
)

# 평가 메트릭 설명:
# Faithfulness: 답변의 각 주장이 검색 컨텍스트로 지지되는가
#   = |지지되는 주장| / |전체 주장|

# Answer Relevancy: 역질문 생성으로 측정
#   LLM이 답변에서 여러 질문 생성 → 원본 질문과의 코사인 유사도

# Context Precision: 검색된 문서 중 실제로 유용한 문서 비율
#   = |관련 문서 순위 상위에 있는 비율|

# Context Recall: 정답에 필요한 문장이 검색 컨텍스트에 있는가
#   = |컨텍스트로 지지되는 정답 문장| / |정답 문장 수|

result = evaluate(
    dataset=eval_dataset,  # question, answer, contexts, ground_truth
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)  # DataFrame 출력
```

**간단한 RAG 평가:**
```python
def evaluate_rag(qa_pairs: list[dict], rag_pipeline) -> dict:
    """
    qa_pairs: [{"question": ..., "answer": ..., "contexts": [...]}]
    """
    metrics = {"faithfulness": [], "relevance": [], "retrieval_precision": []}

    for qa in qa_pairs:
        response, contexts = rag_pipeline.run(qa["question"])

        # Faithfulness: 생성 답변과 참조 답변 비교
        faithfulness_score = compute_faithfulness(response, contexts)
        metrics["faithfulness"].append(faithfulness_score)

        # Answer Relevance: 질문-답변 의미적 유사도
        relevance_score = compute_similarity(qa["question"], response)
        metrics["relevance"].append(relevance_score)

    return {k: sum(v)/len(v) for k, v in metrics.items()}
```

---

## 10. RAG 시스템 최적화

### Chunking 최적화

```
실험 권장 청크 크기:
  짧은 Q&A: 128-256 토큰
  일반 텍스트: 512 토큰
  기술 문서: 1024 토큰
  코드: 함수/클래스 단위

Overlap:
  일반: 10-20% (인접 컨텍스트 연속성)
  너무 크면: 중복 검색 결과 증가
```

### 메타데이터 필터링

```python
# 메타데이터로 검색 범위 좁히기
results = collection.query(
    query_embeddings=query_emb,
    n_results=5,
    where={
        "category": "technical",
        "date": {"$gte": "2024-01-01"},  # 최신 문서만
        "language": "ko"
    }
)
```

### 쿼리 전처리

```python
def preprocess_query(query: str, llm) -> str:
    """쿼리 품질 개선"""
    # 1. 오타 교정
    # 2. 쿼리 확장 (동의어)
    # 3. 검색 의도 명확화

    prompt = f"""Rewrite this search query to be more specific and effective for retrieval:
Query: {query}
Rewritten:"""
    return llm.generate(prompt)
```

### 캐싱 전략

```python
import hashlib
from functools import lru_cache

class CachedRAG:
    def __init__(self, rag_pipeline, cache_size=1000):
        self.rag = rag_pipeline
        self.cache = {}  # query_hash → (results, timestamp)

    def retrieve(self, query: str, ttl: int = 3600) -> list[str]:
        # 쿼리 해시로 캐시 키 생성
        query_hash = hashlib.md5(query.encode()).hexdigest()

        if query_hash in self.cache:
            results, timestamp = self.cache[query_hash]
            if time.time() - timestamp < ttl:
                return results  # 캐시 히트

        results = self.rag.retrieve(query)
        self.cache[query_hash] = (results, time.time())
        return results
```

---

## 11. Production RAG 시스템

```
아키텍처:
  ┌─────────────────────────────────────┐
  │  API Gateway / Rate Limiting        │
  └──────────────┬──────────────────────┘
                 │
  ┌──────────────▼──────────────────────┐
  │  Query Service                       │
  │  - 쿼리 전처리 (오타, 확장)           │
  │  - 쿼리 라우팅 (RAG vs direct)       │
  │  - 결과 캐싱                         │
  └──────────────┬──────────────────────┘
                 │
  ┌──────────────▼──────────────────────┐
  │  Retrieval Service                   │
  │  - Dense Retrieval (Vector DB)       │
  │  - Sparse Retrieval (Elasticsearch) │
  │  - Hybrid (RRF)                      │
  └──────────────┬──────────────────────┘
                 │
  ┌──────────────▼──────────────────────┐
  │  Re-ranking Service                  │
  │  - Cross-encoder 재순위              │
  └──────────────┬──────────────────────┘
                 │
  ┌──────────────▼──────────────────────┐
  │  Generation Service (LLM)            │
  │  - Prompt 구성                       │
  │  - Streaming 응답                    │
  │  - Citation 삽입                     │
  └─────────────────────────────────────┘

인덱싱 파이프라인 (별도):
  원본 문서 → OCR/파싱 → 청크 → 임베딩 → Vector DB
  새 문서 → 점진적 업데이트 (Online Indexing)
```

**Citation 추가:**
```python
def generate_with_citations(query: str, docs: list[dict], llm) -> str:
    """검색 문서에 출처 번호 부여, 인용 포함 답변 생성"""
    context = ""
    for i, doc in enumerate(docs):
        context += f"[{i+1}] {doc['title']}: {doc['text']}\n\n"

    prompt = f"""Answer the question using the provided sources.
Include citation numbers [1], [2] etc. when referencing sources.

Sources:
{context}

Question: {query}
Answer:"""

    return llm.generate(prompt)
```

---

## RAG vs Fine-tuning 결정 기준

```
RAG 선택:
  ✓ 자주 업데이트되는 정보
  ✓ 출처 추적 필요
  ✓ 도메인 특화 사실 지식
  ✓ 빠른 배포
  ✓ 개인화 (사용자별 문서)
  ✓ 규정 준수 (문서 기반 답변 보장)

Fine-tuning 선택:
  ✓ 특정 형식/스타일 학습
  ✓ 도메인 언어 패턴 학습
  ✓ 추론 능력 강화
  ✓ 레이턴시 중요 (검색 단계 없음)
  ✓ 정적 지식 (업데이트 드뭄)

RAG + Fine-tuning 조합:
  Fine-tuning: 스타일, 언어 패턴, 추론 방식
  RAG: 최신 사실 정보, 기업 내부 문서
```

---

## Further Questions

**Q. RAG에서 청크 크기 결정 방법은?**
> 작은 청크(128-256): 정확한 검색, 맥락 부족. 큰 청크(1024+): 맥락 풍부, 노이즈 포함. 일반적으로 512 토큰, overlap 10-20%. Parent-Child 구조로 작은 청크 검색 + 큰 청크 반환이 최적. 태스크마다 실험 필요. RAGAS로 자동 평가.

**Q. Vector DB 선택 기준은?**
> 규모(수백만 이하: Qdrant/Chroma, 수억: Milvus), 성능 요구(속도/정확도 트레이드오프), 하이브리드 검색 필요(Weaviate, Elasticsearch+pgvector), 필터링 요구(Qdrant 탁월), 관리 편의성. 프로토타입: Chroma. 프로덕션 소규모: Qdrant. 대규모 분산: Milvus.

**Q. RAG 환각을 줄이는 방법은?**
> 1) Faithfulness 평가 후 임계값 이하면 재검색. 2) Cross-encoder로 관련 없는 문서 필터링. 3) 프롬프트에 "문서에 없으면 모른다고 답해"지시. 4) Self-RAG처럼 [Supported]/[Contradicted] 체크. 5) 출처 인용 강제. 6) NLI 모델로 답변-문서 일관성 검사.

**Q. Dense vs Sparse 검색, 언제 어느 것을?**
> Dense: 의미적 유사도, 패러프레이즈, 도메인 이해 필요. Sparse(BM25): 정확한 키워드 매칭, 고유명사, 제품 코드. 일반적으로 하이브리드(RRF)가 최선. 짧은 키워드 쿼리: Sparse 가중치↑. 자연어 긴 쿼리: Dense 가중치↑.

**Q. 수백만 개 문서를 RAG에 사용할 때 고려사항은?**
> 1) Sharding: 의미적/메타데이터 기반 파티셔닝. 2) Pre-filtering: 메타데이터로 검색 범위 축소. 3) Two-stage: Coarse→Fine 검색. 4) Caching: 인기 쿼리 캐싱. 5) 점진적 인덱싱: 새 문서 스트리밍 추가. 6) 모니터링: 검색 품질 지속 추적 (RAGAS).
