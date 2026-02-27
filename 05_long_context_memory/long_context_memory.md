# Section 5: Long Context & Memory

> **학습 목표**: LLM이 긴 문서를 어떻게 처리하는지, Retrieval-Augmented Generation(RAG)의 전체 파이프라인, 그리고 모델의 "기억"을 확장하는 다양한 기법들.

---

## 5.1 Context Window Scaling

### Context Window란?

Transformer 기반 언어 모델은 **한 번에 처리할 수 있는 토큰 수에 제한**이 있음. 이 최대 토큰 수를 **context window** (또는 context length)라고 부름. Context window 안에 있는 토큰들만이 attention 계산에 참여할 수 있으며, 그 밖의 정보는 모델이 "볼 수 없음".

```
[system prompt | user input | retrieved documents | model output]
 ←————————————— context window (N tokens) ——————————————→
```

### 역사적 진화: 2K → 1M+

| 모델 | 출시 연도 | Context Length |
|---|---|---|
| GPT-2 | 2019 | 1,024 tokens |
| GPT-3 | 2020 | 2,048 tokens |
| LLaMA-1 | 2023.02 | 2,048 tokens |
| GPT-3.5-Turbo | 2023.03 | 4,096 → 16,384 tokens |
| LLaMA-2 | 2023.07 | 4,096 tokens |
| Claude 2 | 2023.07 | 100,000 tokens |
| GPT-4-Turbo | 2023.11 | 128,000 tokens |
| Gemini 1.5 Pro | 2024.02 | 1,000,000 tokens |
| LLaMA-3.1 | 2024.07 | 128,000 tokens |

### 왜 Context Window를 늘리기 어려운가?

#### 1. Attention Memory: O(N²) 문제

Transformer의 self-attention은 모든 토큰 쌍 사이의 attention score를 계산함:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

- `Q, K, V` ∈ ℝ^(N × d_k) 행렬
- `QK^T` 계산 결과는 ℝ^(N × N) 행렬

따라서 메모리 사용량은 **O(N²)** 로 증가함:

| Context Length | Attention Matrix 크기 | 메모리 (fp16, 1 head) |
|---|---|---|
| 2K | 2K × 2K = 4M | ~8 MB |
| 32K | 32K × 32K = 1B | ~2 GB |
| 128K | 128K × 128K = 16B | ~32 GB |
| 1M | 1M × 1M = 1T | ~2 TB (!!) |

Gemini 1.5가 1M context를 달성할 수 있었던 핵심 기술 중 하나는 **Flash Attention** (memory-efficient attention 알고리즘, Section 4에서 다룸)과 함께 **Ring Attention** (여러 디바이스에 걸쳐 attention을 분산)을 사용했기 때문.

#### 2. Positional Encoding Extrapolation 문제

원래 Transformer는 **sinusoidal positional encoding** 또는 **learned positional embedding**을 사용했음. 이 방식들은 학습 때 본 길이 이상으로는 **일반화(extrapolation)가 매우 나쁨**.

예를 들어 최대 2K 토큰으로 학습한 모델에게 4K 토큰의 텍스트를 주면, 2K 이상의 position에 대한 embedding 값을 모델이 학습한 적이 없으므로 성능이 급격히 떨어짐.

**RoPE (Rotary Position Embedding)**는 이 문제를 부분적으로 해결함. LLaMA, Mistral, Gemma 등 현대 모델 대부분이 RoPE를 사용함.

RoPE는 position `m`에서의 벡터를 다음과 같이 회전 행렬로 표현함:

```
f(x, m) = x · e^(i·m·θ)

θ_j = 1 / (10000^(2j/d))   (j = 0, 1, ..., d/2 - 1)
```

Q와 K의 내적이 relative position `m - n`에만 의존하게 되어, 학습 때 보지 못한 긴 sequence로 어느 정도 외삽(extrapolation)할 수 있음.

**RoPE Scaling 기법들:**

| 방법 | 핵심 아이디어 |
|---|---|
| **Linear Scaling** | θ를 `s`배 줄여서 모든 position을 `1/s`로 압축 |
| **NTK-aware Scaling** | 고주파/저주파 성분을 다르게 scaling (YaRN의 기반) |
| **YaRN** | 고주파 성분은 interpolation 없이, 저주파 성분만 scaling |
| **LongRoPE** | position을 더 촘촘하게 rescaling, fine-tuning 없이 context 확장 |

#### 3. Lost-in-the-Middle Problem

Context window가 길어지면 모델이 **긴 context 중간에 위치한 정보를 잘 활용하지 못하는** 현상이 관찰됨.

Stanford 연구팀의 2023년 논문 "Lost in the Middle"에 따르면:

```
정답 위치별 모델 성능 (N개의 문서 중 정답 문서의 위치):

성능
 ↑
 |■■■■                                   ■■■■
 |    ■■■                           ■■■
 |       ■■                       ■■
 |         ■■■                 ■■■
 |            ■■■■■■■■■■■■■■■■■
 |
 +——————————————————————————————————→ 위치
  첫 번째              중간          마지막
```

모델은 context의 **처음**과 **끝** 부분에 있는 정보를 훨씬 잘 활용함. 이는 RAG 시스템 설계에서 중요한 시사점을 가짐: **가장 관련성 높은 청크를 context의 앞부분이나 뒷부분에 배치**해야 함.

이 현상의 원인으로는:
- Attention score의 **recency bias** (최근 토큰에 더 집중)
- **Primacy effect** (처음 나온 정보에 대한 편향)
- 긴 sequence에서 gradient가 중간 위치에 덜 전달되는 학습상의 문제

---

## 5.2 Chunking Strategies

RAG 시스템의 첫 번째 단계는 긴 문서를 검색 가능한 단위인 **chunk**로 나눔. Chunking 전략은 retrieval 품질에 직접적인 영향을 미침.

### 5.2.1 Fixed-Size Chunking

가장 단순한 방법: **N개의 토큰(또는 문자)마다 고정적으로 분할**함.

```python
def fixed_size_chunking(text, chunk_size=512, overlap=50):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        chunks.append(tokenizer.decode(chunk))
        start += (chunk_size - overlap)  # overlap만큼 앞으로 돌아감
    return chunks
```

**장점**: 구현 간단, 일정한 chunk 크기 보장
**단점**: 문장/단락 중간에서 잘릴 수 있음 → 의미 파괴

### 5.2.2 Sentence/Paragraph-Aware Chunking

텍스트의 **자연스러운 경계** (문장 끝, 단락 끝)를 존중하면서 분할함.

```python
import re

def sentence_aware_chunking(text, max_tokens=512):
    # 문장 단위로 분리
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence))

        if current_size + sentence_tokens > max_tokens and current_chunk:
            # 현재 청크를 저장하고 새로 시작
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_size += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

**단락(paragraph) 기반 분할**은 더욱 의미 단위를 잘 보존함. HTML 문서라면 `<p>` 태그, Markdown이라면 `\n\n` 기준으로 나눌 수 있음.

### 5.2.3 Semantic Chunking

**의미적으로 유사한 문장들을 하나의 chunk로 묶는** 방법. 문장들을 embedding하고, 인접 문장 간의 cosine similarity가 급격히 떨어지는 지점을 경계로 삼음.

```
문장 1 ——similarity=0.92—— 문장 2 ——similarity=0.88—— 문장 3
                                                        |
                                               similarity=0.31 (급격히 낮아짐)
                                                        |
문장 4 ——similarity=0.85—— 문장 5 ——similarity=0.90—— 문장 6
```

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def semantic_chunking(sentences, embeddings, threshold=0.5):
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        # 연속된 문장 사이의 cosine similarity 계산
        sim = cosine_similarity(
            embeddings[i-1].reshape(1, -1),
            embeddings[i].reshape(1, -1)
        )[0][0]

        if sim < threshold:
            # 의미적 경계 → 새 청크 시작
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    chunks.append(" ".join(current_chunk))
    return chunks
```

**장점**: 의미 단위가 잘 보존됨
**단점**: Embedding 비용이 추가로 필요, threshold 설정이 까다로움

### 5.2.4 Chunk Overlap

연속된 chunk 간에 **겹치는 부분(overlap)**을 두면 chunk 경계에서 잘리는 정보를 보완할 수 있음.

```
Chunk 1: [토큰 1 ~ 512]
Chunk 2: [토큰 462 ~ 973]   ← 50 토큰 overlap
Chunk 3: [토큰 923 ~ 1434]  ← 50 토큰 overlap
```

Overlap이 너무 크면:
- 저장 공간 낭비
- 중복 정보가 retrieval 결과에 반복 등장

Overlap이 너무 작으면:
- 중요한 정보가 chunk 경계에서 소실

일반적으로 **chunk size의 10~20%**가 적절한 overlap 크기로 권장됨.

### 5.2.5 Chunk Size가 Retrieval Quality에 미치는 영향

| Chunk Size | 장점 | 단점 |
|---|---|---|
| **너무 작음** (64~128 tokens) | 정밀한 retrieval 가능, 노이즈 적음 | 맥락 부족, 완전한 답변을 담기 어려움 |
| **중간** (256~512 tokens) | 균형 잡힌 성능 | 대부분의 경우 권장 |
| **너무 큼** (1K~2K tokens) | 풍부한 맥락 제공 | 관련 없는 내용 포함, embedding이 희석됨 |

**Parent-Child Chunking** 전략: 큰 청크(parent)를 검색 인덱스에 저장하되, 실제 retrieval 시에는 작은 청크(child)로 검색하고, 생성 시에는 parent 청크의 내용을 제공함.

---

## 5.3 Retrieval-Augmented Generation (RAG)

### RAG의 동기

LLM은 학습 데이터의 **knowledge cutoff** 이후 정보를 모름. 또한 model weights에 사실 정보를 "압축"하는 과정에서 **hallucination(환각)**이 발생할 수 있음.

RAG는 이를 해결하기 위해 **관련 문서를 동적으로 검색하여 LLM의 context에 제공**함:

```
[Query] → [Retriever] → [Top-K Chunks] → [LLM] → [Answer]
                ↑
        [Vector Database]
              ↑
        [Document Corpus]
```

### 5.3.1 Indexing Phase (오프라인 단계)

문서들을 미리 처리하여 검색 가능한 형태로 저장함.

```
원본 문서들
    ↓  [1. 문서 로드 & 전처리]
청크 분할 (Section 5.2)
    ↓  [2. Embedding]
각 청크의 dense vector (e.g., 768차원)
    ↓  [3. 인덱싱]
Vector DB (FAISS, Pinecone, etc.)
```

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 문서 로드
documents = load_documents("./docs/")

# 2. 청크 분할
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# 3. Embedding & 인덱싱
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("./index/")
```

### 5.3.2 Retrieval Phase (실시간 단계)

사용자 query가 들어오면 관련 chunk를 검색함.

```
Query: "트랜스포머의 attention은 어떻게 작동하나요?"
    ↓  [Query Embedding]
query_vector = embed("트랜스포머의 attention은 어떻게 작동하나요?")
    ↓  [ANN Search]
Top-K 청크 = vectorstore.similarity_search(query_vector, k=5)
```

### 5.3.3 Generation Phase

검색된 chunk들을 LLM의 context에 삽입하여 답변을 생성함.

```python
def rag_generate(query, vectorstore, llm, k=5):
    # 관련 청크 검색
    relevant_chunks = vectorstore.similarity_search(query, k=k)

    # Context 구성
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    # Prompt 구성
    prompt = f"""다음 문서를 참고하여 질문에 답하세요.

문서:
{context}

질문: {query}

답변:"""

    return llm.generate(prompt)
```

### RAG가 Factuality와 Freshness를 개선하는 이유

1. **Factuality**: 모델이 "기억"에 의존하는 대신 **실제 문서를 보고** 답하므로 hallucination 감소
2. **Freshness**: Vector DB를 업데이트하면 **모델 재학습 없이** 새로운 정보 반영 가능
3. **Attribution**: 출처 문서를 함께 제공할 수 있어 **검증 가능성** 증가
4. **Domain Adaptation**: 특정 도메인 문서만 인덱싱하여 **특화된 지식** 제공

### RAG의 한계

- **Retrieval 품질**에 전적으로 의존 (관련 chunk를 못 찾으면 답변 품질 저하)
- **Multi-hop reasoning** 어려움 (여러 문서에 걸쳐 추론해야 하는 경우)
- **Context window 제한** (너무 많은 chunk를 넣으면 lost-in-the-middle 문제)
- **Latency 증가** (embedding + search 단계가 추가됨)

---

## 5.4 Embeddings

### Dense Vector Representation이란?

**Embedding**은 텍스트(단어, 문장, 문단)를 고차원의 연속적인 벡터 공간에 매핑함.

```
"고양이" → [0.23, -0.45, 0.71, 0.12, ..., -0.33]  ← 768차원 벡터
"cat"    → [0.21, -0.43, 0.69, 0.15, ..., -0.31]  ← 의미적으로 유사 → 근접
"개"     → [0.18, -0.41, 0.65, 0.19, ..., -0.28]  ← 동물이므로 어느정도 근접
"수학"   → [-0.51, 0.33, -0.12, 0.88, ..., 0.44]  ← 의미적으로 멀리 위치
```

핵심 속성: **의미적으로 유사한 텍스트는 벡터 공간에서 가까이 위치**함.

### 대표적인 Embedding 모델들

#### text-embedding-ada-002 (OpenAI)
- 차원: 1,536
- 강력한 성능, API 형태로 제공
- 2023년까지 MTEB(Massive Text Embedding Benchmark) 상위권

#### E5 시리즈 (Microsoft)
```
intfloat/e5-small     (33M params,  384차원)
intfloat/e5-base      (110M params, 768차원)
intfloat/e5-large     (335M params, 1024차원)
intfloat/e5-mistral-7b (7B params,  4096차원)
```
E5의 특징: 학습 시 쿼리에 `"query: "`, 문서에 `"passage: "` 접두어를 붙여 검색 용도에 최적화.

#### BGE 시리즈 (BAAI)
```
BAAI/bge-small-en-v1.5
BAAI/bge-base-en-v1.5
BAAI/bge-large-en-v1.5
BAAI/bge-m3  ← 다국어, 긴 문서 지원 (8K tokens)
```

#### GTE, Nomic-Embed, Voyage AI 등도 경쟁력 있는 선택지.

### Cosine Similarity vs Dot Product

두 벡터 `a`와 `b`의 유사도를 측정하는 두 가지 주요 방법:

**Cosine Similarity:**
```
cosine_sim(a, b) = (a · b) / (||a|| · ||b||)

값의 범위: [-1, +1]
```

벡터의 **방향**만 비교함 (크기 정규화). 문서 길이에 관계없이 의미적 유사성을 잘 포착함.

**Dot Product:**
```
dot_product(a, b) = Σ aᵢ · bᵢ

값의 범위: (-∞, +∞)
```

벡터의 **방향과 크기** 모두 고려함.

| | Cosine Similarity | Dot Product |
|---|---|---|
| 정규화 벡터 (||v||=1) | 동일한 결과 | 동일한 결과 |
| 비정규화 벡터 | 크기 무시, 방향만 비교 | 크기도 반영 |
| 연산 속도 | 정규화 비용 추가 | 더 빠름 |
| 권장 사용 | 벡터 크기가 의미 없을 때 | 크기도 중요할 때 |

> 실전 팁: 대부분의 최신 embedding 모델은 **L2 정규화된 벡터**를 출력하므로, cosine similarity와 dot product가 동일함. 이 경우 dot product가 더 빠르기 때문에 ANN 라이브러리에서 dot product를 사용하는 것이 일반적.

### Embedding Space의 흥미로운 속성

**Semantic Arithmetic:**
```
embed("왕") - embed("남자") + embed("여자") ≈ embed("여왕")
```

**Multilingual Alignment:**
잘 학습된 다국어 embedding 모델에서는 서로 다른 언어의 같은 의미 표현이 가까운 벡터를 가짐:
```
embed("고양이", ko) ≈ embed("cat", en) ≈ embed("猫", zh)
```

**Matryoshka Representation Learning (MRL):**
최신 embedding 모델들은 **차원을 줄여도 성능이 크게 떨어지지 않도록** 학습됨. 예를 들어 1536차원 벡터의 앞 256차원만 사용해도 좋은 성능을 냄 → 저장 공간과 연산을 절약 가능.

---

## 5.5 Vector Database

### ANN (Approximate Nearest Neighbor) Search

정확한 최근접 이웃 탐색(Exact Nearest Neighbor Search)은 O(N·d) 복잡도 (N: 벡터 수, d: 차원). 수백만 개의 벡터에서 실시간으로 탐색하려면 **근사 탐색(ANN)**이 필수적.

ANN은 **약간의 정확도를 희생하고 대신 속도를 극적으로 향상**시킴.

주요 ANN 알고리즘:

### 5.5.1 HNSW (Hierarchical Navigable Small World)

현재 가장 널리 사용되는 ANN 알고리즘.

**아이디어**: 여러 계층의 그래프를 구성함. 상위 계층은 sparse한 long-range 연결, 하위 계층은 dense한 short-range 연결.

```
Layer 2 (Sparse):    *————*————*————*
                              |
Layer 1 (Medium):    *—*—*—*—*—*—*—*
                          |   |
Layer 0 (Dense):  *—*—*—*—*—*—*—*—*—*—*—*
                              ↑
                         Query 진입점
```

검색 시: 최상위 layer에서 시작해 query와 가까운 노드를 따라가면서 계층을 내려감 → 최하위 layer에서 실제 nearest neighbor 탐색.

| 특성 | 값 |
|---|---|
| 탐색 복잡도 | O(log N) |
| 인덱스 구축 복잡도 | O(N log N) |
| 메모리 사용량 | 상대적으로 높음 |
| 정확도 (recall@10) | 95~99% 달성 가능 |
| 동적 삽입 | 지원 |

핵심 하이퍼파라미터:
- `M`: 각 노드의 최대 연결 수 (크면 높은 recall, 높은 메모리)
- `ef_construction`: 인덱스 구축 시 탐색 폭 (크면 높은 품질, 느린 구축)
- `ef_search`: 검색 시 탐색 폭 (크면 높은 recall, 느린 검색)

### 5.5.2 IVF (Inverted File Index)

**아이디어**: 벡터 공간을 K개의 클러스터로 나누고(K-means), 검색 시 query와 가까운 몇 개의 클러스터만 탐색.

```
학습 단계: K-means로 벡터 공간을 분할
         [Centroid 1] [Centroid 2] ... [Centroid K]
              |             |                |
         [벡터들...] [벡터들...]      [벡터들...]

검색 단계:
1. Query에 가장 가까운 nprobe개의 centroid 찾기
2. 해당 클러스터 내의 벡터들만 탐색
```

| 특성 | 값 |
|---|---|
| 탐색 복잡도 | O(nprobe · (N/K)) |
| 메모리 효율 | PQ 압축과 결합 시 매우 효율적 |
| 동적 삽입 | 어려움 (재클러스터링 필요) |

**IVF + PQ (Product Quantization)**: 각 벡터를 여러 서브벡터로 쪼개고 각각을 양자화하여 메모리를 수십 배 압축. 대규모 corpus에 효과적.

### 5.5.3 주요 Vector DB 비교

| | FAISS | Pinecone | Weaviate | Chroma |
|---|---|---|---|---|
| **타입** | 라이브러리 | 관리형 서비스 | 오픈소스 DB | 오픈소스 |
| **배포** | 로컬 | 클라우드 | 자체 호스팅 or 클라우드 | 로컬 or 클라우드 |
| **ANN 알고리즘** | IVF, HNSW, PQ | HNSW | HNSW | HNSW |
| **필터링** | 제한적 | 강력한 메타데이터 필터 | GraphQL 쿼리 | 메타데이터 필터 |
| **확장성** | 단일 머신 | 자동 확장 | 분산 지원 | 소~중규모 |
| **주요 용도** | 연구, 프로토타입 | 프로덕션 SaaS | 엔터프라이즈 | 개발, 소규모 |

**FAISS 예제:**
```python
import faiss
import numpy as np

d = 768          # 벡터 차원
n_vectors = 100000

# IVF + HNSW 인덱스 생성
quantizer = faiss.IndexHNSWFlat(d, 32)  # coarse quantizer
index = faiss.IndexIVFFlat(quantizer, d, 1024)  # nlist=1024 클러스터

# 학습 (클러스터 중심점 학습)
vectors = np.random.rand(n_vectors, d).astype('float32')
index.train(vectors)
index.add(vectors)

# 검색
query = np.random.rand(1, d).astype('float32')
index.nprobe = 10  # 탐색할 클러스터 수
distances, indices = index.search(query, k=10)
```

### Exact Search vs Approximate Search 트레이드오프

| | Exact Search | ANN (HNSW) | ANN (IVF+PQ) |
|---|---|---|---|
| **Recall@10** | 100% | ~98% | ~90% |
| **QPS** (10M 벡터) | 매우 낮음 | 높음 | 매우 높음 |
| **메모리** | 많음 | 많음 | 적음 (압축) |
| **적합 규모** | <100K | <10M | >10M |

---

## 5.6 Reranking

### Two-Stage Retrieval 아키텍처

검색 품질을 높이기 위해 **두 단계 검색(two-stage retrieval)**을 사용함:

```
Query
  ↓
[Stage 1: Bi-encoder Retrieval]
  빠르고 저렴, Top-100 후보 선별
  ↓
[Stage 2: Cross-encoder Reranking]
  느리지만 정확, Top-5로 압축
  ↓
LLM Generation
```

### Bi-encoder (현재 방식)

```
Query  → [Encoder] → q_vec
Doc    → [Encoder] → d_vec
Score  = cosine_sim(q_vec, d_vec)
```

- Query와 document를 **독립적으로** 인코딩 → 미리 계산 가능
- 매우 빠르지만, 두 텍스트 간의 세밀한 상호작용을 포착하기 어려움

### Cross-encoder Reranker

```
[Query + Document] → [Encoder] → Score
```

- Query와 document를 **함께** 인코딩 → 상호작용을 직접 모델링
- 훨씬 정확하지만 모든 문서 쌍에 대해 inference 필요 → 느림
- 따라서 stage 1에서 선별한 소수의 후보에만 적용

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Stage 1: Bi-encoder로 top-100 검색
candidates = bi_encoder_retrieval(query, k=100)

# Stage 2: Cross-encoder로 rerank
pairs = [[query, doc.page_content] for doc in candidates]
scores = reranker.predict(pairs)

# 점수 기준 정렬
reranked = sorted(
    zip(candidates, scores),
    key=lambda x: x[1],
    reverse=True
)
top_5 = [doc for doc, score in reranked[:5]]
```

### ColBERT-style Late Interaction

**ColBERT (Contextualized Late Interaction over BERT)**는 bi-encoder와 cross-encoder의 중간에 해당하는 접근법.

```
Query:  [q₁, q₂, q₃, q₄]  → [Encoder] → [v_q1, v_q2, v_q3, v_q4]
Doc:    [d₁, d₂, ..., dₙ] → [Encoder] → [v_d1, v_d2, ..., v_dn]

Score = Σᵢ max_j cosine_sim(v_qi, v_dj)   ← MaxSim 연산
```

각 query 토큰이 document의 **가장 유사한 토큰과의 유사도**를 찾는 "late interaction"을 사용함.

**장점**: Cross-encoder보다 빠르면서 bi-encoder보다 정확
**단점**: 벡터 저장 비용이 bi-encoder보다 N배 (N: 문서 토큰 수) 더 필요

---

## 5.7 Query Rewriting

원본 query가 좋지 않거나 (너무 짧거나, 모호하거나), 검색 index에 적합하지 않을 수 있음. **Query rewriting**은 retrieval 전에 query를 변환하여 검색 품질을 높임.

### 5.7.1 HyDE (Hypothetical Document Embedding)

**아이디어**: Query를 직접 검색하는 대신, "이 query에 대한 가상의 이상적인 답변 문서"를 LLM으로 생성하고, 그 가상 문서의 embedding으로 검색함.

```
Query: "블랙홀은 왜 빛을 흡수하는가?"
  ↓ LLM으로 가상 답변 생성
Hypothetical Document:
  "블랙홀은 일반 상대성 이론에 따라 공간을 극도로 휘게 만드는
   질량을 가진 천체다. 탈출 속도가 빛의 속도를 초과하기 때문에
   빛조차 빠져나오지 못한다. 슈바르츠실트 반지름 내부에서는..."
  ↓ 가상 문서 embedding으로 검색
실제 관련 문서 검색 (Query embedding보다 훨씬 풍부한 검색)
```

```python
def hyde_retrieval(query, llm, vectorstore, k=5):
    # 가상 문서 생성
    hypothetical_doc = llm.generate(
        f"다음 질문에 대한 상세한 답변을 작성하세요: {query}"
    )

    # 가상 문서로 검색 (query 대신)
    results = vectorstore.similarity_search(hypothetical_doc, k=k)
    return results
```

### 5.7.2 Query Expansion

Query에 관련 키워드나 동의어를 추가하여 더 많은 관련 문서를 포착함.

```
원본: "LLM 학습"
확장: "LLM 학습 OR 언어모델 훈련 OR large language model training OR
       GPT fine-tuning OR 트랜스포머 pretraining"
```

또는 LLM을 사용해 자동으로 확장함:
```python
expansion_prompt = f"""
쿼리: {query}

이 쿼리와 관련된 다른 검색어 5개를 생성하세요.
동의어, 관련 개념, 다른 표현 방식을 포함하세요.
"""
expanded_queries = llm.generate(expansion_prompt)
```

### 5.7.3 Step-Back Prompting

**아이디어**: 구체적인 질문을 더 일반적인(추상화된) 질문으로 변환한 뒤, 그 일반적 질문으로 먼저 배경 지식을 검색하고, 원래 질문과 합쳐서 답변함.

```
구체적 질문:
"2024년 파리 올림픽에서 한국 양궁팀의 성적은?"

Step-back 질문 (LLM이 생성):
"올림픽에서 양궁 경기는 어떻게 진행되며, 한국 양궁의 역사적 강점은?"

→ Step-back 질문으로 배경 지식 검색
→ 원본 질문과 배경 지식을 함께 사용하여 최종 답변 생성
```

Google DeepMind의 연구에 따르면 step-back prompting은 특히 물리학, 화학 같은 원리 기반 질문에서 성능을 크게 향상시킴.

### Multi-Query Retrieval

하나의 query 대신 **여러 변형 query를 생성**하고 각각으로 검색한 결과를 합침:

```python
def multi_query_retrieval(query, llm, vectorstore, n_variants=3):
    # 다양한 관점의 query 생성
    variants_prompt = f"""
    다음 질문을 {n_variants}가지 다른 방식으로 표현하세요:
    원본: {query}
    """
    variants = llm.generate(variants_prompt).split('\n')

    # 모든 변형으로 검색
    all_results = []
    for v in [query] + variants:
        results = vectorstore.similarity_search(v, k=3)
        all_results.extend(results)

    # 중복 제거 & 재정렬
    unique_results = deduplicate(all_results)
    return unique_results[:5]
```

---

## 5.8 Sliding Window Inference

### 긴 문서 처리의 문제

Context window를 초과하는 긴 문서(예: 100페이지 PDF, 긴 소설)를 처리해야 할 때 사용하는 전략.

### Sliding Window 방식

```
문서 길이: 10,000 tokens
Window 크기: 2,048 tokens
Stride: 1,500 tokens (overlap = 548 tokens)

Window 1: 토큰 [0, 2048]
Window 2: 토큰 [1500, 3548]
Window 3: 토큰 [3000, 5048]
...

각 window에서 LLM 실행 → 결과 집계
```

```python
def sliding_window_inference(text, llm, window_size=2048, stride=1500):
    tokens = tokenizer.encode(text)
    results = []

    for start in range(0, len(tokens), stride):
        end = min(start + window_size, len(tokens))
        window_tokens = tokens[start:end]
        window_text = tokenizer.decode(window_tokens)

        result = llm.generate(window_text)
        results.append({
            'start': start,
            'end': end,
            'output': result
        })

        if end == len(tokens):
            break

    return aggregate_results(results)
```

### 결과 집계 (Aggregation) 전략

- **요약 합산**: 각 window의 요약을 마지막에 한번 더 요약
- **Map-Reduce**: Map 단계에서 각 window 처리 → Reduce 단계에서 합산
- **최다 투표(Majority Vote)**: 분류 태스크에서 각 window의 예측을 투표로 집계

---

## 5.9 Memory Compression

### 왜 Memory Compression이 필요한가?

대화가 길어질수록 이전 대화 내용이 context window를 채워나감. 오래된 대화 내용을 그대로 유지하면:
- Context window 낭비
- 오래된 정보가 최신 정보에 비해 중요도가 낮음

### Summarization-based Compression

이전 대화를 **요약**하여 압축된 형태로 유지함:

```
[초기 상태]
Context: [대화 1][대화 2][대화 3][대화 4][대화 5]

[압축 후]
Context: [대화 1~3의 요약][대화 4][대화 5]
```

```python
class ConversationMemory:
    def __init__(self, max_tokens=4000, summary_threshold=3000):
        self.messages = []
        self.summary = ""
        self.max_tokens = max_tokens
        self.summary_threshold = summary_threshold

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

        # 토큰 수가 임계값 초과시 압축
        if self.count_tokens() > self.summary_threshold:
            self.compress()

    def compress(self):
        # 오래된 메시지들 요약
        old_messages = self.messages[:-4]  # 최근 4개 제외
        self.messages = self.messages[-4:]  # 최근 4개 유지

        summary_prompt = f"다음 대화를 핵심만 요약하세요:\n{format(old_messages)}"
        new_summary = llm.generate(summary_prompt)

        # 기존 요약과 합치기
        if self.summary:
            self.summary = f"{self.summary}\n{new_summary}"
        else:
            self.summary = new_summary
```

### MemGPT 접근법

2023년 발표된 **MemGPT** (Memory GPT)는 OS의 가상 메모리 시스템에서 영감을 받은 계층적 메모리 구조를 제안했음:

```
┌─────────────────────────────────────┐
│          Main Context               │  ← LLM이 직접 접근 가능
│  [System Prompt] [Working Memory]   │    (빠름, 작음)
│  [FIFO Queue of Conversation]       │
└─────────────┬───────────────────────┘
              │ 메모리 관리 함수 호출
┌─────────────▼───────────────────────┐
│       External Storage              │  ← 검색을 통해 접근
│  [Archival Memory] [Recall Memory]  │    (느림, 매우 큼)
└─────────────────────────────────────┘
```

- **Working Memory**: 현재 작업과 직접 관련된 최근 정보
- **Recall Memory**: 이전 대화 내용 (검색 가능한 형태로 저장)
- **Archival Memory**: 장기 지식 베이스

LLM이 특수 함수 (`archival_memory_insert`, `recall_memory_search` 등)를 호출하여 스스로 메모리를 관리함.

### Hierarchical Memory

더 일반화된 계층적 메모리 구조:

```
L0 (즉각적 컨텍스트): 현재 대화 3~5턴 (완전한 형태)
L1 (작업 기억): 현재 세션 요약 (압축됨)
L2 (에피소드 기억): 이전 세션들의 요약 (더 압축됨)
L3 (의미 기억): 사용자에 대한 장기 프로파일 (고도 압축)
```

---

## 5.10 Recurrent Memory Transformers

### 표준 Transformer의 한계

표준 Transformer는 각 segment를 독립적으로 처리함. Segment 경계를 넘어서는 정보를 전달하려면 이전 내용 전체를 context에 유지해야 함.

### Recurrent Memory Token 주입

**Recurrent Memory Transformer (RMT)** 는 각 segment 처리 시 **learnable memory token**들을 추가하고, 이를 다음 segment로 전달함:

```
Segment 1:
Input: [mem_tokens(이전)] + [token 1~512] + [mem_tokens(다음)]
           ↓                                      ↓
      (초기화 또는             정상 처리         다음 segment로
       이전 segment에서)                          전달될 요약

Segment 2:
Input: [mem_tokens(segment1 출력)] + [token 513~1024] + [mem_tokens(다음)]
```

Memory token들은 gradient를 통해 학습되며, **세그먼트 간에 중요한 정보를 선택적으로 전달**하는 법을 배움.

이론적으로 segment 수를 늘리면 무한한 길이의 시퀀스를 처리할 수 있지만, 실제로는 매우 긴 sequence에서 memory token의 정보 bottleneck이 문제가 됨.

### Mamba: Recurrent Memory로서의 State Space Model

**Mamba** (2023, Gu & Dao)는 **State Space Model (SSM)**에 기반한 아키텍처로, Transformer의 attention을 대체함.

핵심 수식:
```
h'(t) = A · h(t) + B · x(t)   ← state 업데이트 (recurrent)
y(t) = C · h(t)                ← 출력

A: state transition matrix (학습됨)
B: input projection (학습됨)
C: output projection (학습됨)
h(t): hidden state (recurrent memory 역할)
```

**선택적 처리 (Selective State Spaces)**가 Mamba의 핵심:
- `A`, `B`, `C`가 입력 `x(t)`에 따라 **동적으로 변화**함
- 이를 통해 중요한 정보는 state에 유지하고, 불필요한 정보는 빠르게 잊음

```
Transformer attention: O(N²) 메모리, O(N²) 연산
Mamba:                 O(N) 메모리,  O(N) 연산
```

**장점**:
- 매우 긴 sequence에서 O(N) 효율
- RNN처럼 recurrent 추론 가능 (inference 시 hidden state 크기 고정)
- Transformer에 준하는 성능

**단점**:
- 이론적으로는 hidden state `h(t)` 크기에 모든 정보를 압축해야 하므로, 매우 긴 distance의 정확한 recall이 필요한 task에서 Transformer 대비 약함
- 학습 패러다임이 Transformer와 달라서 기존 인프라 재사용 어려움

실제로는 **Mamba + Transformer의 Hybrid** 아키텍처 (예: Jamba, Zamba)가 두 방식의 장점을 결합하는 방향으로 발전하고 있음.

---

## 정리: Long Context & Memory 전략 선택 가이드

```
질문: 얼마나 많은 정보가 필요한가?

[수백 KB ~ GB의 문서 코퍼스]
  → RAG 사용
    → 품질 향상 필요? → Reranking + Query Rewriting 추가
    → 최신성 중요? → 정기적인 인덱스 업데이트

[긴 단일 문서 (context window 초과)]
  → Sliding Window Inference
    → 또는 요약 후 RAG

[긴 다중 턴 대화]
  → Memory Compression (Summarization)
    → 정교한 시스템 필요? → MemGPT 스타일 계층적 메모리

[긴 sequence를 처리하는 새 모델 설계]
  → Context Extension (YaRN, LongRoPE)
    → 또는 Mamba/SSM 계열 아키텍처 고려
```

---

*다음 섹션: Section 6 - Reasoning & Test-Time Compute*
