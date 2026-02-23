# Data Pipeline & Processing

## 데이터 수집

### 웹 크롤링

```
Common Crawl (CC):
  가장 큰 공개 웹 크롤링 데이터
  매달 새 크롤링 추가 (~20-40TB/crawl, HTML)
  WARC 포맷 (Web ARChive) + WET (텍스트 추출본)

  URL 규모: 수천억 개
  총 데이터: 수백 PB

텍스트 추출 도구:
  trafilatura: 주요 콘텐츠 추출 (광고, 내비게이션 제거)
  jusText: 문단 수준 필터링 (boilerplate 제거)
  resiliparse: 빠른 대규모 처리 (CC 전용 최적화)

  pip install trafilatura
  text = trafilatura.extract(html_content)
```

### 특화 소스

```
코드: GitHub, StackOverflow, CodeSearchNet
  The Stack: 3.1TB 코드 (500+ 언어)
  StarCoder data: GitHub Stars 기반 필터링

학술/과학: arXiv, PubMed, S2ORC
  LaTeX → 텍스트 변환 필요
  수식 처리 (UniMath, LaTeXML)

책: Project Gutenberg, Books3, OpenLibrary

Wikipedia: Wikidump, 다국어 지원
  kiwix로 오프라인 접근 가능

수학: ProofWiki, Mathematics Stack Exchange

다국어: CC-100 (100개 언어), mC4, OPUS
```

---

## 데이터 정제 파이프라인

### 1단계: URL 필터링

```python
# URL 기반 도메인 필터링
BLOCKED_DOMAINS = {
    "pornhub.com", "xvideos.com",  # 성인 콘텐츠
    "reddit.com/r/nsfw",           # 특정 서브레딧
    "spam-domain.com",             # 스팸
}

def url_filter(url: str) -> bool:
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.lower()
    return not any(blocked in domain for blocked in BLOCKED_DOMAINS)

# 품질 높은 도메인에 가중치 부여
QUALITY_DOMAINS = {
    "wikipedia.org": 2.0,
    "arxiv.org": 1.5,
    "stackoverflow.com": 1.3,
}
```

### 2단계: 언어 탐지

```python
import fasttext

# fastText 언어 분류기 (176개 언어)
model = fasttext.load_model('lid.176.bin')

def detect_language(text: str) -> tuple[str, float]:
    # 첫 줄의 개행 제거 (fasttext 요구사항)
    text_clean = text.replace('\n', ' ')[:512]
    predictions = model.predict(text_clean, k=1)
    lang = predictions[0][0].replace('__label__', '')
    confidence = predictions[1][0]
    return lang, confidence

# 영어 필터링 예시
def is_english(text: str, threshold: float = 0.65) -> bool:
    lang, conf = detect_language(text)
    return lang == 'en' and conf >= threshold
```

### 3단계: 중복 제거 (Deduplication)

#### MinHash LSH

```python
from datasketch import MinHash, MinHashLSH
import re

def get_shingles(text: str, n: int = 5) -> set:
    """n-gram 단위 문자 집합 생성"""
    text = re.sub(r'\s+', ' ', text.lower().strip())
    return {text[i:i+n] for i in range(len(text)-n+1)}

def compute_minhash(text: str, num_perm: int = 128) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    shingles = get_shingles(text)
    for shingle in shingles:
        mh.update(shingle.encode('utf-8'))
    return mh

# LSH 인덱스 구축
lsh = MinHashLSH(threshold=0.8, num_perm=128)

def deduplicate_dataset(documents: list) -> list:
    """
    Jaccard 유사도 0.8 이상 문서를 중복으로 처리
    """
    unique_docs = []

    for doc_id, text in enumerate(documents):
        mh = compute_minhash(text)

        # 유사한 문서 검색
        similar = lsh.query(mh)

        if not similar:  # 중복 없음
            lsh.insert(f"doc_{doc_id}", mh)
            unique_docs.append(text)
        # 중복이면 버림

    return unique_docs

# 스케일링: 수십억 문서 처리
# → 분산 환경 (Spark, Dask)에서 구현 필요
```

#### Suffix Array 기반 정확한 중복 제거

```
ExactSubstr (Lee et al., 2022):
  Suffix Array + BWT (Burrows-Wheeler Transform)
  정확한 n-gram 수준 중복 탐지

  1. 모든 문서를 하나의 큰 텍스트로 연결
  2. Suffix Array 구축 (O(n log n))
  3. 인접한 suffix 비교 → 겹치는 substring 탐지
  4. 특정 길이(50+ 토큰) 이상의 중복 substring 표시
  5. 표시된 span을 학습 데이터에서 마스킹

The Pile의 중복 제거에 사용
```

### 4단계: 품질 필터링

#### 규칙 기반 필터링

```python
def quality_filter(text: str) -> bool:
    """
    FineWeb, C4 스타일 규칙 기반 필터링
    """
    words = text.split()

    # 최소/최대 길이
    if len(words) < 50 or len(words) > 100_000:
        return False

    # 특수문자 비율 (너무 높으면 스팸/코드 등)
    special_char_ratio = sum(
        1 for c in text if not c.isalnum() and not c.isspace()
    ) / max(len(text), 1)
    if special_char_ratio > 0.25:
        return False

    # 단어 반복 비율 (너무 낮으면 repetitive)
    if len(set(words)) / max(len(words), 1) < 0.2:
        return False

    # 줄 당 평균 단어 수
    lines = [l for l in text.split('\n') if l.strip()]
    if lines:
        avg_words_per_line = sum(len(l.split()) for l in lines) / len(lines)
        if avg_words_per_line < 3:  # 너무 짧은 줄들
            return False

    # Bullet point / 리스트 비율
    bullet_ratio = sum(1 for l in lines if l.strip().startswith(
        ('•', '-', '*', '·', '>')
    )) / max(len(lines), 1)
    if bullet_ratio > 0.9:  # 거의 모두 리스트
        return False

    # 대문자 비율
    alpha_chars = [c for c in text if c.isalpha()]
    if alpha_chars:
        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if upper_ratio > 0.4:  # 지나치게 많은 대문자
            return False

    # 마지막 문장 완성 여부
    if not text.rstrip().endswith(('.', '!', '?', '"', "'", '…')):
        pass  # 엄격하게 적용하지 않아도 됨

    return True
```

#### Perplexity 필터링

```python
import kenlm
import math

# KenLM n-gram 언어 모델 (미리 학습된)
lm = kenlm.Model('en_wiki_5gram.arpa')

def compute_perplexity(text: str, model: kenlm.Model) -> float:
    """문자 수준 perplexity 계산"""
    total_log_prob = 0
    n_words = 0

    for log_prob, ngram_len, _ in model.full_scores(text):
        total_log_prob += log_prob
        n_words += 1

    if n_words == 0:
        return float('inf')

    avg_log_prob = total_log_prob / n_words
    return math.pow(10, -avg_log_prob)

def perplexity_filter(text: str, model, min_ppl=10, max_ppl=500) -> bool:
    """
    너무 높은 PPL (이상한 텍스트) → 제거
    너무 낮은 PPL (너무 단순/반복) → 제거
    """
    ppl = compute_perplexity(text, model)
    return min_ppl <= ppl <= max_ppl

# C4 필터링 임계값:
# 중앙 배포의 10~98% 사이 PPL만 유지
```

#### 분류기 기반 필터링

```python
# FineWeb 방식: 품질 분류기 훈련
# 고품질 (Wikipedia 등) vs 저품질 이진 분류기

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer

def train_quality_classifier(high_quality_texts, low_quality_texts):
    """
    간단한 품질 분류기 학습
    실제로는 더 복잡한 모델 사용 (fastText, DeBERTa 등)
    """
    vectorizer = HashingVectorizer(ngram_range=(1,2), n_features=2**18)

    X_high = vectorizer.transform(high_quality_texts)
    X_low = vectorizer.transform(low_quality_texts)

    import scipy.sparse as sp
    X = sp.vstack([X_high, X_low])
    y = [1] * len(high_quality_texts) + [0] * len(low_quality_texts)

    clf = LogisticRegression(C=0.1, max_iter=1000)
    clf.fit(X, y)
    return clf, vectorizer

# FineWeb (HuggingFace, 2024):
# 15T tokens, CC 기반
# EDU quality scorer (교육적 가치 평가)
# GPT-3.5로 교육적 가치 점수 → 분류기 학습

# DCLM (DataComp-LM, 2024):
# 표준화된 필터링 파이프라인
# fastText 분류기로 Wikipedia/Books와 유사한 텍스트 선별
```

---

## Tokenization & Packing

### 효율적 시퀀스 패킹

```python
def pack_sequences(
    documents: list,
    max_length: int,
    eos_token_id: int,
    tokenizer
) -> list:
    """
    여러 문서를 하나의 긴 시퀀스에 이어붙임
    패딩 없이 GPU 활용률 최대화
    """
    packed_sequences = []
    current_tokens = []

    for doc in documents:
        # 문서 토큰화
        tokens = tokenizer.encode(doc) + [eos_token_id]

        if len(current_tokens) + len(tokens) > max_length:
            if current_tokens:
                # 현재 버퍼 저장 (남은 공간은 패딩)
                packed_sequences.append(
                    current_tokens + [tokenizer.pad_token_id] * (
                        max_length - len(current_tokens)
                    )
                )
                current_tokens = []

        # 문서가 단독으로 max_length 초과
        if len(tokens) > max_length:
            # Truncate and split
            for i in range(0, len(tokens), max_length):
                chunk = tokens[i:i+max_length]
                if len(chunk) == max_length:
                    packed_sequences.append(chunk)
        else:
            current_tokens.extend(tokens)

    # 마지막 버퍼 처리
    if current_tokens:
        packed_sequences.append(
            current_tokens + [tokenizer.pad_token_id] * (
                max_length - len(current_tokens)
            )
        )

    return packed_sequences

# Document-aware attention mask:
# 서로 다른 문서 간 attention 차단 (교차 오염 방지)
# Flash Attention의 varlen (variable-length) 기능 활용
```

### 효율적 데이터 로딩

```python
from torch.utils.data import IterableDataset
import numpy as np

class StreamingTokenDataset(IterableDataset):
    """
    대규모 데이터를 메모리에 올리지 않고 스트리밍 처리
    """
    def __init__(self, data_files, seq_len, tokenizer):
        self.data_files = data_files
        self.seq_len = seq_len
        self.tokenizer = tokenizer

    def __iter__(self):
        buffer = []

        for file_path in self.data_files:
            with open(file_path) as f:
                for line in f:
                    doc = json.loads(line)["text"]
                    tokens = self.tokenizer.encode(doc)
                    tokens.append(self.tokenizer.eos_token_id)
                    buffer.extend(tokens)

                    while len(buffer) >= self.seq_len:
                        yield {
                            "input_ids": torch.tensor(buffer[:self.seq_len]),
                            "labels": torch.tensor(buffer[:self.seq_len])
                        }
                        buffer = buffer[self.seq_len:]
```

---

## 데이터 혼합 전략

### Proportional Sampling

```python
class MixedDataLoader:
    """
    여러 도메인 데이터를 지정 비율로 혼합 샘플링
    """
    domain_weights = {
        "web": 0.50,          # Common Crawl 등
        "code": 0.17,         # GitHub, Stack
        "books": 0.12,        # Books, arXiv
        "wikipedia": 0.10,    # Wikipedia
        "academic": 0.06,     # 학술 논문
        "conversation": 0.05, # 대화, Reddit
    }

    def __init__(self, domain_dataloaders: dict, domain_weights: dict):
        self.loaders = domain_dataloaders
        self.weights = domain_weights
        self.iterators = {k: iter(v) for k, v in domain_dataloaders.items()}

    def __iter__(self):
        domains = list(self.weights.keys())
        probs = [self.weights[d] for d in domains]

        while True:
            # 확률에 따라 도메인 선택
            chosen = np.random.choice(domains, p=probs)
            try:
                yield next(self.iterators[chosen])
            except StopIteration:
                self.iterators[chosen] = iter(self.loaders[chosen])
                yield next(self.iterators[chosen])
```

### Curriculum Learning

```
초기 학습: 짧고 쉬운 텍스트
  - 평균 문장 길이 짧음
  - 일반적인 주제
  - 형식적인 언어

나중 학습: 길고 어려운 텍스트
  - 긴 문서, 긴 컨텍스트
  - 전문 도메인 (수학, 코딩, 과학)
  - 추론이 필요한 텍스트

구현:
  초기 epoch: 데이터셋의 "쉬운" 부분 (PPL 필터)
  나중 epoch: 더 다양한 데이터 포함

데이터 혼합 변화:
  0-50%: web 60%, code 15%, books 15%, wiki 10%
  50-80%: web 40%, code 25%, books 20%, wiki 15%
  80-100%: code 30%, academic 30%, books 20%, web 20%
```

---

## PII 제거 (개인정보)

```python
import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

def remove_pii(text: str) -> str:
    """
    개인식별정보(PII) 탐지 및 제거/대체
    """
    # 패턴 기반 (빠름)
    patterns = {
        # 이메일
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]',
        # 미국 전화번호
        r'\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b': '[PHONE]',
        # 신용카드번호 (간단한 패턴)
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b': '[CREDITCARD]',
        # IP 주소
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b': '[IP]',
        # 소셜 시큐리티 넘버 (미국)
        r'\b\d{3}-\d{2}-\d{4}\b': '[SSN]',
    }

    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text)

    # Presidio (Microsoft): 더 정확한 NER 기반 PII 탐지
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    results = analyzer.analyze(text=text, language='en')
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)

    return anonymized.text
```

---

## 주요 데이터셋 비교

| 데이터셋 | 크기 | 특징 | 출처 |
|---------|------|------|------|
| Common Crawl | 수백 PB | 웹 원본 (raw) | 웹 크롤링 |
| C4 (Colossal Clean Crawled Corpus) | 305GB | CC 정제 버전 | T5 논문 |
| The Pile | 825GB | 22개 소스 혼합 | EleutherAI |
| RedPajama-v2 | 30T tokens | CC 기반 대규모 | Together AI |
| FineWeb | 15T tokens | 고품질 CC 필터링 | HuggingFace |
| FineWeb-Edu | 1.3T tokens | 교육적 콘텐츠 | HuggingFace |
| DCLM-baseline | 4T tokens | 표준 필터링 | DataComp |
| Dolma | 3T tokens | 다소스 혼합 | Allen AI |
| ROOTS | 1.6TB | 다국어 (59개 언어) | BigScience |
| SlimPajama | 627B tokens | RedPajama 정제 | Cerebras |

---

## 합성 데이터 (Synthetic Data)

### Phi-1 / Textbooks Are All You Need (Microsoft, 2023)

```
GPT-4로 "교육적" 합성 데이터 생성:
  교과서 스타일 설명 (다양한 개념)
  연습 문제와 풀이
  다양한 난이도

결과:
  1.3B 파라미터로 GPT-4에 준하는 코딩 능력
  데이터 품질 > 데이터 양 입증

Phi 시리즈 전략:
  "textbooks only": 교육적, 설명적, 논리적 텍스트
  웹 텍스트 대신 합성 교과서 → 더 효율적 학습
```

### Magpie (Meta, 2024)

```python
# LLaMA-3-Instruct에서 자기 지시 데이터 자동 생성
def generate_magpie_data(model, tokenizer, n_samples=100000):
    """
    사용자 turn 시작 토큰까지만 제공 → 모델이 질문 자동 생성
    """
    # LLaMA-3 user header까지만 제공
    user_prefix = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"

    questions = []
    for _ in range(n_samples):
        # 모델이 자연스럽게 질문 생성
        input_ids = tokenizer(user_prefix, return_tensors="pt").input_ids
        output = model.generate(input_ids, max_new_tokens=200, temperature=1.0)
        question = tokenizer.decode(output[0], skip_special_tokens=False)

        # 질문 + 답변 생성
        full_response = model.generate(output, max_new_tokens=512)
        questions.append(extract_qa(full_response))

    return questions

# 결과: 3M+ 고품질 SFT 데이터 (인간 annotation 없음)
```

### Cosmopedia (HuggingFace, 2024)

```
Mixtral-8x7B로 교과서/이야기 스타일 합성 데이터
  25B 토큰 규모
  다양한 주제, 교육적 스타일
  여러 타겟 독자 설정 (초등학생, 대학생, 전문가)
```

### 합성 데이터의 한계

```
Model Collapse:
  모델 생성 데이터로 모델 학습 반복
  → 분포 수렴 → 다양성 감소
  해결: 실제 데이터와 비율 조정 (30~50% 실제 데이터 유지)

편향 증폭:
  기반 모델의 편향이 합성 데이터에 반영
  → 정치적 편향, 사실 오류 증폭 가능

사실 정확성:
  LLM이 사실과 다른 합성 데이터 생성 가능
  검증 없이 사용 위험
```

---

## FineWeb 파이프라인 (HuggingFace 표준)

```
1단계: CC WARC → 텍스트 추출 (trafilatura)
2단계: URL 블랙리스트 필터링
3단계: 언어 탐지 (fastText ≥ 0.65)
4단계: Quality filtering (규칙 기반)
   - 반복 라인 비율 < 0.3
   - 짧은 라인 비율 < 0.67
   - 특수문자 비율 < 0.1
   - 최소 단어 수 ≥ 100
5단계: MinHash dedup (5-gram, threshold=0.7)
6단계: C4 필터 호환성 적용
결과: 15T tokens 고품질 데이터

FineWeb-Edu (추가 단계):
7단계: LLaMA-3 교육적 점수 1-5 부여
8단계: Score ≥ 3 이상만 선택
결과: 1.3T tokens 교육적 고품질 데이터
```

---

## Further Questions

**Q. 데이터 중복 제거가 왜 중요한가?**
> 1) 모델 암기 방지 (privacy, test set contamination). 2) 일반화 성능 향상 (중복 ≠ 새 지식). 3) 학습 효율 향상 (같은 정보 반복 학습 = 낭비). 중복 학습: "Grokking" 현상에서 필요할 수도 있지만, 대규모 pretraining에서는 손해. MinHash: 문서 수준 (~80% 유사도). Suffix Array: n-gram 수준 정확한 중복.

**Q. Sequence packing의 장단점은?**
> 장점: GPU 활용률 극대화 (패딩 없음), 학습 속도 향상. 단점: 여러 문서 내용이 하나의 sequence에 섞임 → cross-document attention 차단 필요. Flash Attention의 varlen이 이를 지원 (attention_mask로 문서 경계 설정). 문서 경계 마스킹 없이 패킹하면 문서 간 오염 발생 가능.

**Q. 합성 데이터의 한계는?**
> 1) Model collapse: 모델로 모델 학습 반복 → 분포 수렴. 2) 실제 세계 노이즈/다양성 부재. 3) 모델 편향 증폭. 4) 사실 오류 가능성. 실제 데이터와 혼합 (30-50%) 권장. FineWeb이나 Dolma 같은 고품질 실제 데이터로 앵커링.

**Q. FineWeb이 기존 필터링 데이터셋보다 나은 이유는?**
> 기존 CC 데이터셋 (C4, RedPajama)보다 체계적인 필터링. EDU scorer: 교육적 가치가 높은 텍스트 선별 → 학습 효율 향상. HuggingFace가 필터링 파이프라인을 오픈소스로 공개 → 재현 가능. 실제 LLM 학습 실험으로 효과 검증. 동일 compute에서 더 나은 성능 달성.
