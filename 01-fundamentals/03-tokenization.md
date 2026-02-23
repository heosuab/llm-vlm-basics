# Tokenization

## 왜 토크나이저가 중요한가

```
모델의 실제 "입력 언어":
  텍스트 → 토큰 시퀀스 → 모델 처리
  토크나이저는 텍스트와 모델 사이의 인터페이스

핵심 영향:
  1. 시퀀스 길이: 비효율 토크나이저 → 긴 시퀀스 → 비용 증가
  2. 모델 능력: 숫자, 코드, 다국어 처리 품질 직결
  3. 컨텍스트 활용: 같은 컨텍스트 창으로 더 많은 정보 처리 가능
  4. 학습 효율: 같은 배치 크기로 더 많은 텍스트 처리

한국어 예시:
  GPT-2 BPE: "안녕하세요" → 15+ 토큰
  cl100k (GPT-4): "안녕하세요" → 5~8 토큰
  Qwen 토크나이저: "안녕하세요" → 3~4 토큰
  → 한국어 기반 서비스: 토크나이저 효율 = 비용 직결
```

---

## BPE (Byte Pair Encoding)

### 알고리즘
```
핵심 아이디어:
  가장 자주 등장하는 연속된 심볼 쌍을 반복 병합
  → 데이터에 최적화된 subword 어휘 자동 구성

알고리즘 단계:
  1. 초기 어휘: 개별 문자 (또는 바이트)
  2. 모든 심볼 쌍의 빈도 계산
  3. 가장 빈번한 쌍 선택 → 새 토큰으로 병합
  4. 원하는 vocab_size 도달까지 반복

예시:
  코퍼스: "low low lower lowest" (단순화)
  초기: l, o, w, e, r, s, t
  빈도: (l,o)=4, (o,w)=4, (w,_)=3 ...

  1차 병합: (l,o) → lo  빈도 4
  2차 병합: (lo,w) → low  빈도 3
  3차 병합: (low,e) → lowe  빈도 2
  ...

  최종 vocab에 "low", "low", "lowe", "r", "st" 등 포함
```

### BPE 구현
```python
from collections import defaultdict
import re

def get_vocab(corpus):
    """텍스트를 공백 분리 후 문자 단위 vocab 생성"""
    vocab = defaultdict(int)
    for word in corpus.split():
        # 단어를 문자로 분리, 끝에 </w> 마커
        vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def get_stats(vocab):
    """모든 심볼 쌍의 빈도 계산"""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    """가장 빈번한 쌍을 새 토큰으로 병합"""
    new_vocab = {}
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in vocab:
        new_word = pattern.sub(''.join(pair), word)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def bpe_train(corpus, num_merges=10):
    vocab = get_vocab(corpus)
    merges = []

    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
        merges.append(best_pair)
        print(f"Merge {i+1}: {best_pair} -> {''.join(best_pair)}")

    return vocab, merges

# 실행 예시
corpus = "low low lower lower lowest"
vocab, merges = bpe_train(corpus, num_merges=5)
```

### Byte-Level BPE
```
GPT-2에서 도입:
  기본 단위: 개별 문자 대신 바이트 (0-255)
  → 256개 기본 토큰으로 모든 텍스트 표현 가능
  → OOV (Unknown) 토큰 없음

장점:
  이모지, 특수문자, 임의 Unicode 처리 가능
  "Byte-fallback": 희귀 문자도 바이트로 분해

단점:
  희귀 Unicode 문자 → 여러 바이트 토큰 = 긴 시퀀스
  예: 한자 1글자 → 3바이트 → 효율 낮음

GPT-2 vocab: 50,257 (50,000 BPE + 256 bytes + 1 special)
GPT-4 cl100k: 100,277
GPT-4o o200k: 200,019
```

---

## WordPiece

```
BERT, DistilBERT에서 사용

BPE와의 차이:
  BPE: 빈도 기준으로 병합
  WordPiece: 우도(likelihood) 증가 기준으로 병합

  score(a, b) = freq(ab) / (freq(a) × freq(b))
  = P(ab) / (P(a) × P(b))  (PMI와 유사)

직관:
  "ab"가 "a"와 "b"를 독립적으로 보는 것보다
  얼마나 더 자주 등장하는가?

특징:
  subword prefix: ## 로 표시
  "playing" → ["play", "##ing"]
  "unaffable" → ["un", "##aff", "##able"]

장점: 정보 이론적으로 더 의미있는 병합
단점: BPE보다 구현 복잡
```

---

## SentencePiece

```
특징:
  언어별 전처리 불필요 (raw text 직접 처리)
  공백을 ▁ (U+2581) 특수 심볼로 처리
  → "Hello world" = ["▁Hello", "▁world"]

  BPE 또는 Unigram LM 선택 가능
  다국어에 강함 (공백 없는 언어도 처리)

사용 모델:
  LLaMA 1/2/3: SentencePiece BPE, 32K vocab
  Gemma: SentencePiece BPE, 256K vocab
  T5: SentencePiece Unigram, 32K vocab

코드:
import sentencepiece as spm

# 학습
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    model_type='bpe',  # 또는 'unigram'
    character_coverage=0.9999,  # 다국어 커버리지
    pad_id=0, unk_id=1, bos_id=2, eos_id=3
)

# 사용
sp = spm.SentencePieceProcessor()
sp.load('tokenizer.model')
sp.encode("Hello world", out_type=str)
# → ['▁Hello', '▁world']
```

### Unigram Language Model Tokenization
```
목표:
  학습된 단어 확률로 최적 분절 선택
  각 토큰 t_i가 독립 → P(tokenization) = Π P(t_i)

알고리즘:
  1. 큰 초기 vocab 구성 (모든 subword)
  2. EM 알고리즘:
     E-step: 현재 확률로 각 단어의 최적 분절 계산
     M-step: 분절 통계로 토큰 확률 업데이트
  3. 확률이 낮은 토큰 제거
  4. 목표 vocab_size까지 반복

BPE와 차이:
  BPE: 병합 순서 결정론적 (greedy)
  Unigram: 확률 기반, 여러 분절 가능
  → Regularization에 활용 가능 (학습 시 다양한 분절)

SentencePiece Unigram 사용 예:
  T5, mT5 → Unigram 방식으로 다국어 효율적 처리
```

---

## Tiktoken (OpenAI)

```
OpenAI의 빠른 BPE 구현 (Rust 기반)

어휘:
  cl100k_base: 100,277 tokens
    - GPT-3.5, GPT-4, text-embedding-ada-002
    - 한국어 더 효율적 (GPT-2 대비)

  o200k_base: 200,019 tokens
    - GPT-4o
    - 더 넓은 다국어 커버리지
    - 한국어/중국어 압축률 향상

특징:
  Pre-tokenization: 정규식으로 텍스트 분할 먼저
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|
         \p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|
         \s+(?!\S)|\s+"""
  → 숫자: 최대 3자리씩 분리, 공백 처리 등

코드:
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("Hello, world! 안녕하세요")
print(tokens)  # [9906, 11, 1917, 0, ...]
print(enc.decode(tokens))  # "Hello, world! 안녕하세요"

# GPT-4o
enc_o200k = tiktoken.get_encoding("o200k_base")
```

---

## 토크나이저 학습 (실전)

```python
# HuggingFace tokenizers 라이브러리로 BPE 학습
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# 1. BPE 모델 초기화
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# 2. Pre-tokenization (공백 기준 분리)
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 3. Trainer 설정
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

# 4. 학습
files = ["corpus.txt"]
tokenizer.train(files, trainer)

# 5. Decoder 설정
tokenizer.decoder = decoders.ByteLevel()

# 6. 저장
tokenizer.save("my_tokenizer.json")

# 7. HuggingFace AutoTokenizer로 불러오기
from transformers import PreTrainedTokenizerFast
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="my_tokenizer.json")
```

---

## Vocabulary Size 선택

```
Vocab 크기 Trade-off:

vocab 크기 ↑ (예: 100K)
  장점:
    더 긴 단어를 하나의 토큰으로 (시퀀스 짧아짐)
    의미 단위 보존 (다국어, 전문 용어)
    효율적 다국어 처리
  단점:
    Embedding 행렬 크기 ↑ (파라미터 증가)
    희귀 토큰 학습 어려움 (underfit)
    메모리 증가

vocab 크기 ↓ (예: 32K)
  장점:
    모든 토큰 충분한 학습 기회
    Embedding 행렬 작음
  단점:
    다국어 시퀀스 길어짐
    의미 파편화

현대 모델 추세:
  LLaMA-1/2: 32K
  LLaMA-3: 128K (다국어 강화)
  Gemma-2: 256K
  GPT-4: 100K
  GPT-4o: 200K
  Qwen: 152K

128K ~ 256K: 다국어 LLM의 새 표준
```

---

## 토크나이저가 모델 능력에 미치는 영향

### 숫자/수학
```
문제: 숫자 분절 방식이 수학 능력에 영향

GPT-2 방식: "12345" → ["123", "45"] (임의 분절)
GPT-4 방식 (cl100k): "12345" → ["123", "45"] (3자리 단위)

이상적: 자릿수 단위로 분절
  "12345" → ["1", "2", "3", "4", "5"]
  → 덧셈/뺄셈 시 자릿수 정렬 쉬움
  but 시퀀스 길이 증가

연구 결과:
  수학 학습 시 자릿수 분리가 계산 능력 향상에 도움
  하지만 실제 LLM들은 mixed approach
  → Chain-of-Thought으로 보완
```

### 코드
```
공백/들여쓰기:
  Python: 들여쓰기가 문법
  "    " (4 spaces) → [" "] × 4 또는 ["    "] 하나?
  → 공백 압축 토크나이저 유리

코드 식별자:
  camelCase, snake_case 분절 방식
  "getUserName" → ["get", "User", "Name"] or ["getUserName"]

전용 코드 토크나이저:
  Codex, CodeLLaMA: 코드 특화 토크나이저
  코드 코퍼스로 학습 → 코드 패턴에 최적화
```

### 다국어
```
Fertility (출산율) = 단어당 평균 토큰 수
낮을수록 효율적

언어별 fertility 비교 (GPT-4 cl100k 기준):
  영어:   1.3 tokens/word (기준)
  한국어: 1.7 tokens/word
  중국어: 1.5 characters/token (글자당)
  아랍어: 2.5 tokens/word
  히브리어: 2.0 tokens/word

Fertility 영향:
  높은 fertility → 같은 정보에 더 많은 토큰 필요
  → 컨텍스트 창 낭비, 비용 증가

해결:
  언어별 특화 토크나이저
  대형 vocab (128K+)으로 다국어 coverage
```

---

## Special Tokens & Chat Templates

### Special Tokens
```
역할          | BOS/EOS    | PAD      | 기타
──────────────|────────────|──────────|───────────────────
시퀀스 시작   | <s>, [BOS] |          |
시퀀스 끝     | </s>, [EOS]|          |
패딩          |            | [PAD]    |
미등록(레거시)| 없음       |          | [UNK] (byte-level BPE에서 불필요)
마스킹(BERT)  |            |          | [MASK]
화자 구분     |            |          | <|im_start|>, <|im_end|>
```

### Chat Templates
```
# ChatML (OpenAI GPT-4, 많은 오픈소스 모델)
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
What is 2+2?
<|im_end|>
<|im_start|>assistant
4
<|im_end|>

# LLaMA-2 (Meta)
[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

What is 2+2? [/INST] 4 </s><s>[INST] ...

# LLaMA-3 (Meta)
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is 2+2?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
4
<|eot_id|>

# Mistral
[INST] What is 2+2? [/INST] 4 </s>

# Qwen-2
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
What is 2+2?
<|im_end|>
<|im_start|>assistant

HuggingFace 자동 적용:
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
```

---

## 토크나이저 평가 지표

```
Fertility:
  정의: 단어당 평균 토큰 수
  계산: total_tokens / total_words
  해석: 낮을수록 효율적
  영어 기준: 1.3 (GPT-4), 1.5+ (GPT-2)

Compression Rate:
  정의: 원본 문자 수 / 토큰 수
  해석: 높을수록 압축 효율 좋음

Vocabulary Coverage:
  정의: 실제 텍스트의 고유 토큰 중 vocab에 있는 비율
  byte-level BPE: 항상 100% (UNK 없음)
  character-based: rare char가 UNK 될 수 있음

Token Length Distribution:
  평균/중앙값 토큰 길이
  → 길수록 의미 단위 보존

Cross-lingual Parity:
  같은 정보량의 텍스트가 다른 언어에서 비슷한 토큰 수?
  불균형하면 다국어 모델 학습에 편향
```

---

## Token Healing (토큰 치유)

```
문제: LLM이 문장 중간에서 생성 시작 시 경계 문제

예시:
  프롬프트: "The color is bl"
  다음 토큰 예측: "ue" (blue의 일부)
  하지만 실제 토큰이 "bl"로 끊겼으므로 "ue"만 학습됨
  → "blue"가 실제로는 하나의 토큰

Token Healing 해결:
  마지막 부분 토큰을 제거하고 완전한 토큰으로 재생성
  또는 partial token을 허용하는 특수 처리

실제 사용:
  코드 자동완성 (Copilot)
  구조화된 출력 생성
  특수 도구 호출
```

---

## 토크나이저 문제점 & 해결책

```
문제 1: 숫자 분절 불일치
  "100" → 1토큰 but "101" → 2토큰
  → 수학적 패턴 학습 어려움
  해결: 더 일관된 숫자 분절 규칙 (3자리 단위)

문제 2: 공백 처리
  " hello" (앞 공백 있음) ≠ "hello"
  → 프롬프트 형식에 따라 결과 달라짐
  해결: 일관된 채팅 템플릿 사용

문제 3: 대소문자
  "Hello" ≠ "hello" (별개 토큰인 경우)
  → 모델이 대소문자 변형에 취약
  해결: case normalization (but 정보 손실)

문제 4: 영어 중심성
  영어: 효율적 / 비영어: 비효율적
  → API 비용 언어별 큰 차이
  해결: 대형 vocab (128K+), 언어 특화 토크나이저
```

---

## Further Questions

**Q. BPE와 WordPiece의 차이는?**
> 병합 기준 차이. BPE는 단순 빈도(freq(ab)), WordPiece는 정보량(freq(ab)/freq(a)×freq(b) = PMI와 유사). WordPiece는 "an" 같은 흔한 조합보다 "##ing" 같이 함께 등장할 때만 의미있는 쌍을 선호. BPE는 구현 단순, WordPiece는 더 언어학적으로 의미있는 병합.

**Q. 왜 Byte-level BPE를 사용하나?**
> OOV 토큰이 없음 (모든 텍스트 표현 가능). 256바이트로 어떤 Unicode 문자도 커버. 이모지, 특수문자, 임의 바이너리도 처리. 단점은 희귀 Unicode 문자(한자 등)가 3바이트 = 3토큰이 되어 효율 떨어짐.

**Q. 토크나이저가 모델 성능에 미치는 영향은?**
> 수학: 숫자 분절 방식이 자릿수 연산에 영향. 다국어: Fertility 높으면 같은 컨텍스트 창에 더 적은 정보. 코드: 들여쓰기/식별자 처리 방식이 코드 이해에 영향. 중요도: 토크나이저 설계 > 보통의 모델 크기 차이 (실제 다국어 LLM에서 핵심 격차 요인).

**Q. 왜 LLaMA-3는 vocab을 32K에서 128K로 늘렸나?**
> 다국어 능력 강화. 32K에서 비영어 언어(한국어, 아랍어 등)의 Fertility가 높아 비효율. 128K로 늘리면 더 많은 언어의 단어를 직접 토큰으로 처리 → 시퀀스 짧아짐 → 같은 컨텍스트에 더 많은 정보. 단점: Embedding 행렬 ~4× 증가.

**Q. Chat template이 왜 중요한가?**
> 모델이 학습한 특정 포맷에서만 instruction following 성능 최대화. 잘못된 템플릿 사용 시 모델이 "대화 시작"을 인식 못 함 → 성능 급락. HuggingFace apply_chat_template() 자동 사용 권장. 파인튜닝 시에도 베이스 모델의 채팅 템플릿 일치 필요.
