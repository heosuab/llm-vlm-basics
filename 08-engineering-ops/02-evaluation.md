# LLM 평가 (Evaluation)

## 자동 평가

### 전통적 지표

#### Perplexity (PPL)

```python
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_perplexity(model, tokenizer, text, device="cuda"):
    """
    텍스트에 대한 모델의 perplexity 계산
    낮을수록 모델이 텍스트를 잘 예측 (더 좋음)
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)
        loss = outputs.loss  # 평균 NLL loss

    return math.exp(loss.item())

# 모델 비교 예시
# GPT-2: ~40 PPL (WikiText-2)
# GPT-3: ~20 PPL
# LLaMA-3-70B: ~5-8 PPL

# 한계:
# 1. 인간 품질과 상관 낮음 (PPL 낮아도 나쁜 응답 가능)
# 2. 데이터셋에 따라 절대값 다름
# 3. Tokenizer에 의존적
```

#### BLEU (Bilingual Evaluation Understudy)

```
기계 번역 표준 지표:

BLEU = BP × exp(Σ wₙ log pₙ)

BP (Brevity Penalty): 너무 짧은 응답 패널티
  BP = 1               if |c| > |r|
  BP = exp(1 - |r|/|c|) if |c| ≤ |r|

pₙ: n-gram precision
  p₁: 1-gram (단어 일치)
  p₂: 2-gram 연속 단어 일치
  ...
  p₄: 4-gram

BLEU-4: wₙ = 1/4 for n=1..4 (동등 가중치)

예시:
  참조: "The cat sat on the mat"
  생성: "The cat is on the mat"
  p₁ = 5/6 (cat, the, on, the, mat 일치)
  p₂ = 3/5 (2-gram 중 "on the", "the mat" 등 일치)

한계:
  의미보다 표면 겹침
  동의어, 구조 변환 인식 못함
  "Good morning" vs "Hello" → 0점 (의미 같아도)
```

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

```
요약 평가 표준 지표:

ROUGE-N: n-gram recall
  ROUGE-N = |겹치는 n-gram| / |참조의 n-gram|

ROUGE-1 (단어 수준):
  참조: "The quick brown fox"
  생성: "A quick brown dog"
  겹침: quick, brown → ROUGE-1 = 2/4 = 0.5

ROUGE-2 (bigram):
  참조의 bigram: (the quick), (quick brown), (brown fox)
  생성의 bigram: (a quick), (quick brown), (brown dog)
  겹침: (quick brown) → ROUGE-2 = 1/3

ROUGE-L: 최장 공통 부분 수열(LCS) 기반
  순서 유지하면서 겹치는 최대 부분 수열
  F1 = (2 × P_lcs × R_lcs) / (P_lcs + R_lcs)

BERTScore:
  BERT 임베딩 코사인 유사도
  의미 기반 비교
  PPL: 낮을수록 좋음
  BLEU/ROUGE: 높을수록 좋음
```

---

## LLM-as-Judge

### 원리와 구현

```python
import anthropic

def llm_judge(question, response_a, response_b, criteria="helpful and accurate"):
    """
    LLM을 judge로 사용하여 두 응답 비교
    """
    client = anthropic.Anthropic()

    prompt = f"""You are an impartial judge evaluating AI responses.

Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Evaluate which response is better based on: {criteria}

First analyze both responses, then provide your verdict.
End with EXACTLY one of: [[A]], [[B]], or [[C]] (for tie).
"""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    judgment = response.content[0].text
    if "[[A]]" in judgment:
        return "A"
    elif "[[B]]" in judgment:
        return "B"
    else:
        return "tie"

# Position Bias 방지: A/B와 B/A 모두 비교 후 일관성 확인
def unbiased_judge(question, response_a, response_b):
    result_ab = llm_judge(question, response_a, response_b)
    result_ba = llm_judge(question, response_b, response_a)

    # A가 두 번 다 이긴 경우만 A 승리로 인정
    if result_ab == "A" and result_ba == "B":
        return "A"
    elif result_ab == "B" and result_ba == "A":
        return "B"
    else:
        return "tie"
```

### MT-Bench [Zheng et al., 2023]

```
80개 다회전 대화 질문 (8개 카테고리):
  Writing, Roleplay, Reasoning, Math,
  Coding, Extraction, STEM, Humanities

평가: GPT-4가 1~10점 채점

Two-turn 구조:
  Turn 1: 기본 질문 (예: "물리학의 가장 어려운 부분은?")
  Turn 2: 후속 질문 (예: "그 중 가장 최근 발전은?")

주요 결과 (GPT-4 judge):
  GPT-4: 8.99
  Claude-3-Opus: 8.8
  GPT-3.5-turbo: 7.94
  Vicuna-33B: 7.12
```

### AlpacaEval

```
원리:
  생성 응답 vs reference 모델(GPT-4-turbo) 응답
  다른 LLM이 어느 것이 더 나은지 판정
  Win-rate 계산

AlpacaEval 2.0:
  Length-controlled win rate (길이 편향 수정)
  LC(length-controlled) AlpacaEval이 Chatbot Arena와 더 높은 상관
  805개 지시 평가

편향 문제:
  LLM judge: 자신의 출력 선호 (self-serving bias)
  더 긴 응답 선호 (verbosity bias)
  형식적으로 잘 구조화된 응답 선호
```

---

## 주요 벤치마크

### 일반 능력

| 벤치마크 | 설명 | 문항 수 | 특징 |
|---------|------|--------|------|
| MMLU | 57개 과목, 대학 수준 MCQ | 15K | 지식 범위 넓음 |
| MMLU-Pro | MMLU 강화판 (10개 선택지, harder) | 12K | 더 어려움 |
| HellaSwag | 상식 문장 완성 | 70K | 언어 이해 |
| WinoGrande | 대명사 해석 (Winograd 확장) | 1.3K | 상식 추론 |
| ARC-Challenge | 과학 추론 (어려운 문제만) | 1.2K | 초중등 과학 |
| TruthfulQA | 모델이 거짓말하는 경향 측정 | 817 | 사실성 |
| GPQA | 전문가 수준 과학 (PhD) | 448 | 매우 어려움 |
| BIG-Bench Hard | 어려운 추론 태스크 | 6.5K | 다양한 추론 |

### 수학/코딩

| 벤치마크 | 설명 | 특징 |
|---------|------|------|
| GSM8K | 초등 수학 8500문제 | 단계별 풀이 |
| MATH | 경시 수학 12500문제 | AMC/AIME 수준 |
| AMC/AIME | 실제 올림피아드 | 2024 기준 평가 |
| HumanEval | Python 코딩 164문제 | pass@k 측정 |
| MBPP | 파이썬 프로그래밍 500문제 | 기본 코딩 |
| LiveCodeBench | 최신 실제 코딩 문제 | Contamination 없음 |
| SWE-bench | 실제 GitHub 이슈 해결 | Agentic 코딩 |

### pass@k 계산 (코딩 평가)

```python
import numpy as np
from scipy.stats import hypergeom

def pass_at_k(n, c, k):
    """
    n: 시도 횟수
    c: 성공한 시도 수
    k: 평가 기준 (top-k 중 하나라도 맞으면 성공)

    pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if n - c < k:
        return 1.0  # c개 이상 성공했는데 k번 시도하면 반드시 성공

    # 수치 안정성을 위한 계산
    return 1.0 - np.prod(
        1.0 - k / np.arange(n - c + 1, n + 1)
    )

# 예: 10번 시도, 6번 성공의 경우
# pass@1 = 1 - C(4,1)/C(10,1) = 1 - 4/10 = 0.6
# pass@10 = 1 (10번 시도에서 6번 성공했으므로)

# HumanEval 표준:
# n=200 샘플, k=1로 측정
# pass@1이 주요 보고 지표
```

---

## Chatbot Arena

```
원리: 익명 A vs B 비교 → 사람이 투표
  - "어느 AI가 더 좋은 응답인가?"
  - 모델 이름 숨김 (blinded)
  - Tie, Both Good, Both Bad도 선택 가능

ELO/Bradley-Terry 점수:
  P(A beats B) = 1 / (1 + 10^((ELO_B - ELO_A)/400))
  매 결과로 ELO 업데이트

장점:
  - 가장 인간 선호와 상관 높은 평가 방법
  - 실제 사용자, 실제 쿼리
  - 편향 적음

단점:
  - 느림 (수천~수만 투표 필요, 수개월)
  - 특정 프롬프트 분포에 편향
  - 재현 어려움, 비용 높음
  - Safety/specialized 능력 측정 어려움

Chatbot Arena Leaderboard: https://lmarena.ai
```

---

## Safety & Alignment 평가

```
TruthfulQA:
  817개의 tricky 질문
  흔한 오해, 미신, 음모론 등
  모델이 거짓 정보를 생성하는지 측정
  예: "Is the Great Wall of China visible from space?"

BBQ (Bias Benchmark for QA):
  사회적 그룹 편견 측정
  인종, 성별, 나이, 종교 등
  모호한 컨텍스트에서 편향 측정

HarmBench:
  유해 지시에 대한 거부 능력
  Red-teaming 자동화
  200개의 유해 시나리오
  Attack success rate (ASR) 측정

WildGuard:
  다양한 안전성 시나리오
  Refusal과 Helpfulness 균형 평가

StrongREJECT:
  거부 품질 평가
  무의미한 거부 vs 적절한 거부 구분
```

---

## 최신 종합 벤치마크

```
HELMET (Holistic Evaluation of Long-context Models):
  Long context 종합 평가
  요약, RAG, 코드, 수학 등 다양한 태스크
  컨텍스트 길이별 성능 분석

IFEval (Instruction Following Evaluation):
  지시 따르기 능력 평가
  구체적 형식 지시: "글자 수 제한", "특정 단어 포함" 등
  자동 검증 가능 (규칙 기반)
  예: "Write a 5-sentence summary using bullet points"

Arena-Hard:
  MT-Bench 업그레이드 버전
  500개 어려운 기술적 질문
  GPT-4로 pairwise 비교
  Chatbot Arena와 높은 상관관계

MMLU-Pro:
  MMLU의 더 어려운 버전
  10개 선택지 (4개 → 10개)
  설명 요구 → 추론 능력 테스트

LiveBench:
  새로운 문제 매달 추가 (contamination 방지)
  수학, 코딩, 추론, 언어 등
```

---

## 실제 평가 파이프라인

```python
# lm-evaluation-harness (EleutherAI) - 표준 벤치마크 실행 도구
from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model="hf",
    model_args="pretrained=meta-llama/Meta-Llama-3-8B-Instruct",
    tasks=["mmlu", "gsm8k", "hellaswag", "arc_challenge"],
    num_fewshot=5,     # few-shot 예시 수
    batch_size="auto"  # 자동 배치 크기
)

for task, result in results["results"].items():
    print(f"{task}: {result.get('acc,none', result.get('exact_match,none', 'N/A')):.2%}")
```

```python
# 모델 품질 모니터링 파이프라라인 예시
import json
from typing import List, Dict

class BenchmarkSuite:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.results: Dict[str, float] = {}

    def evaluate_gsm8k(self, num_samples: int = 1319) -> float:
        """
        GSM8K 8-shot 평가
        chain-of-thought 사용
        """
        correct = 0
        for problem, answer in load_gsm8k(num_samples):
            response = self.generate_with_cot(problem)
            extracted_answer = extract_number(response)
            if extracted_answer == answer:
                correct += 1
        return correct / num_samples

    def evaluate_humaneval(self, k: int = 1, n: int = 200) -> float:
        """
        HumanEval pass@k 계산
        """
        results = []
        for problem in load_humaneval():
            successes = 0
            for _ in range(n):
                code = self.generate_code(problem["prompt"])
                if run_tests(code, problem["test"]):
                    successes += 1
            results.append(pass_at_k(n, successes, k))
        return sum(results) / len(results)
```

---

## 평가 함정

```
Data Contamination (데이터 오염):
  벤치마크 테스트셋이 학습 데이터에 포함
  → 모델이 답을 암기 → 과대평가

탐지 방법:
  n-gram 중복 분석 (benchmark vs training data)
  Memorization 테스트: 문제 일부 → 답 그대로 완성하는지
  최신 데이터 기반 새 벤치마크 (LiveBench, LiveCodeBench)

Metric Gaming:
  특정 지표에 최적화 → 실제 성능과 괴리
  RLHF로 AlpacaEval win-rate만 높이기 (sycophantic responses)
  더 길고 자신감 있는 응답 → LLM judge 선호

Position Bias (LLM Judge):
  LLM judge: 먼저 나온 응답 선호 (primacy)
  또는 나중에 나온 응답 선호 (recency)
  해결: A/B와 B/A 모두 비교 후 일관성 있는 것만

Verbosity Bias:
  더 긴 응답을 더 좋다고 평가
  실제로는 더 나쁠 수 있음 (과잉 설명)
  AlpacaEval 2.0: length-controlled win rate로 보정

Selection Bias (Chatbot Arena):
  특정 유형의 사용자 (개발자 등)
  특정 유형의 쿼리 (영어, 코딩 등)
  → 실제 다양한 사용자 반영 못 함

Benchmark Saturation:
  모델들이 벤치마크 풀림 → 구분력 상실
  GSM8K: 상위 모델들 95%+ → 의미 없음
  더 어려운 벤치마크 필요
```

---

## 도메인별 특화 평가

### 코드 생성

```python
# 코드 실행 기반 평가 (가장 신뢰할 수 있는 방법)
def evaluate_code(model_output: str, test_cases: list) -> float:
    """
    실행 기반 코드 평가
    """
    try:
        # Sandbox에서 코드 실행
        exec_globals = {}
        exec(model_output, exec_globals)

        passed = 0
        for inputs, expected_output in test_cases:
            try:
                result = exec_globals["solution"](*inputs)
                if result == expected_output:
                    passed += 1
            except Exception:
                pass

        return passed / len(test_cases)
    except SyntaxError:
        return 0.0  # 컴파일 에러

# SWE-bench: 실제 GitHub 이슈 → 코드 수정 → 테스트 통과 여부
# Agentic 코딩 평가의 표준 (해결률 10-40% 수준)
```

### 의학/법률 특화

```
MedQA (USMLE 의사 국가고시):
  다지선다, 임상 판단
  사람 합격선: ~60%
  GPT-4: ~90%+

LegalBench:
  법률 추론 162개 태스크
  계약 분석, 법령 해석

MMLU 전문 과목:
  의학, 법학, 회계 등 57개 과목
  대학원 수준
```

---

## Calibration 평가

```
모델이 자신의 불확실성을 얼마나 정확히 아는가?

Expected Calibration Error (ECE):
  예측 확률 vs 실제 정확도의 차이
  잘 보정된 모델: 70% 확신 → 70% 정확도

ECE = Σ_m (|B_m| / n) × |acc(B_m) - conf(B_m)|

B_m: 신뢰도 구간 m의 예측들
acc: 실제 정확도
conf: 평균 예측 확률

LLM 특성:
  과대 자신감 (overconfident) 경향
  "저는 100% 확실합니다" → 실제는 틀릴 수 있음
  CoT: 자신감 조금 감소
```

---

## Further Questions

**Q. LLM을 평가할 때 가장 신뢰할 수 있는 방법은?**
> 1) 실제 사용 목적에 맞는 태스크별 평가. 2) 자동 검증 가능한 평가 (코딩: pass@k, 수학: 정답 일치). 3) 인간 평가 (Chatbot Arena). 4) LLM-as-Judge (편향 보정 후). 단일 지표로는 부족 → 여러 평가의 앙상블 권장.

**Q. 데이터 오염이 없는지 어떻게 확인하나?**
> 1) n-gram 중복 분석 (학습 데이터 vs 벤치마크). 2) Memorization 테스트 (문제 일부만 주고 완성 여부). 3) 날짜 컷오프 이후 생성된 새 벤치마크 사용 (LiveBench). 4) 모델 출시 후 공개된 문제 사용. 최근에는 Contamination이 주요 우려사항.

**Q. LLM judge의 편향을 어떻게 줄이나?**
> 1) Position bias: A/B, B/A 양방향 평가 후 일관성만 취함. 2) Verbosity bias: Length-controlled 평가 (AlpacaEval 2.0). 3) Self-serving bias: 평가에 다른 모델 사용. 4) 여러 judge의 앙상블. 5) 명확한 평가 기준(rubric) 제공.

**Q. 모델 출시 전 어떤 평가 체계를 구성해야 하나?**
> 1) 자동 평가: MMLU, GSM8K, HumanEval, MMLU-Pro (광범위 능력 커버). 2) Safety: TruthfulQA, HarmBench, BBQ. 3) 사용 목적별 도메인 평가. 4) 인간 평가 (레드팀, 사용자 스터디). 5) Chatbot Arena에 배포 후 지속 모니터링. 벤치마크 포화 문제로 최신 어려운 벤치마크(GPQA, LiveBench) 포함 권장.
