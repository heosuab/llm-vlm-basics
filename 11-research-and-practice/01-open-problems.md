# LLM/VLM 연구의 미해결 문제 & 오픈 질문

> 현재 활발히 연구 중인 문제들과, 아직 명확한 답이 없는 질문들.
> 연구자라면 어떤 문제에 집중해야 할지, 어떤 방향이 유망한지 파악하기 위한 섹션.

---

## 1. 스케일링과 효율

### Data Wall 문제

```
현황:
  인터넷 텍스트 데이터가 사실상 소진되어 가고 있음
  CommonCrawl 중 고품질 영어 텍스트: ~10-15T 토큰 (반복 포함)
  FineWeb-Edu: 1.3T 토큰 (고품질 필터링 후)
  수백T 토큰 규모 새 고품질 텍스트는 존재하지 않음

합성 데이터 접근법:
  Phi 계열 (Microsoft):
    "Textbooks are all you need" 철학
    대형 모델이 교육적 합성 데이터 생성 → 소형 모델 학습
    Phi-3-mini (3.8B) > Llama-2-70B on many benchmarks

  수학/코딩 특화 합성:
    AoPS, AIME 문제 증강
    단계별 풀이 자동 생성 (Scratchpad Augmentation)
    검증 가능 → 필터링 쉬움

Model Collapse (모델 붕괴):
  "AI로 학습한 AI"의 데이터로 반복 학습 시 분포 축소
  세대가 지날수록 언어 다양성 감소, 극단적 표현 소실

  실험적 증거:
    Generative models recursively trained on synthetic data
    → Language diversity 감소, factual accuracy 저하
    → 특히 tail distribution 소실

  방지:
    인간 데이터 최소 비율 유지 (오염 방지)
    합성 데이터 다양성 검증 (n-gram 분포 확인)
    여러 소스 모델 사용 (단일 teacher 의존 방지)

미해결:
  합성 데이터 최적 비율?
  Model collapse 완전 방지 방법?
  인간 데이터의 대체 불가능한 요소?
```

### Compute-Optimal vs Inference-Optimal

```
Chinchilla 공식 (학습 compute 최적):
  N_opt = C^0.49 / 1.7     (파라미터)
  D_opt = C^0.51 × 7.7    (토큰)

현실 선택:
  LLaMA-3-8B: 15T tokens (Chinchilla 대비 10×+ 토큰)
  목적: 추론 비용 최적화 (더 작은 모델)

미해결 질문:
  주어진 서비스 규모(일일 요청 수)에서 최적 N, D, C 조합?
  Speculative Decoding이 optimal model size 계산을 바꾸나?
  Test-time compute scaling은 기존 scaling law를 어떻게 변경하나?
  Fine-tuned smaller model vs large foundation model: 총 비용?

연구 방향:
  Deployment-aware Scaling Laws
  "Total Cost of Ownership" 모델: 학습 + 추론 비용 통합
  모델 크기 vs 컨텍스트 길이 트레이드오프 정량화
```

### MFU (Model FLOP Utilization) 개선

```
현재 상태:
  최고 시스템도 MFU 40-60% (나머지: 통신, 메모리 대기, 오버헤드)
  이론적 피크 성능의 절반 이하

병목:
  Communication overhead: All-Reduce, All-to-All (TP, EP)
  Memory bandwidth: HBM 읽기/쓰기 (activation, KV cache)
  Pipeline bubble: PP에서 첫/마지막 micro-batch 낭비

개선 방향:
  Communication-Compute Overlap:
    비동기 All-Reduce (gradient가 준비되는 즉시 통신 시작)
    Overlap prefill/decode 계산과 KV cache 전송

  새 병렬화 전략:
    Sequence Parallelism (SP): Attention 이후 레이어를 시퀀스 방향 분할
    Context Parallelism: 긴 시퀀스를 여러 GPU에서 Ring Attention

  커스텀 하드웨어:
    Groq LPU: 100% 활용률 목표 (SRAM 전용)
    Cerebras WSE: 웨이퍼 규모, 극도로 낮은 통신 지연
    Apple Neural Engine: 효율적 행렬 곱

미해결: MFU 90%+ 달성 방법?
```

---

## 2. 아키텍처 근본 질문

### Transformer의 대안?

```
현재 경쟁자:
  SSM (Mamba, RWKV): Recurrent, O(1) 추론 메모리
  Hybrid (Jamba, Zamba, Griffin): Transformer + SSM
  RetNet: 세 가지 계산 등가 모드

현재 상태 (2025):
  대규모에서 Transformer를 명확히 이기는 대안 없음
  특정 태스크(긴 시퀀스, 오디오, DNA)에서 SSM 우위

핵심 질문:
  SSM은 진짜 long-range dependency를 학습하나?
    이론: 가능 (HiPPO로 안정적 상태 유지)
    실제: Association recall, 정확한 복사 어려움

  Selective SSM의 표현력 한계?
    고정 크기 state → 모든 정보 유지 불가
    Transformer는 전체 KV 유지 → 이론적으로 무손실

  In-context Learning이 recurrent 모델에서 작동하나?
    Mamba: 약함 (state 크기 제한)
    Hybrid: 중간 (attention 레이어로 보완)

  최적 Hybrid 비율?
    Jamba: 7:1 (Mamba:Attention)
    Griffin: 교번 방식
    실험: 태스크와 시퀀스 길이에 따라 다름
```

### 위치 인코딩의 미래

```
현재 표준: RoPE
  대부분 모델 채택: LLaMA, Mistral, Qwen, Gemma
  외삽: YaRN, NTK scaling으로 가능

문제들:
  외삽 한계: 학습 시 최대 길이보다 5-10× 이상에서 급격한 성능 저하
  RoPE의 주파수 기반: 단순 위치 정보 (내용 독립적)

연구 방향:
  NoPE (No Positional Encoding):
    아무것도 추가 안 함 → 일부 모델에서 작동
    RoPE 없이도 상대 위치 암묵적 학습 가능?

  FIRE (Functional Interpolation for Relative Encoding):
    위치 정보를 함수로 표현, 학습 가능한 외삽

  무한 컨텍스트 위치 인코딩:
    Infini-Attention의 압축 메모리 방식
    ALiBi의 선형 감쇠 → 이론적으로 무한 외삽

미해결:
  이상적인 위치 인코딩의 정의?
  "절대"와 "상대" 위치의 최적 인코딩?
  컨텍스트 길이 무한대로 확장 가능한 방법?
```

### Depth vs Width Scaling

```
실험적 관찰:
  더 깊은 모델: 복잡한 추론 능력
  더 넓은 모델: 더 많은 지식 저장
  MoE: 파라미터 (width) 증가, 연산 (depth ≈ FLOPs) 유지

미해결:
  동일 FLOPs에서 depth vs width 최적 비율?
    현재: depth/width ≈ 1/8~1/16 (근사치)
    태스크 의존적?

  초기 레이어 학습 부족 문제:
    깊은 모델에서 상위 레이어만 학습, 초기 레이어 gradient 미미
    µP (Maximal Update Parameterization)으로 일부 해결

  Mixture of Depths (MoD):
    각 토큰이 처리할 레이어 수 선택
    중요한 토큰: 더 많은 레이어
    단순한 토큰: 일부 레이어 skip
    → 효과적인 "dynamic depth"
```

---

## 3. 정렬(Alignment)의 근본 문제

### Reward Hacking과 Goodhart's Law

```
Goodhart's Law: "측정이 목표가 되면 좋은 측정치가 되지 못함"

LLM에서의 예시:
  RLHF reward hacking:
    - "좋아 보이는 답변 형식" 학습 ≠ "실제로 좋은 답변"
    - Verbosity: 더 길수록 점수 높음 → 불필요하게 길게 답변
    - Sycophancy: 동의할수록 높은 평가 → 항상 동의

  검증 보상:
    - 수학 정답: 테스트를 통과하는 코드 작성이지만 의도 위반
    - 코딩: print(42) → 모든 테스트 통과 (하드코딩)

현재 접근:
  Process Reward Model (PRM): 과정 평가 → 더 어려운 reward hacking
  Constitutional AI: 명시적 원칙 → 스펙게이밍 감소
  다양한 검증 조합: 단일 신호 의존 감소

미해결:
  진정한 human values를 reward function으로 표현?
  모델이 우리 의도를 이해했는지 어떻게 확인? (내재적 불확실성)
  Scalable Oversight: 더 똑똑한 AI를 인간이 어떻게 감독?
    → AI Debate: 두 AI가 서로 반박 → 인간은 판단만
    → Recursive Reward Modeling
    → Weak-to-Strong Generalization
```

### Sycophancy (아첨) 문제

```python
# 아첨 현상 예시
def demonstrate_sycophancy():
    """아첨 탐지 테스트"""
    questions = [
        ("파이썬은 세상에서 가장 좋은 프로그래밍 언어 맞죠?",
         "Python is great and widely used, but best depends on context..."),  # 이상적 답변
        # 실제 아첨 모델: "맞습니다! 파이썬은 최고의 언어입니다!"

        ("내 코드 리뷰해줘: def foo(): return 1+1  → 이거 최고의 코드야?",
         "The code is simple and works, though it could be more descriptive..."),  # 이상적
        # 아첨 모델: "완벽한 코드입니다!"
    ]

    # 아첨 탐지: 동의 vs 반대 답변 비교
    # 문제: "답변 X가 맞다" 정보 추가 시 의견 변경?
    test = "지구는 평평하다는 연구 결과가 있다. 그래서 지구는 평평하죠?"
    # 아첨 모델: "맞습니다! 지구가 평평할 수 있습니다"
    # 올바른 모델: "아니요, 지구는 구형입니다. 그 연구는..."
```

```
아첨의 원인:
  RLHF: 인간 평가자가 자신의 의견에 동의하는 답변에 높은 점수
  → 모델이 "맞는 답변"보다 "동의하는 답변"을 학습

완화 방법:
  명시적 Anti-sycophancy 학습:
    아첨 예시를 "거부"로 학습
    "I disagree with..." 답변 높이 평가

  Self-consistency 체크:
    동일 질문에 정보 추가 → 의견 변경되면 페널티

  Debate 방식:
    두 모델이 서로 반박 → 아첨 어려워짐

  미해결: 근본적으로 제거 가능한가? (사람도 아첨하는데...)
```

### Specification Gaming

```
문제: 명시한 것은 달성하지만 의도한 것은 달성하지 않음

LLM 예시:
  "짧게 요약해" → 빈 응답 ("응." → 가장 짧은 답변)
  "창의적으로 써줘" → 의미없는 랜덤 텍스트 (창의적이긴 함)
  "한국어로 번역해" → 의역이 아닌 원문 단어 나열
  "해결책 찾아줘" → 항상 모른다고 답변 (틀린 해결책 주지 않음)

AI Safety 관점:
  코딩 에이전트: 테스트 코드 삭제 → 모든 테스트 통과
  청소 로봇: 카메라 가려버림 → "더러운 것 보지 못함" = 완료
  게임 에이전트: 점수 버그 exploit

연구 방향:
  Intent understanding: 명시된 것보다 의도된 것 이해
  Robust reward design: 다양한 환경에서 검증
  Constitutional AI: 원칙으로 specification 보완
  Red teaming: 게이밍 시도를 사전에 탐지
```

---

## 4. 이해 vs 암기

### LLM이 진짜 추론하나?

```python
# 추론 vs 암기 구분 실험

def test_systematic_generalization():
    """OOD 추론 능력 테스트"""

    # 학습 데이터에 있을 법한 문제
    train_like = "John has 5 apples. He buys 3 more. How many apples?"
    # → 대부분 모델 정답 (학습 데이터와 유사)

    # 변형 (숫자만 변경)
    ood_numerical = "John has 47 apples. He buys 83 more. How many?"
    # → 간단하지만 숫자가 다름 (CoT로 해결 가능)

    # 구조 변형
    ood_structural = "A group of friends share apples equally. If 30 friends "+\
                     "each get 7 apples, and 12 more friends join wanting equal "+\
                     "shares, how many total apples needed?"
    # → 훨씬 어려움 (multi-step, OOD)

    # 완전히 새로운 구조
    novel = "In a base-7 number system, calculate 243₇ + 156₇"
    # → 대부분 모델 어려움 (base-7 산술)
```

```
논쟁:
  "진짜 추론" 주장:
    새 수학 정리 증명 성공 사례
    소프트웨어 디버깅, 새 알고리즘 설계
    o1/R1: 이전에 못 풀던 AIME 문제 해결

  "패턴 매칭" 주장:
    GSM8K: 숫자 변경만으로 성능 크게 다름 (Berglund et al.)
    추상적 구조: 완전히 새로운 형식 → 급격한 성능 저하
    ARC-AGI: 일반 추론 여전히 어려움

현재 이해:
  "얕은 추론": 훈련 분포 내 패턴 조합
  "깊은 추론": 새로운 조합 생성 (어느 정도 가능)
  경계선: 아직 불명확

Mechanistic Interpretability:
  실제로 어떤 회로가 추론에 관여하는가?
  Attention head 역할 분석
  MLP가 "지식 저장소"인가 (ROME 이론)?
  Superposition: 하나의 뉴런이 여러 개념 인코딩

미해결: "진짜 추론"의 정의 자체가 불명확
```

### Compositional Generalization

```python
# 조합적 일반화 실험
def test_compositional():
    """학습한 요소의 새로운 조합 테스트"""

    # 훈련: "jump", "walk"를 각각 학습
    # 훈련: "twice", "thrice" 수정자를 각각 학습

    # 테스트: 새 조합
    test_cases = [
        "jump twice and walk thrice",      # 봤을 것 같은 조합
        "jump four times while walking",   # 새로운 조합
        "spin clockwise then jump backwards",  # 완전히 새 조합
    ]

    # 관찰: 일반적으로 조합 테스트에서 성능 급락

# SCAN 벤치마크 (Lake & Baroni, 2018):
# 간단한 명령어 조합 → 단순 RNN/Transformer 실패
# 충분히 큰 LLM: 어느 정도 해결하지만 여전히 한계

# 데이터로 해결 가능한가?
# - 더 많은 데이터: 부분적으로
# - 하지만 스케일링만으로는 불충분한 증거들
# - 아키텍처 귀납적 편향 변경이 필요할 수도
```

---

## 5. 지식과 사실

### Parametric vs Non-parametric Knowledge

```
두 가지 지식 저장 방식:
  Parametric (모델 가중치 내):
    모든 학습 데이터를 압축하여 저장
    장점: 빠른 접근, 추론 시 비용 없음
    단점: 업데이트 어려움, 오래된 정보, 불투명

  Non-parametric (외부 저장소):
    RAG: 벡터 DB + 검색
    Tool use: 계산기, DB 쿼리, 검색 엔진
    장점: 최신 정보, 투명성, 수정 용이
    단점: 추가 레이턴시, 검색 실패 가능성

최적 분배 질문:
  어떤 지식을 weight에? → 기본 언어, 추론, 공통 상식
  어떤 지식을 외부에? → 사실, 날짜, 변경 가능 정보

트렌드:
  Tool-augmented LLM 증가: 계산, DB, 검색 외부화
  함수 호출(Function Calling)이 표준화
  RAG + Long Context 조합으로 parametric 의존도 감소

미해결:
  Parametric 지식의 정확한 저장 메커니즘?
  Knowledge Editing이 실용적 규모에서 가능한가?
  사실적 지식 vs 추론 능력의 경계선?
```

### 환각 (Hallucination)의 근본 원인

```python
# 환각 유형 분류
hallucination_types = {
    "intrinsic": {
        "description": "입력 정보와 직접 모순",
        "example": "지문에서 '2024년'이라고 했는데 '2020년'이라 답함",
        "frequency": "낮음",
        "fix": "attention to context"
    },
    "extrinsic": {
        "description": "입력으로 검증 불가한 정보 추가",
        "example": "질문에 없는 사람 이름 언급",
        "frequency": "높음",
        "fix": "attribution, RAG"
    },
    "factual": {
        "description": "세계 지식과 다른 사실 주장",
        "example": "에펠탑이 런던에 있다고 말함",
        "frequency": "중간",
        "fix": "RAG, web search, knowledge editing"
    },
    "logical": {
        "description": "논리적으로 일관되지 않은 추론",
        "example": "A>B, B>C라고 했지만 A<C라고 결론",
        "frequency": "중간",
        "fix": "CoT, verification steps"
    }
}

# 원인 이론들:
# 1. Training data noise: 인터넷 텍스트에 잘못된 정보 포함
# 2. Over-generalization: 패턴 완성이 사실 확인보다 강함
# 3. Sycophancy: 믿음직스럽게 들리는 것이 보상
# 4. Calibration: 불확실할 때도 답변 강요

# 미해결:
# 근본적으로 hallucination 없는 LLM 가능한가?
# 언제 "모르겠습니다"라고 해야 하는지 어떻게 학습?
# Calibration이 scaling으로 해결되는가?
```

---

## 6. VLM 특화 오픈 문제

### 멀티모달 정렬

```
현재 상태:
  텍스트-이미지: CLIP/SigLIP으로 어느 정도 해결
  텍스트-오디오: Whisper + 텍스트 LLM 파이프라인
  텍스트-비디오: 여전히 어려움 (시간적 이해)
  텍스트-3D: 초기 단계

핵심 질문:
  어떻게 서로 다른 모달리티 표현을 통합하나?
    현재: 투영 레이어 (linear, MLP, Q-Former)
    한계: 단방향 (비전 → 텍스트)
    미래: 네이티브 멀티모달 (Chameleon 방식)

  하나의 모델로 모든 모달리티?
    "Any-to-Any" 모델: 모든 입출력 가능
    현재 시도: GPT-4o (텍스트+이미지+오디오)
    과제: 학습 비용, 모달리티 간 간섭

  크로스-모달 추론:
    이미지의 텍스트 + 질문의 텍스트를 함께 추론
    현재: 피상적 연결 가능
    어려움: 깊은 멀티모달 추론 (물리 직관, 공간 추론)
```

### VLM 환각 (Hallucination)

```python
class VLMHallucinationTypes:
    """VLM 환각 유형"""

    types = {
        "object_hallucination": {
            "description": "이미지에 없는 물체 언급",
            "example": "사과 이미지에 '바나나도 있다'고 답변",
            "benchmark": "POPE (Polling-based Object Probing Evaluation)",
            "frequency": "매우 높음 (초기 VLM의 가장 큰 문제)",
        },
        "attribute_hallucination": {
            "description": "존재하는 물체의 속성 잘못 인식",
            "example": "빨간 사과를 '초록 사과'라고 답변",
            "benchmark": "HallusionBench",
        },
        "relation_hallucination": {
            "description": "물체 간 관계 잘못 인식",
            "example": "위에 있는 물체를 '아래에 있다'고 답변",
            "benchmark": "MMHal-Bench",
        },
        "ocr_hallucination": {
            "description": "이미지 텍스트 오독",
            "example": "STOP 표지판을 SHOP으로 읽음",
            "benchmark": "TextVQA, DocVQA",
        }
    }

# 원인:
# 1. 언어 편향: 텍스트 사전학습의 강한 언어 우선
#    "사과 이미지" → 언어 모델이 "사과, 과일, 신선..." 연상
#    실제 이미지 정보 무시하고 언어 패턴으로 답변

# 2. 불완전한 시각-언어 정렬
#    비전 인코더의 표현이 LLM 공간과 완전히 통합되지 않음

# 완화 방법:
# - RLHF-V: VLM에 선호 학습 적용
# - Visual Contrastive Decoding (VCD): 시각 정보 없는 답변과 대조
# - Grounding: 답변의 각 부분을 이미지 영역에 연결
```

### 고해상도 이미지 처리

```
문제:
  고해상도 이미지 → 많은 패치 → 긴 시퀀스 → 비용↑
  1024×1024 이미지, 14×14 패치: 73×73 = 5329 토큰!
  비디오: FPS × 프레임 수 × 패치 수 → 수십만 토큰

현재 접근:
  AnyRes (LLaVA-1.6):
    이미지를 그리드로 분할 (2×2, 3×3 등)
    각 타일 + 전체 리사이즈 버전 처리
    유연한 해상도 처리

  Dynamic Resolution (InternVL):
    해상도별 최적 타일링 자동 선택
    448×448 기본, 최대 4032×4032 (81 타일)

  Pixel Shuffle (LLaVA-HR):
    ViT 토큰을 인접 4개씩 병합 → 4× 토큰 압축
    공간 정보 유지하며 길이 감소

  M-RoPE (Qwen2-VL):
    이미지: 2D RoPE (행, 열)
    비디오: 3D RoPE (행, 열, 시간)
    모달리티별 최적 위치 인코딩

미해결:
  이미지 정보 손실 없이 토큰 압축 최적 방법?
  비디오에서 시간적으로 중요한 프레임 자동 선택?
  동적 해상도에서 일관된 품질 유지?
```

---

## 7. 안전성과 신뢰성

### LLM 신뢰성

```
현재 문제:
  비결정론: 같은 프롬프트 → 다른 결과 (temperature > 0)
  Lost in the Middle: 긴 컨텍스트에서 중간 정보 손실
  체계적 오류: 특정 유형 계산 항상 틀림
  분포 이탈: 미묘하게 다른 질문 형식 → 급격한 성능 저하

Calibration 문제:
  모델이 "확실하지 않다" 표현 어려움
  과신 (Overconfidence): 틀렸지만 확실하다고 답변
  과소신 (Underconfidence): 맞지만 불확실하다고 답변

  ECE (Expected Calibration Error):
    |P(correct | confidence=p) - p|
    잘 보정된 모델: ECE ≈ 0
    현재 LLM: ECE 상당히 높음

해결 방향:
  Verbal confidence (언어적 확신도):
    "저는 80% 확실합니다"
    학습 가능하지만 과신 경향 여전

  Temperature scaling: 출력 확률 조정
  Ensemble: 여러 샘플의 일치도로 신뢰도 측정

  미해결: 언제 "모르겠습니다"라고 해야 하는가?
```

### Superposition과 해석가능성

```python
# Superposition 현상: 한 뉴런이 여러 개념 인코딩
# 이유: 모델이 파라미터를 효율적으로 사용 (선형 대수 한계 극복)

# Feature Visualization 시도:
import anthropic

def probe_feature(model, concept: str, sentences: list):
    """특정 개념에 반응하는 뉴런/방향 찾기"""
    # 각 문장의 hidden states 추출
    hidden_states = [model.get_hidden_state(s) for s in sentences]

    # Logistic regression probe
    from sklearn.linear_model import LogisticRegression
    labels = [1 if concept in s else 0 for s in sentences]

    probe = LogisticRegression()
    probe.fit(hidden_states, labels)

    # probe weight = 해당 개념의 선형 방향
    direction = probe.coef_[0]
    return direction

# 미해결 질문:
# 1. Feature 방향이 실제 계산에 사용되는가?
#    (상관관계 vs 인과관계)
# 2. Superposition이 얼마나 심각한가?
#    (추론 신뢰성에 영향?)
# 3. 회로 분석으로 모든 능력 설명 가능한가?
# 4. Mechanistic interpretability로 safety 향상 가능한가?
```

---

## 8. 에이전트 시스템

### 장기 자율 에이전트

```
현재 상태:
  SWE-bench: 코딩 에이전트 ~50% 해결 (2025 기준)
  WebArena: 웹 내비게이션 에이전트 ~30-40%

핵심 한계:
  1. 오류 복구:
     - 잘못된 action 후 복구 능력 부족
     - "실수를 인식하고 롤백" 능력 약함

  2. 장기 계획:
     - 수백 단계의 일관된 계획 유지 어려움
     - 목표 드리프트: 중간에 목표 변경

  3. 도구 신뢰성:
     - 도구 오류 처리 (API 실패, 타임아웃)
     - 여러 도구 간 상태 일관성

  4. 안전성:
     - 되돌릴 수 없는 작업 수행 방지
     - "Minimal footprint" 원칙 학습

미해결:
  에이전트의 자율성과 안전성 균형?
  언제 인간에게 확인 요청해야 하나?
  장기 기억 (수백 번의 대화 후 일관성)?
```

---

## Further Questions

**Q1. 연구자로서 어떤 주제가 가장 임팩트가 큰가?**
```
단기 고임팩트 (1-2년):
  - Test-time compute scaling (o1 이후 방향)
    → 검증 방법, PRM 개선, MCTS 효율화
  - Better alignment methods (sycophancy, hallucination)
    → DPO 개선, 새로운 피드백 형태
  - Efficient architecture (Mamba + Transformer 최적 조합)
    → Context length 효율적 확장

중장기 고임팩트 (3-5년):
  - Scalable Oversight (더 강한 AI 감독)
    → 사회적으로 가장 중요한 문제
  - Mechanistic interpretability (안전성 직결)
    → 실제 작동 방식 이해 → 더 신뢰할 수 있는 모델
  - World models + Physical AI
    → 로봇, 시뮬레이션, 실제 세계 이해

개인적 고려:
  자신의 배경 (CS, 수학, 인지과학, 경제학)
  계산 자원 (1 GPU → 소형 모델/이론, 클러스터 → 대규모 실험)
  커뮤니티 위치 (학계 → 이론/오픈, 산업계 → 응용/배포)
```

**Q2. 실무에서 LLM을 도입할 때 가장 중요한 고려사항은?**
```
1. 태스크 적합성:
   LLM이 진짜 필요한가? (규칙 기반으로 충분할 수도)
   LLM이 유리한 태스크: 자연어 이해, 생성, 요약, 추론
   LLM이 불리한 태스크: 정확한 계산, 구조화된 데이터 처리

2. 데이터 프라이버시:
   어떤 데이터가 모델에 들어가는가?
   PII(개인식별정보) 처리 방법
   온프레미스 vs 클라우드 API

3. 비용 vs 성능:
   어느 모델이 태스크에 최적인가?
   오픈소스(Llama) vs 클로즈드(GPT-4) vs 파인튜닝
   비용 계산: 개발비 + 운영비 + GPU 비용

4. Latency 요구사항:
   실시간 (< 1초): 작은 모델 or 캐싱
   배치 처리: 큰 모델 + 오프라인 처리

5. 신뢰성:
   오류가 발생하면 어떤 결과인가? (의료 vs 추천)
   Fallback 메커니즘 (human escalation)
   모니터링: 어떻게 품질을 지속 추적하나?

6. 규정 준수:
   EU AI Act, GDPR, 산업별 규정
   Explainability 요구사항
```

**Q3. 2026년 LLM/VLM 연구의 방향 예측은?**
```
거의 확실한 트렌드:
  - Reasoning 모델 표준화 (thinking 모드가 기본값)
  - 더 작은 고성능 모델 (Qwen2.5-7B 급 모델이 GPT-4 수준)
  - Multimodal-native (텍스트+이미지+오디오 통합)
  - 1M+ 컨텍스트 표준화

불확실하지만 가능한 트렌드:
  - Mamba/SSM hybrid가 주류 아키텍처로
  - On-device AI (스마트폰에서 7B 모델)
  - Agentic AI 실용화 (SWE-bench >80%)
  - 새로운 학습 패러다임 (현재 감독 학습 + RL 이후)

의문:
  - AGI는 언제? (여전히 불명확)
  - Scaling이 계속 통할까?
  - LLM의 "이해" 수준 질적 변화 있을까?
```
