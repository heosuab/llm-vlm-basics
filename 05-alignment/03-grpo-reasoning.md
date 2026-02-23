# GRPO & Reasoning 특화 RL

## GRPO (Group Relative Policy Optimization) [DeepSeek, 2024]

### 배경: PPO의 문제

```
PPO는 Critic(가치 함수) 모델 필요:
  - Policy와 같은 크기 → 메모리 2배
  - Critic 학습도 불안정
  - 전체 파이프라인 복잡
  - 학습 속도 느림

GRPO의 핵심 인사이트:
  Critic을 같은 질문에 대한 여러 응답의 상대적 비교로 대체
```

### GRPO 수학적 정의

```
같은 질문 x에 G개 응답 샘플링:
  {y₁, y₂, ..., y_G} ~ π_θ_old(·|x)

각 응답의 보상 계산:
  {r₁, r₂, ..., r_G} = {r(x, y₁), ..., r(x, y_G)}

Advantage 계산 (그룹 내 상대적 품질):
  Â_i = (r_i - mean({r_j})) / (std({r_j}) + ε)

GRPO Loss:
  L_GRPO(θ) = -E[1/G · Σᵢ₌₁ᴳ min(
    (π_θ(yᵢ|x) / π_θ_old(yᵢ|x)) · Âᵢ,
    clip(ratio, 1-ε, 1+ε) · Âᵢ
  )] + β · KL(π_θ || π_ref)

  ε (clip): 보통 0.1~0.2
  β (KL penalty): 보통 0.001~0.01
  G (group size): 보통 4~16
```

### GRPO 구현 (PyTorch 스타일)

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class GRPOTrainer:
    def __init__(self, model, ref_model, tokenizer,
                 group_size=8, beta=0.001, clip_eps=0.2):
        self.model = model
        self.ref_model = ref_model  # 고정된 레퍼런스 모델
        self.tokenizer = tokenizer
        self.G = group_size
        self.beta = beta
        self.clip_eps = clip_eps

    def compute_log_probs(self, model, input_ids, response_ids):
        """전체 응답의 log probability 계산"""
        input_len = input_ids.shape[1]
        full_ids = torch.cat([input_ids, response_ids], dim=1)

        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            logits = model(full_ids).logits

        # 응답 부분만 선택
        shift_logits = logits[:, input_len-1:-1, :]
        shift_labels = full_ids[:, input_len:]

        log_probs = -F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='none'
        ).reshape(shift_labels.shape)

        return log_probs.sum(dim=-1)  # 시퀀스 전체 log prob

    def generate_group_responses(self, input_ids):
        """같은 질문에 G개 응답 생성"""
        # 입력을 G번 복제
        repeated_input = input_ids.repeat(self.G, 1)

        responses = self.model.generate(
            repeated_input,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.9,  # 다양성을 위해 온도 높임
        )
        return responses[:, input_ids.shape[1]:]  # 응답 부분만

    def compute_grpo_loss(self, input_ids, rewards):
        """
        input_ids: [G, seq_len] - 같은 프롬프트 G개
        rewards: [G] - 각 응답의 보상
        """
        # 보상 정규화 (그룹 내 상대적 품질)
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r

        # 각 응답에 대한 log prob 계산
        responses = self.generate_group_responses(input_ids[:1])

        log_probs_new = self.compute_log_probs(
            self.model, input_ids[:1], responses
        )
        with torch.no_grad():
            log_probs_old = self.compute_log_probs(
                self.ref_model, input_ids[:1], responses
            )

        # Policy ratio
        ratios = torch.exp(log_probs_new - log_probs_old)

        # Clipped objective
        clipped_ratios = torch.clamp(
            ratios, 1 - self.clip_eps, 1 + self.clip_eps
        )
        policy_loss = -torch.min(
            ratios * advantages,
            clipped_ratios * advantages
        ).mean()

        # KL divergence regularization
        kl_div = (log_probs_old - log_probs_new).mean()
        total_loss = policy_loss + self.beta * kl_div

        return total_loss
```

### GRPO vs PPO 비교

```
특성            PPO                 GRPO
────────────────────────────────────────────
Critic 모델     필요 (1개 추가)    불필요
메모리          4× model size      2× model size
Advantage 계산  GAE (TD lambda)    그룹 내 상대적 비교
구현 복잡도     높음               낮음
안정성          중간               높음 (보상이 명확할 때)
적합한 태스크   다양              수학/코딩 등 검증 가능
```

---

## RLOO (REINFORCE Leave-One-Out) [Ahmadian et al., 2024]

```
GRPO와 유사하나 Leave-One-Out baseline:

  baseline_i = (Σⱼ≠ᵢ rⱼ) / (G-1)  (자신 제외 나머지 평균)
  Advantage_i = r_i - baseline_i

차이점:
  GRPO: 전체 평균을 baseline으로 사용
  RLOO: 자신을 제외한 나머지 평균 → 분산 감소

수학적으로:
  RLOO는 control variate로 분산을 더 효과적으로 감소
  N이 클수록 GRPO와 수렴

실용적:
  구현이 약간 더 복잡
  보통 GRPO와 비슷한 성능
```

---

## DeepSeek-R1 학습 과정

### DeepSeek-R1-Zero (순수 RL)

```
SFT 없이 직접 RL로 reasoning 능력 학습:

1. 시작: DeepSeek-V3-Base (pretrained, SFT 없음)
2. 보상 함수:
   a) 정확성 보상:
      - 수학: 최종 답이 정답 DB와 일치 → r = 1
      - 코딩: 테스트 케이스 통과 → r = pass_rate
      - 형식 조건 없으면 r = 0
   b) 형식 보상:
      - <think>...</think> 태그 사용 → r = 0.5~1
      - 태그 없으면 패널티
3. GRPO로 학습 (G=8, 수천 gradient steps)

결과:
  - Chain-of-Thought reasoning 자발적 등장
  - "Aha moment": 풀다가 막히면 스스로 재고
    "Wait, let me reconsider..." 형식의 자기 반성
  - 문제: 가독성 낮음, 영어/중국어 혼용
  - 기본 지시 따르기 능력 제한
```

### DeepSeek-R1 (Cold Start → RL → 재증류)

```
Step 1: Cold Start SFT
  소량(수천 개)의 고품질 CoT 데이터로 SFT
  데이터 형식:
    <think>
    [단계별 추론 과정 작성...]
    </think>
    [최종 답변]
  목적: 기본 추론 형식 학습, 가독성 개선

Step 2: Reasoning-Oriented RL (GRPO)
  수학, 코딩 등 검증 가능한 태스크
  보상 함수:
    - 정확성: 수학 정답 여부, 코드 테스트 통과
    - 형식: think 태그 사용 여부
  수천~수만 step 학습

Step 3: Rejection Sampling + SFT
  RL 모델에서 N개 샘플링 → 정답인 것만 선택
  → 고품질 응답 필터링
  선별된 응답으로 추가 SFT
  일반 태스크 데이터도 추가 (글쓰기, QA 등)

Step 4: RL for All Tasks (GRPO + DPO)
  수학/코딩: GRPO (검증 가능 보상)
  일반 태스크: DPO (인간/AI 선호도)
  안전성: 안전하지 않은 응답 필터링

결과: OpenAI o1 수준 추론 능력, 오픈소스
```

### R1 증류 모델

```
R1의 고품질 응답 → 소형 모델 SFT 데이터로 활용

R1-Distill-Qwen-1.5B: 작은 추론 모델
R1-Distill-Qwen-7B: 실용적 추론
R1-Distill-LLaMA-8B: LLaMA 기반
R1-Distill-Qwen-14B/32B: 강력한 추론
R1-Distill-LLaMA-70B: 대형 추론 모델

핵심: 큰 모델의 thinking 데이터로 작은 모델 학습
→ 직접 RL 없이 reasoning 능력 전이
```

---

## 검증 가능한 보상 (Verifiable Rewards)

```
RLHF의 RM 대신 자동 검증 가능한 보상:

수학:
  - 최종 답 추출: \boxed{...} 또는 = ... 파싱
  - 정답 DB와 비교: r = 1 (정답), r = 0 (오답)
  - LaTeX 정규화: "1/2" = "0.5" = "\frac{1}{2}"
  도구: sympy로 수식 등가성 검증

코딩:
  - 제출 코드를 sandbox에서 실행
  - 테스트 케이스별 통과 여부
  - r = (통과한 케이스 수) / (전체 케이스 수)
  - 컴파일 에러: r = 0

논리/퀴즈:
  - MCQ: 정답 선택지와 일치
  - Open-ended: 키워드 포함 여부 (rough)

장점:
  - RM 학습 불필요 → 파이프라인 단순
  - 정확한 바이너리/연속 신호
  - Reward hacking 어려움 (실제 정답 필요)
  - 재현 가능, 확장 가능

한계:
  - 검증 가능한 태스크에만 적용
  - 오픈엔드 태스크 (에세이, 창작) 어려움
  - 엣지 케이스: 형식은 맞지만 다른 정답
```

---

## Process Reward Model (PRM) vs Outcome Reward Model (ORM)

```
ORM (Outcome RM):
  최종 답의 정답 여부만 평가
  단순, 구현 쉬움
  중간 추론 과정 무시

  문제:
    운좋게 맞은 답 (틀린 추론 → 우연히 정답) 구분 불가
    RL에서 sparse reward (마지막에만 신호)
    긴 추론 체인에서 어느 단계가 틀렸는지 모름

PRM (Process RM):
  각 추론 단계의 품질 평가
  잘못된 추론 단계 조기 탐지 가능

  PRM800K (OpenAI):
    수학 문제 추론 단계별 인간 레이블 데이터셋
    각 단계: 좋음/나쁨/중립

  Math-Shepherd (Peng et al., 2024):
    자동으로 PRM 훈련 데이터 생성
    각 단계에서 Monte Carlo rollout:
      "이 단계까지 왔을 때 끝까지 풀면 정답인가?"
      MC rollout으로 K번 시도 → 정답률로 레이블

  활용:
    Best-of-N: PRM 점수로 최고 응답 선택
    MCTS: 단계별 평가로 탐색 효율화
    RL: Dense reward (매 단계 신호)

비교:
  실용성: ORM이 구현 훨씬 쉬움
  성능: PRM이 복잡한 추론에서 더 좋음
  데이터: PRM 학습 데이터 수집 어려움 (단계별 레이블)
```

---

## Constitutional AI (Anthropic)

```
목표: AI가 스스로 원칙에 따라 자기 비평 → 개선

단계 1: SL-CAI (Supervised Learning CAI)
  1) AI가 유해할 수 있는 질문에 응답
  2) 헌법(principles) 기반으로 자기 비평:
     "이 응답이 [원칙]에 위배되는가?"
  3) 비평을 반영해 응답 개선
  4) 개선된 응답으로 SFT → Helpful & Harmless 모델

헌법 원칙 예시:
  "Choose the response that is most helpful, harmless, and honest"
  "Choose the response that is least likely to contain harmful content"
  "Choose the response that respects human rights and dignity"

단계 2: RL-CAI (RL with AI feedback)
  1) SL-CAI 모델로 대화 쌍 생성
  2) AI (Feedback Model)가 헌법에 따라 선호도 평가
  3) 인간 annotation 없이 선호도 데이터 생성
  4) RM 학습 → PPO 학습

장점:
  - 인간 annotation 최소화 (확장 가능)
  - 명시적 원칙으로 일관된 가치 기준
  - 원칙 수정만으로 모델 동작 조정 가능

Claude 모델들이 이 방식 사용
```

---

## Rejection Sampling Fine-Tuning (RSF / RFT)

```
Self-improvement 기법:

알고리즘:
  1. 현재 모델로 각 문제에 K개 응답 생성
  2. 정답 확인 (검증 가능한 보상)으로 좋은 응답 선택
  3. 선택된 (문제, 응답) 쌍으로 SFT 추가 학습
  4. 더 좋은 모델 → 더 많은 정답 → 반복

LLaMA-2에서 사용:
  대규모 Rejection Sampling → 고품질 SFT 데이터 생성
  PPO 없이도 강력한 alignment

STaR (Self-Taught Reasoner, Zelikman et al., 2022):
  - 틀린 문제에서도 정답 힌트 제공
  - 모델이 "왜 이 정답이 나왔는가" retrospective rationale 생성
  - 성공한 rationale로 SFT → iterative 개선

Iterative RFT / Self-Play:
  1. 현재 모델로 N개 생성
  2. RM으로 최고 선택
  3. SFT 학습
  4. 더 강력한 모델 → 반복
  → 모델이 자기 자신보다 나아지는 방향으로 학습
```

---

## 최신 Reasoning 모델 비교

| 모델 | 방법 | MATH-500 | AIME 2024 | 오픈소스 |
|------|------|---------|-----------|---------|
| OpenAI o1 | 내부 RL | ~96% | 79.2% | ✗ |
| OpenAI o3 | 더 강력 | ~99%+ | 91.6% | ✗ |
| DeepSeek-R1 | GRPO | 97.3% | 79.8% | ✓ |
| QwQ-32B | RL | 95.0% | 79.5% | ✓ |
| Claude 3.7 (Extended) | 내부 | ~90%+ | - | ✗ |
| Kimi k1.5 | 내부 RL | - | 77.5% | ✗ |

---

## Test-Time Scaling 방법

```
"더 많은 추론 시간 = 더 좋은 결과"

1. Best-of-N (가장 단순):
   N개 생성 → PRM/ORM으로 최고 선택
   N=1 대비 20-30% 향상 (N=100)

2. Sequential Revision:
   1) 초기 답 생성
   2) 스스로 검토 ("이 답이 맞는가?")
   3) 오류 발견 시 수정
   4) 반복 (K회)

3. MCTS (Monte Carlo Tree Search):
   State: 현재까지의 추론 텍스트
   Action: 다음 추론 단계 생성 (여러 후보)
   Reward: PRM(단계 평가) 또는 ORM(최종 답)
   Selection: UCB 공식으로 탐색/활용 균형

   UCB = Q(s,a) + c · √(ln N(s) / N(s,a))

   → Tree를 확장하면서 최적 추론 경로 탐색
   → 가장 강력하지만 가장 비쌈

4. Beam Search:
   여러 추론 경로 병렬 탐색
   각 단계에서 상위 K개 유지
   NLP 전통 방법의 LLM 적용

Scaling:
   수학/코딩: Compute ↑ → Performance ↑ (검증 가능)
   Open-ended: 효과 제한적
```

---

## Further Questions

**Q. GRPO가 PPO보다 효율적인 이유는?**
> Critic(Value) 모델이 없어 메모리/계산 절반. Group 내 상대적 보상으로 Advantage 계산 (정규화). 특히 수학/코딩처럼 정답이 명확한 태스크에서 보상 신호가 안정적이어서 Critic의 역할이 줄어듦. DeepSeek에서 입증: GRPO로 o1 수준 reasoning 달성.

**Q. DeepSeek-R1이 추론 능력을 얻은 방법은?**
> 핵심: 검증 가능한 수학/코딩 문제로 GRPO RL 학습. R1-Zero: 순수 RL만으로 CoT reasoning 자발적 등장. R1: Cold Start SFT → GRPO RL → Rejection Sampling + SFT → GRPO+DPO 4단계. Aha moment (자기 반성)는 RL 과정에서 자연스럽게 등장한 창발적 행동.

**Q. PRM vs ORM 언제 쓰나?**
> ORM: 구현 간단, 검증 가능한 태스크에 충분. PRM: 복잡한 다단계 추론에서 더 나은 성능, 중간 단계 오류 탐지. 데이터 수집 어려워 자동화 필요 (Math-Shepherd의 MC rollout). 실용적으로는 ORM + 좋은 Best-of-N으로 대부분 해결 가능.

**Q. Constitutional AI와 일반 RLHF의 차이는?**
> RLHF: 인간이 모든 선호도 레이블 제공 → 비용 높음. Constitutional AI: AI가 헌법(명시적 원칙)을 기반으로 자기 비평 → 인간 annotation 최소화. 원칙을 수정하면 쉽게 모델 동작 변경 가능. 하지만 AI의 판단이 인간 판단을 완전히 대체 못할 수 있음.
