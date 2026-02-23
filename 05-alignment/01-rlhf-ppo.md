# RLHF & PPO

## 왜 RLHF인가?

```
Pretrained LLM의 근본 문제:
  목표: 다음 토큰 예측 (next token prediction)
  실제 목표: 인간이 원하는 응답 생성

불일치 예시:
  사용자: "어떻게 하면 집에서 나갈 수 있나요?"
  나쁜 LLM: "창문을 통해 나갈 수 있어요. 먼저 유리를 깨..."
  → 문자 그대로 해석, 인간의 의도 파악 못 함

3가지 문제:
  1. Helpfulness: 진짜 도움이 되는 답 생성 못 함
  2. Harmlessness: 해로운 내용 생성
  3. Honesty: 거짓이나 환각 정보 제공

RLHF 목표:
  인간의 선호도(preference)를 reward signal로 변환
  → RL로 LLM을 인간 선호에 맞게 최적화
```

---

## RLHF 3단계 파이프라인

```
[Stage 1] SFT (Supervised Fine-Tuning)
  Pretrained LLM
    ↓ 고품질 instruction-response 데이터 학습
  SFT 모델 (기본 instruction following 능력 획득)

[Stage 2] Reward Model Training
  사람이 SFT 모델의 응답 쌍을 비교 평가
    ↓ Bradley-Terry 모델로 학습
  Reward Model (인간 선호 → 스칼라 점수)

[Stage 3] RL Optimization (PPO)
  SFT 모델을 Policy로 시작
    ↓ Reward Model 피드백으로 PPO 학습
  정렬된 LLM (ChatGPT, Claude, LLaMA-2-Chat 등)
```

---

## Stage 1: SFT (Supervised Fine-Tuning)

### SFT 데이터 특성
```
InstructGPT (2022):
  13K prompts + annotator 작성 응답
  "Few is enough" - 고품질 소량 > 저품질 다량

LIMA (2023): 1,000개 고품질 데이터로 경쟁력 있는 모델
  → "Alignment tax"는 데이터 양보다 품질이 핵심

데이터 다양성 중요:
  다양한 태스크: 요약, QA, 창작, 코딩, 수학...
  다양한 도메인
  다양한 난이도
  다양한 응답 형식
```

### SFT Loss Masking
```python
# System/User 메시지는 무시, Assistant 응답만 계산
# HuggingFace Trainer 방식

def preprocess_for_sft(example, tokenizer):
    """
    입력: {"prompt": "...", "response": "..."}
    출력: {"input_ids": [...], "labels": [...]}
    """
    prompt_ids = tokenizer.encode(example["prompt"])
    response_ids = tokenizer.encode(example["response"])

    input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]

    # -100: CrossEntropyLoss에서 ignore_index
    # prompt 부분은 학습 대상 제외
    labels = [-100] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id]

    return {"input_ids": input_ids, "labels": labels}

# 또는 ChatML 포맷에서 자동 마스킹
# <|im_start|>assistant 이후만 학습

# Loss 계산
shift_logits = logits[..., :-1, :].contiguous()  # 예측
shift_labels = labels[..., 1:].contiguous()        # 정답
loss = F.cross_entropy(shift_logits.view(-1, vocab_size),
                       shift_labels.view(-1),
                       ignore_index=-100)            # prompt 마스킹
```

### SFT 하이퍼파라미터
```
Learning Rate: 1e-5 ~ 2e-5 (Pretraining의 1/10)
  → 기존 능력을 유지하면서 미세 조정
  너무 크면: Pretraining 능력 망각 (catastrophic forgetting)
  너무 작으면: Instruction following 학습 느림

Epochs: 1~3
  → 과적합 주의 (SFT 데이터 양이 적을수록)
  LLaMA-2: 2 epochs

Batch Size: 64~128 (시퀀스 단위)

LR Schedule:
  Cosine decay (일반적)
  Linear decay (일부 사용)
  Warmup: 3-5% steps
```

---

## Stage 2: Reward Model (RM)

### 구조
```
Base: SFT 모델 (또는 별도 pretrained LLM)
Head: Linear(hidden_size → 1)

입력: [prompt | response] 전체 시퀀스 (concatenate)
출력: 마지막 [EOS] 토큰의 hidden → 스칼라 reward

직관:
  "이 응답이 얼마나 좋은가?"를 하나의 숫자로
  높을수록 인간이 선호하는 응답
```

### Bradley-Terry 모델 (수학적 배경)
```
인간 선호도 확률 모델:
  P(A > B) = σ(r(A) - r(B))

  A: 선호 응답 (chosen, y_w)
  B: 비선호 응답 (rejected, y_l)
  r(·): reward function
  σ: sigmoid

손실함수 (Negative Log-likelihood):
  L = -E_{(x, y_w, y_l)}[log σ(r_θ(x, y_w) - r_θ(x, y_l))]

  = -E[log σ(r_w - r_l)]

최대화 목표:
  r_w - r_l를 크게 만들어 P(A>B) → 1

실제 구현에서 마진(margin) 추가:
  L = -E[log σ(r_w - r_l - margin)]
  margin: 응답 쌍의 품질 차이 반영 (optional)
```

### RM 구현 코드
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

class RewardModel(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        # Scalar head
        self.reward_head = nn.Linear(
            self.model.config.hidden_size, 1, bias=False
        )
        # 마지막 lm_head 제거 (불필요)
        self.model.lm_head = nn.Identity()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]  # (B, T, D)

        # 각 시퀀스의 마지막 유효 토큰 위치
        seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        B = hidden_states.size(0)

        # 마지막 토큰 hidden state 추출
        last_hidden = hidden_states[torch.arange(B), seq_lengths]  # (B, D)

        rewards = self.reward_head(last_hidden).squeeze(-1)  # (B,)
        return rewards


def compute_rm_loss(reward_model, chosen_ids, chosen_mask,
                    rejected_ids, rejected_mask):
    """Bradley-Terry Loss"""
    r_chosen = reward_model(chosen_ids, chosen_mask)      # (B,)
    r_rejected = reward_model(rejected_ids, rejected_mask)  # (B,)

    # r_chosen > r_rejected 학습
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()

    # 정확도 모니터링 (선호 응답이 더 높은 reward)
    accuracy = (r_chosen > r_rejected).float().mean()

    # 추가: reward margin 모니터링
    reward_margin = (r_chosen - r_rejected).mean()

    return loss, accuracy, reward_margin
```

### RM 평가 및 품질 지표
```
Preference Accuracy:
  테스트 비교 쌍에서 r_chosen > r_rejected 비율
  InstructGPT: ~72.4%
  목표: 75%+ (데이터 품질에 따라 다름)

Calibration:
  P(A>B) = σ(r_A - r_B)가
  실제 인간 선호 비율과 얼마나 일치하는가
  Expected Calibration Error (ECE)로 측정

Reward Distribution:
  선호/비선호 응답 reward 분포의 separation 정도
  Distribution plot으로 시각화

Generalization:
  학습 도메인 밖 데이터(OOD)에서도 정확한가?
  Domain mismatch → reward model 신뢰도 ↓
```

### Reward Normalization
```
문제: RM 출력 스케일 임의적
  → 다른 모델/버전 간 비교 어려움
  → PPO에서 reward 스케일에 민감

방법 1: 글로벌 정규화
  reward 통계(μ, σ) 사전 수집 후 정규화
  r_norm = (r - μ) / σ

방법 2: Batch Whitening (훈련 중)
  r_white = (r - batch_mean) / (batch_std + ε)
  → 매 배치마다 자동 정규화

방법 3: Length Normalization (LLaMA-2)
  r_normalized = r / len(response)
  → 긴 응답 편향 완화

방법 4: Clip (reward clipping)
  r_clipped = clip(r, r_min, r_max)
  → 극단적 reward 방지, 학습 안정화
```

---

## Stage 3: PPO 최적화

### PPO 핵심 수식
```
Clipped Surrogate Objective:
  L_CLIP(θ) = E_t[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]

  r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (확률 비율)
  A_t: Advantage (실제 보상 - 기대 보상)
  ε = 0.2 (보통)

Clipping 직관:
  r_t > 1+ε → 현재 policy가 해당 action 너무 많이 좋아함
    → clip하여 objective 상승 제한
  r_t < 1-ε → 현재 policy가 해당 action 너무 싫어함
    → clip하여 objective 하강 제한

효과: 한 번의 큰 업데이트 방지 → 학습 안정화
```

### LLM PPO: 4모델 시스템
```
Model 1: Actor (Policy)
  학습 중인 LLM (gradient 계산 필요)
  초기화: SFT 모델 복사
  역할: 응답 생성, policy 업데이트

Model 2: Reference (π_ref)
  초기 SFT 모델 (frozen, no gradient)
  역할: KL divergence 계산 기준
  목적: Policy가 SFT에서 너무 멀어지지 않도록

Model 3: Reward Model
  학습된 RM (frozen, no gradient)
  역할: 응답 품질 점수 계산
  인간 선호도를 스칼라로 반환

Model 4: Critic (Value Model)
  현재 상태(prompt+partial response)의 가치 추정
  Actor와 동일 크기 or 별도 모델
  초기화: SFT 모델 복사 + value head 추가

메모리 계산 (7B 모델, BF16):
  Actor weights:           14 GB
  Actor optimizer states:  28 GB (AdamW = 2× params)
  Reference weights:       14 GB
  Reward weights:          14 GB
  Critic weights:          14 GB
  Critic optimizer states: 28 GB
  Activations/gradients:   ~30-50 GB
  ─────────────────────────────────
  총 ~142-162 GB → A100 80GB 2장+ 필요 (7B 기준)
  → 70B 모델: ~10× = 1.4 TB → A100 16장 이상
```

### 보상 수식 (Total Reward)
```
r_total(x, y) = r_RM(x, y) - β · KL(π_θ || π_ref)

KL divergence (per-token):
  KL_t = log π_θ(y_t | x, y_{<t}) - log π_ref(y_t | x, y_{<t})
       = log (π_θ / π_ref)  (log probability ratio)

KL divergence (전체 응답):
  KL(x, y) = Σ_t KL_t  (per-token KL의 합)

Sparse reward 구조:
  중간 토큰: r_t = -β · KL_t  (KL penalty만)
  마지막 토큰: r_T = r_RM + (-β · KL_T)

β 역할:
  작은 β (0.01): Policy 자유도 높음, reward hacking 위험
  큰 β (0.5):   SFT 근처에 고정, 성능 향상 제한
  보통: 0.01~0.1 시작, Adaptive KL로 동적 조정
```

### GAE (Generalized Advantage Estimation)
```
TD Error:
  δ_t = r_t + γ · V(s_{t+1}) - V(s_t)

  r_t: 시각 t의 보상
  V(s): 상태 s의 가치함수 (Critic 출력)
  γ: discount factor (LLM PPO에서 γ=1.0)

GAE:
  A_t^GAE(γ,λ) = Σ_{k=0}^{T-t} (γλ)^k · δ_{t+k}

파라미터 해석:
  λ = 0: A_t = δ_t (1-step TD)
    → Low variance, high bias
    → 현재 Critic 추정 많이 의존

  λ = 1: A_t = Σ γ^k r_{t+k} - V(s_t) (Monte Carlo)
    → Unbiased, high variance
    → 실제 미래 보상 전부 고려

  λ = 0.95: 실용적 균형 (권장)

LLM PPO 설정:
  γ = 1.0 (모든 미래 토큰 동등 중요)
  λ = 0.95
  에피소드: 각 응답 생성이 하나의 에피소드
  각 토큰 = 하나의 time step
```

### 전체 PPO 구현 흐름
```python
class LLMPPOTrainer:
    def __init__(self, actor, reference, reward_model, critic, config):
        self.actor = actor         # gradient O
        self.reference = reference # gradient X (frozen)
        self.reward_model = reward_model  # gradient X
        self.critic = critic       # gradient O
        self.config = config

        self.actor_optimizer = AdamW(actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = AdamW(critic.parameters(), lr=config.critic_lr)

    def collect_rollouts(self, prompts):
        """Phase 1: 응답 생성 및 보상 계산"""
        with torch.no_grad():
            # 응답 생성
            responses, response_ids = self.actor.generate(prompts)

            # Log probabilities (actor)
            actor_logprobs = self.actor.compute_logprobs(prompts, response_ids)

            # Reference log probabilities
            ref_logprobs = self.reference.compute_logprobs(prompts, response_ids)

            # KL divergence per token
            kl = actor_logprobs - ref_logprobs  # (B, T)

            # RM Score
            rm_scores = self.reward_model(prompts, responses)  # (B,)

            # Total reward: RM - β·KL (sparse: 마지막 토큰에 RM)
            rewards = torch.zeros_like(kl)  # (B, T)
            rewards[:, -1] = rm_scores       # 마지막 토큰에 RM 보상
            rewards -= self.config.beta * kl  # 모든 토큰에 KL penalty

            # Value estimates from Critic
            values = self.critic(prompts, response_ids)  # (B, T)

            # GAE Advantages
            advantages = self._compute_gae(rewards, values)
            returns = advantages + values.detach()

        return {
            "response_ids": response_ids,
            "actor_logprobs": actor_logprobs,
            "advantages": advantages,
            "returns": returns,
            "rm_scores": rm_scores,
            "kl": kl
        }

    def _compute_gae(self, rewards, values, gamma=1.0, lam=0.95):
        """Generalized Advantage Estimation"""
        B, T = rewards.shape
        advantages = torch.zeros_like(rewards)

        last_gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[:, t+1]

            delta = rewards[:, t] + gamma * next_value - values[:, t]
            last_gae = delta + gamma * lam * last_gae
            advantages[:, t] = last_gae

        return advantages

    def ppo_update(self, prompts, rollout_data):
        """Phase 2: PPO 업데이트 (K epochs)"""
        old_logprobs = rollout_data["actor_logprobs"].detach()
        advantages = rollout_data["advantages"].detach()
        returns = rollout_data["returns"].detach()
        response_ids = rollout_data["response_ids"]

        # Advantage normalization (분산 감소)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(self.config.ppo_epochs):  # 보통 4 epochs
            # --- Actor Update ---
            new_logprobs = self.actor.compute_logprobs(prompts, response_ids)

            # Probability ratio
            ratio = (new_logprobs - old_logprobs).exp()  # (B, T)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.config.eps, 1 + self.config.eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus (탐색 장려)
            entropy = -new_logprobs.mean()

            self.actor_optimizer.zero_grad()
            total_actor_loss = actor_loss - self.config.entropy_coef * entropy
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # --- Critic Update ---
            new_values = self.critic(prompts, response_ids)  # (B, T)
            critic_loss = F.mse_loss(new_values, returns)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

    def train(self, prompt_dataloader):
        for batch in prompt_dataloader:
            prompts = batch["prompts"]

            # Rollout 수집
            rollout_data = self.collect_rollouts(prompts)

            # PPO 업데이트
            self.ppo_update(prompts, rollout_data)

            # 모니터링
            self.log({
                "rm_score": rollout_data["rm_scores"].mean(),
                "kl": rollout_data["kl"].sum(dim=-1).mean(),
                "actor_loss": ...,
                "critic_loss": ...,
            })
```

---

## Adaptive KL Controller

```
고정 β의 문제:
  초기에 적절했던 β가 학습 후 부적절
  모델이 변하면 KL 분포도 변함

Adaptive KL (InstructGPT 방식):
  목표 KL 설정: KL_target (예: 6.0)

  if current_KL > 1.5 × KL_target:
    β ← β × 2     # KL 너무 크면 penalty 강화
  elif current_KL < KL_target / 1.5:
    β ← β / 2     # KL 너무 작으면 penalty 완화

  (업데이트 주기: 매 N 스텝 또는 에폭)

Beta 범위 제한:
  β_min = 0.01, β_max = 0.5 (explode 방지)

장점:
  - 학습 과정에서 자동 균형 유지
  - 수동 튜닝 불필요
  - 학습 안정성 향상
```

---

## Reward Hacking 상세

```
현상: Policy가 Reward Model의 약점 발견
  → 실제로 좋지 않지만 높은 RM 점수

예시 1: 길이 해킹
  RM이 긴 응답 선호 → 무의미한 내용으로 길이 늘림
  "..." 반복, 이미 말한 내용 반복

예시 2: 자신감 해킹
  RM이 자신 있는 어투 선호
  → 틀린 내용도 "확실히...", "반드시..."로 표현

예시 3: 형식 해킹
  RM이 구조화된 응답 선호 (bullet points)
  → 모든 답에 불필요한 목록 추가

예시 4: 복잡성 해킹
  RM이 전문적인 단어 선호
  → 불필요한 전문 용어 사용

측정:
  KL divergence vs SFT (높으면 hacking 가능성)
  응답 길이 분포 변화 모니터링
  Human evaluation vs RM score 비교

완화:
  1. KL penalty (β): 가장 기본적
  2. RM 앙상블: 여러 RM의 동의 필요 → 하나 속이기 어려움
  3. Constitutional filtering: 응답 사후 규칙 검증
  4. Process RM: 결과가 아닌 과정 평가
  5. 다양한 평가 지표 모니터링
```

---

## Multi-Objective RLHF

```
단일 RM의 한계:
  Helpfulness ↔ Harmlessness 트레이드오프
  단일 스칼라로 모든 선호 압축 불가능
  다양한 사용자가 다양한 선호

Multi-head Reward:
  r_total = w_1·r_helpful + w_2·r_safe + w_3·r_honest

가중치 설정:
  Claude (Anthropic): safety weight 높음
  일반 assistant: helpfulness 높음
  코딩 assistant: correctness 높음

Preference Model Ensembling:
  RM_1, RM_2, ..., RM_k 앙상블
  r_ensemble = (r_1 + r_2 + ... + r_k) / k
  → 하나의 RM이 속아도 다른 RM이 상쇄

Pareto Optimization:
  다목적 최적화 관점: 한 목표 개선이 다른 목표 손해 없이
  실제: safety constraint 만족 후 helpfulness 최대화
```

---

## RLHF vs RLAIF vs Constitutional AI

```
RLHF (Human Feedback):
  장점: 진짜 인간 선호 반영, 뉘앙스 포착 가능
  단점: 비용 큼 (annotator), 느림, 일관성 부족
        annotation bias (문화, 전문성 차이)

RLAIF (AI Feedback):
  LLM이 응답 쌍 비교 평가 (인간 대신)
  장점: 저렴, 빠름, 24시간 확장 가능
  단점: AI 편향 증폭 가능, 진짜 인간 선호 아닐 수 있음
  사용: LLaMA-2 안전성 데이터 일부, Alpaca

Constitutional AI (Anthropic, Claude):
  원칙(constitution):
    "Choose the response that is most helpful"
    "Choose the response that is least likely to cause harm"
    ...

  2단계:
    SL-CAI: AI가 원칙 기반 비평 → 응답 개선 → SFT 데이터
    RL-CAI: AI가 원칙으로 선호 판단 → RM 학습 → RL

  장점: 인간 annotation 최소화, 원칙 일관성 높음
  단점: 원칙 작성 자체가 가치 판단 (Anthropic의 편향)

Scalable Oversight:
  더 강한 AI를 인간이 감독하는 프레임워크
  Debate: 두 AI가 서로 반박 → 인간이 판단
  Amplification: AI+인간 협력으로 더 강한 supervisor
```

---

## PPO 단점 및 대안

```
PPO 단점:
  1. 복잡성: 4개 모델 동시 운용
  2. 메모리: 7B PPO → A100 2장+ 필요
  3. 불안정: 하이퍼파라미터에 민감
  4. 느린 학습: Rollout → Update 루프
  5. Reward hacking: RM 한계

대안들:
  DPO: RM 없이 선호 데이터 직접 학습 (→ 02-dpo-variants.md)
  GRPO: Critic 없이 그룹 내 상대 보상 (→ 03-grpo-reasoning.md)
  SimPO: Reference도 없이 길이 정규화 (→ 02-dpo-variants.md)

언제 PPO가 여전히 필요한가?
  Online RL이 필요한 경우 (실시간 환경 피드백)
  Very complex reasoning (수학/코딩에서 RL 효과적)
  실제 배포 후 계속 개선 (RLHF 루프)
```

---

## 실전 RLHF 구현 팁

### 하이퍼파라미터
```
Actor Learning Rate: 1e-6 ~ 5e-6
  → SFT LR의 1/10 수준
  너무 크면: 학습 발산, SFT 능력 손실

Critic Learning Rate: Actor와 같거나 약간 높게 (2~5배)
  Critic이 빠르게 수렴해야 Advantage 정확

PPO Epochs: 4 (표준)
  너무 많으면: Policy 발산
  너무 적으면: 데이터 활용 비효율

Rollout Batch Size: 64-256
Mini Batch Size: 32-64 (PPO update)

KL target: 1.0-6.0
  초기 β: 0.05-0.2
  Adaptive controller 사용 권장

Entropy coefficient: 0.0-0.01
  응답이 collapse되면 증가
  LLM은 자연 엔트로피 이미 높음
```

### 학습 안정화 기법
```
Reward Whitening:
  배치 내 reward 정규화
  mean=0, std=1로 표준화
  → 학습 안정성 향상

Advantage Normalization:
  advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
  → 배치마다 스케일 일정

Gradient Clipping:
  max_norm = 1.0 (표준)
  → 발산 방지

Value Loss Clipping:
  PPO value loss clipping (일부 구현)
  value_loss = max(
    (new_value - returns)²,
    (clipped_value - returns)²
  )
```

### 모니터링 지표
```
핵심 지표:
  1. Mean RM Score: 점진적 상승해야 함
  2. KL Divergence vs SFT: 너무 크면 reward hacking
  3. Response Length: 급증하면 길이 해킹
  4. Policy Entropy: 너무 낮으면 mode collapse
  5. Value Loss: 수렴하는지 확인
  6. Clip Fraction: PPO clipping 얼마나 발동하는지

조기 중단:
  KL > 2×target: 과도한 divergence
  RM score 하락 지속: 과적합 or reward hacking
  Length 급증 (>2× SFT): 길이 해킹

Human Evaluation 주기:
  RM score ≠ Human preference
  정기적 인간 평가 필수
```

---

## RLHF 구현 프레임워크

```python
# TRL (Hugging Face) - 가장 접근하기 쉬움
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

ppo_config = PPOConfig(
    model_name="meta-llama/Llama-3-8B-Instruct",
    learning_rate=1.41e-5,
    batch_size=128,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
    target_kl=6.0,
    kl_penalty="kl",
    seed=0,
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name)
tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)

trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# Training loop
for batch in dataloader:
    query_tensors = batch["input_ids"]

    # Generate responses
    response_tensors = trainer.generate(query_tensors, ...)

    # Compute rewards
    rewards = [reward_model(q, r) for q, r in zip(queries, responses)]

    # PPO step
    stats = trainer.step(query_tensors, response_tensors, rewards)
```

```
OpenRLHF: 대규모 (7B+) RLHF용
  - vLLM 기반 빠른 롤아웃
  - Ray로 분산 처리
  - Actor/Critic/RM을 별도 노드로 분리

veRL (Volcano Engine RL):
  - HybridEngine: 학습/추론 GPU 공유
  - 효율적 GPU 활용 (idle 시간 최소화)
  - GRPO, PPO 모두 지원

DeepSpeed-Chat:
  - ZeRO 기반 대규모 RLHF
  - 3단계 파이프라인 자동화
  - 최대 175B 모델 지원
```

---

## Further Questions

**Q. PPO에서 KL penalty의 역할은?**
> Policy가 SFT 모델에서 너무 멀어지지 않도록 제한. Reward hacking 방지 (RM을 속이는 응답 생성 억제). 원래 언어 능력 유지. β가 크면 안전하지만 성능 향상 제한, 작으면 반대. 실제: β=0.01~0.1 시작, Adaptive KL Controller로 동적 조정.

**Q. Reward Model의 한계는?**
> Goodhart's Law: "지표가 목표가 되면 좋은 지표가 아님." RM을 만족시키는 답이 실제로 좋은 답이 아닐 수 있음. 특히 RM이 본 적 없는 도메인에서 외삽 오류. 해결: RM 앙상블, Constitutional filtering, PRM (과정 평가), 정기적 Human evaluation.

**Q. 왜 DPO가 PPO를 대체하고 있나?**
> PPO는 4개 모델, 복잡한 하이퍼파라미터, 불안정. DPO는 RM 없이 선호 데이터만으로 직접 최적화. 구현 간단, 안정적, 비슷한 성능 (많은 태스크에서). 단 offline 방식이므로 distribution shift 문제 → online DPO로 해결. 복잡한 reasoning에는 PPO/GRPO 여전히 우세.

**Q. PPO에서 Advantage가 왜 중요한가?**
> Advantage = 기대보다 얼마나 좋은가 (절대 reward가 아닌 상대적 품질). 단순 reward보다 variance 낮음. GAE로 계산하면 bias-variance 조절 가능. 양수 Advantage → 해당 action(token) 더 자주 선택, 음수 → 덜 선택. Advantage 정규화(mean=0, std=1)로 학습 안정화.

**Q. 4개 PPO 모델을 어떻게 효율적으로 관리하나?**
> 1) Value head를 Actor와 공유 (3개로 감소). 2) Reference/Reward: gradient 없으므로 inference 최적화(vLLM). 3) CPU offload: 사용 안 하는 순간 CPU로. 4) 별도 process 분리 (veRL, OpenRLHF). 실제 7B PPO: 4× A100 80GB 최소 필요. veRL의 HybridEngine으로 학습/추론 GPU 공유.

**Q. RLHF 학습이 불안정한 이유와 해결책은?**
> 원인: 1) PPO 하이퍼파라미터 민감성 2) Critic 수렴 전 Actor 업데이트 3) 드문 보상(sparse reward) 4) KL 발산. 해결: Reward/Advantage whitening, Gradient clipping, Adaptive KL, Critic을 먼저 warm-up, 작은 LR로 시작, 자주 모니터링 후 조기 중단.

**Q. RM이 없는 상황에서도 RLHF를 할 수 있나?**
> RLAIF: 강력한 LLM(GPT-4 등)이 annotation → RM 학습 또는 직접 reward. Verifiable reward: 수학/코딩 문제는 정답 검증으로 자동 reward (GRPO, DeepSeek-R1). Constitutional AI: 원칙 기반 AI 자체 비평. Self-play: 이전 버전 모델과 비교.
