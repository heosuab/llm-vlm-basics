# DPO 및 변형 알고리즘

## DPO (Direct Preference Optimization) [Rafailov et al., 2023]

### 핵심 아이디어
```
PPO의 복잡성 제거:
  Reward Model 없이 직접 선호 데이터로 최적화

수학적 통찰 (PPO → DPO 변환):
  RLHF 목표: max_π E[r(y)] - β·KL(π||π_ref)
    → 해석: reward 최대화 + SFT에서 너무 멀어지지 않음

  최적 정책 π*:
    π*(y|x) ∝ π_ref(y|x) · exp(r*(y,x)/β)

  → 역으로 reward를 policy로 표현:
    r*(y|x) = β log(π*(y|x) / π_ref(y|x)) + β log Z(x)

  RM 대신 π를 직접 학습:
    선호 쌍 (y_w, y_l)에서 r*(y_w) > r*(y_l) 조건 활용
    → RM 없이 Policy로만 표현 가능!

DPO Loss (Bradley-Terry 기반):
  L_DPO(θ) = -E_{(x,y_w,y_l)}[log σ(
    β log(π_θ(y_w|x) / π_ref(y_w|x))
    - β log(π_θ(y_l|x) / π_ref(y_l|x))
  )]

  = -E[log σ(β · (log_ratio_w - log_ratio_l))]

직관:
  선호 응답의 log-ratio(π_θ/π_ref) 높이고
  비선호 응답의 log-ratio(π_θ/π_ref) 낮추기
  단, 절대값이 아닌 참조 모델 대비 상대적 비율
```

### DPO 구현 코드
```python
import torch
import torch.nn.functional as F

def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,    # (B,) log P_θ(y_w|x)
    policy_rejected_logps: torch.Tensor,  # (B,) log P_θ(y_l|x)
    ref_chosen_logps: torch.Tensor,       # (B,) log P_ref(y_w|x)
    ref_rejected_logps: torch.Tensor,     # (B,) log P_ref(y_l|x)
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # log (π_θ / π_ref) = log π_θ - log π_ref
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps

    # DPO loss: -log σ(β · (log_ratio_w - log_ratio_l))
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()

    # 모니터링 지표
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()
    reward_accuracies = (chosen_rewards > rejected_rewards).float()

    return loss, chosen_rewards.mean(), rejected_rewards.mean()

def get_batch_logps(model, input_ids, labels, attention_mask):
    """배치의 총 log probability 계산 (SFT loss 방식)"""
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits  # (B, T, V)

    # Shift: labels는 -100으로 마스킹된 부분 무시
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Per-token log probability
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(
        log_probs, 2, shift_labels.unsqueeze(2).clamp(0)
    ).squeeze(2)

    # Masked sum (response 토큰만)
    mask = shift_labels != -100  # (B, T-1)
    return (token_log_probs * mask).sum(-1)  # (B,)
```

### 장점
```
1. 단순: RM 불필요, 2개 모델만 (policy + reference)
2. 안정: PPO보다 학습 안정적 (RL 루프 없음)
3. 빠름: 롤아웃 생성 불필요 (offline)
4. 비슷한 성능: 많은 instruction-following 태스크에서 PPO와 유사
5. 저비용: 7B DPO < 7B PPO의 1/4 GPU 요구
```

### 한계
```
1. Offline 방식:
   정적 선호 데이터셋에 의존
   Policy 변화에 따른 데이터 분포 갱신 없음
   → Distribution shift → 성능 저하 가능

2. Rejected 품질 의존성:
   나쁜 rejected 응답이 없으면 학습 신호 약함
   → Hard negatives 중요

3. 길이 편향 (Length Bias):
   DPO가 긴 응답을 선호하는 경향
   → SimPO의 length normalization으로 해결

4. Likelihood Displacement:
   이론상 선호 응답의 절대 확률이 낮아질 수 있음
   → Reference 대비 상대적으로 높아지는 것만 보장
   → RPO로 해결 (직접 NLL 추가)

5. Reference Model 필요:
   추가 메모리 및 계산 비용
   → SimPO, ORPO로 해결
```

---

## IPO (Identity Preference Optimization) [Azar et al., 2024]

```
DPO의 수렴 문제 해결:
  DPO 가정: 선호 데이터가 완벽하게 correct
  → 실제로는 noisy, probabilistic

IPO Loss:
  L_IPO = E[(
    log(π_θ(y_w|x)/π_ref(y_w|x))
    - log(π_θ(y_l|x)/π_ref(y_l|x)) - 1/β
  )²]

핵심:
  1/β 항 추가 → 목표 margin 명시
  DPO: log_ratio_w - log_ratio_l을 무한대로 최대화
  IPO: log_ratio_w - log_ratio_l = 1/β 목표

효과:
  Overfitting 방지
  데이터 노이즈에 더 강건
```

---

## KTO (Kahneman-Tversky Optimization) [Ethayarajh et al., 2024]

```
핵심 혁신: 쌍 비교 데이터 없이 단독 레이블만 필요

입력:
  (x, y, label)  where label ∈ {good, bad}
  쌍 비교 (x, y_w, y_l) 불필요!

전망 이론 (Prospect Theory) 기반:
  사람은 이득보다 손실에 더 민감 (비대칭)
  같은 크기의 이득 +x < 손실 -x의 심리적 영향

KTO Loss:
  z_0 = KL(π_θ(y|x) || π_ref(y|x))

  if y is desirable:
    L_KTO += max(0, v(x,y) - z_0)  # 기대보다 좋으면 reward
  if y is undesirable:
    L_KTO += max(0, z_0 - v(x,y))  # 기대보다 나쁘면 penalty

  v(x,y) = β · log(π_θ(y|x) / π_ref(y|x))

장점:
  데이터 수집 비용 절감 (쌍 비교 불필요)
  기존 단독 레이블 데이터 활용 가능
  대규모 데이터에서 DPO와 비슷하거나 좋음

단점:
  쌍 비교만큼 효율적이지 않을 수 있음
  z_0 (KL) 추정 필요
```

---

## SimPO (Simple Preference Optimization) [Meng et al., 2024]

```
핵심 혁신: Reference 모델 없이 동작

DPO와의 차이:
  DPO: π_θ(y|x) / π_ref(y|x)  (Reference 비율)
  SimPO: 평균 로그 확률 (Length-normalized)

SimPO Loss:
  L_SimPO = -E[log σ(
    β/|y_w| · log π_θ(y_w|x)
    - β/|y_l| · log π_θ(y_l|x)
    - γ
  )]

  |y|: 응답 길이 (토큰 수)
  γ: reward margin (최소 품질 차이 보장)

길이 정규화 효과:
  DPO: 긴 응답이 유리 (더 많은 토큰 = 더 높은 log prob)
  SimPO: 토큰당 평균 확률 비교 → 공정한 비교

γ (margin) 효과:
  선호 응답이 비선호보다 γ만큼 더 높은 reward 목표
  품질 차이가 명확하지 않은 데이터 처리

장점:
  Reference 모델 없음 → 메모리 절약 (~50%)
  길이 편향 감소
  DPO보다 AlpacaEval에서 우수 (7B, 8B 모델)

단점:
  margin γ 튜닝 필요
  Reference 없어 SFT 능력 유지 어려울 수 있음
```

```python
def compute_simpo_loss(
    policy_chosen_logps: torch.Tensor,   # (B,) total log probs
    policy_rejected_logps: torch.Tensor,
    chosen_lengths: torch.Tensor,         # (B,) token counts
    rejected_lengths: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 0.5,
) -> torch.Tensor:
    """SimPO Loss (no reference model needed)"""

    # Length-normalized log probabilities
    chosen_avg_logps = policy_chosen_logps / chosen_lengths
    rejected_avg_logps = policy_rejected_logps / rejected_lengths

    # SimPO logits with margin
    logits = beta * (chosen_avg_logps - rejected_avg_logps) - gamma

    loss = -F.logsigmoid(logits).mean()
    return loss
```

---

## ORPO (Odds Ratio Preference Optimization) [Hong et al., 2024]

```
핵심 혁신: SFT + 정렬을 단일 단계로

기존 파이프라인:
  SFT → DPO (2단계)

ORPO:
  SFT Loss + Odds Ratio Loss (1단계)

ORPO Loss:
  L_ORPO = L_SFT + λ · L_OR

  L_SFT: 선호 응답의 NLL (일반 SFT)

  L_OR:
    odds(y|x) = π_θ(y|x) / (1 - π_θ(y|x))
    L_OR = -E[log σ(log(odds(y_w|x) / odds(y_l|x)))]

직관:
  odds ratio: 해당 응답을 선택할 확률 vs 선택 안 할 확률의 비율
  선호 응답의 odds를 비선호보다 높이기

장점:
  SFT 단계 불필요 → 학습 파이프라인 단순화
  Reference 모델 없음 → 메모리 효율
  비슷한 성능

단점:
  λ 튜닝 필요
  단일 단계이므로 SFT, 정렬 각각 제어 어려움
```

---

## RPO (Robust Preference Optimization)

```
DPO의 Likelihood Displacement 문제:
  이론적으로 DPO에서 선호 응답의 절대 확률이 낮아질 수 있음
  (참조 대비 상대적으로만 높아짐)

RPO = DPO + NLL Loss:
  L_RPO = L_DPO + α · L_NLL(y_w)

  L_NLL(y_w): 선호 응답의 직접 NLL 최소화

효과:
  선호 응답의 절대 확률도 높아지도록 강제
  Alignment + Language modeling 능력 유지
```

---

## Online DPO 및 Iterative 방식

```
DPO의 가장 큰 문제: Offline (정적 데이터셋)
  → Policy가 변함에 따라 데이터 분포와 불일치

Iterative DPO:
  1. Policy로 응답 생성
  2. 생성된 응답 비교 (human or RM)
  3. DPO 업데이트
  4. 반복

Self-Play Fine-Tuning (SPIN):
  Policy와 이전 버전 Policy 대결
  현재 Policy: 이전 Policy의 응답 거부
  이전 Policy: 현재 Policy의 응답 받아들이기
  → 이전 버전에서 지속 개선

Online DPO (with RM):
  매 배치마다:
    현재 policy로 응답 쌍 생성
    RM으로 preference 레이블 부여
    DPO 업데이트
  → Real-time 데이터 갱신

RLHF-DPO 하이브리드:
  PPO로 탐색, DPO로 안정화
  또는 온라인 DPO + KL constraint
```

---

## RLAIF (RL from AI Feedback)

```
Human Feedback 대신 AI가 피드백 제공:
  Constitutional AI (Anthropic, 2022)
  → AI Supervisor가 헌법 원칙에 따라 응답 평가

Constitutional AI 프로세스:
  1. Supervised Phase (SL-CAI):
     a. 유해 응답 생성 (red-teaming)
     b. 헌법 원칙으로 자체 비판
     c. 수정된 응답 생성
     d. 수정 응답으로 SFT

  2. RL Phase (RLAIF):
     a. SL-CAI 모델로 응답 쌍 생성
     b. AI judge가 헌법 원칙으로 평가 (선호도)
     c. RM 학습
     d. RLHF로 최종 모델 학습

장점:
  Human annotation 비용 절감
  헌법 원칙으로 일관성 있는 피드백
  확장성 (대규모 데이터 생성 가능)

단점:
  AI bias가 강화될 수 있음
  헌법 원칙 설계가 critical
```

---

## Step-DPO (Reasoning용 DPO)

```
일반 DPO의 reasoning 적용 문제:
  전체 응답 단위로 선호 비교
  → 중간 단계의 오류 위치 모름
  → 비효율적인 학습 신호

Step-DPO:
  각 추론 단계(step)별로 선호 비교

  데이터: (x, s_1,...,s_k, [s_k+1_w, s_k+1_l])
    s_k까지는 동일, s_k+1에서 correct/incorrect로 분기

  효과:
    잘못된 단계를 정밀하게 학습
    PRM 레이블과 결합 가능
    수학/코딩 reasoning에서 DPO 대비 향상

  구현:
    각 단계를 별도 DPO 데이터포인트로
    Stage k에서의 context를 prefix로 사용
```

```python
class StepDPODataset:
    """Step-level DPO 데이터 준비"""

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def create_step_pairs(self, problem: str, steps_prefix: list[str],
                          correct_step: str, incorrect_step: str) -> dict:
        """특정 단계에서 correct/incorrect 쌍 생성"""
        prefix = f"Problem: {problem}\n" + "\n".join(steps_prefix)

        chosen = prefix + f"\nNext step: {correct_step}"
        rejected = prefix + f"\nNext step: {incorrect_step}"

        chosen_tokens = self.tokenizer(chosen, return_tensors="pt",
                                       max_length=self.max_length, truncation=True)
        rejected_tokens = self.tokenizer(rejected, return_tensors="pt",
                                         max_length=self.max_length, truncation=True)

        # prefix length (마스킹용)
        prefix_length = len(self.tokenizer(prefix)["input_ids"])

        return {
            "chosen_input_ids": chosen_tokens["input_ids"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "prefix_length": prefix_length  # prefix는 loss 계산에서 제외
        }
```

---

## 선호 데이터 품질

```
DPO 계열의 공통 병목:
  모델 학습 품질 = 선호 데이터 품질

좋은 선호 데이터 특성:
  1. Informativeness:
     y_w와 y_l의 차이가 명확해야 함
     비슷한 응답 쌍 → 약한 학습 신호

  2. Diversity:
     다양한 태스크, 도메인, 난이도
     특정 패턴에 치우치면 편향 학습

  3. Consistency:
     Annotator 간 일치도 높아야 함
     Cohen's κ > 0.7 목표

  4. Balance:
     helpfulness와 harmlessness 균형
     한쪽만 최적화 시 trade-off 발생

데이터 수집 방법:
  Human: 비용 높지만 가장 신뢰
  GPT-4: 저렴, 일관성 높음 (but AI 편향)
  Verifiable: 코딩/수학 자동 검증 (가장 신뢰)

데이터 크기:
  DPO: 수천~수만 쌍 (RLHF보다 적어도 됨)
  Ultrafeedback: 200K pairs (공개 데이터셋)
  Nectar: 분야별 183K pairs
```

### DPO 학습 모니터링
```python
class DPOTrainingMonitor:
    """DPO 학습 상태 모니터링"""

    def __init__(self):
        self.metrics = {
            "chosen_rewards": [],
            "rejected_rewards": [],
            "reward_margin": [],
            "reward_accuracy": [],
        }

    def update(self, chosen_rewards: torch.Tensor,
               rejected_rewards: torch.Tensor):
        margin = (chosen_rewards - rejected_rewards).mean().item()
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

        self.metrics["chosen_rewards"].append(chosen_rewards.mean().item())
        self.metrics["rejected_rewards"].append(rejected_rewards.mean().item())
        self.metrics["reward_margin"].append(margin)
        self.metrics["reward_accuracy"].append(accuracy)

    def check_health(self) -> dict:
        """학습 상태 진단"""
        recent = lambda key, n=100: self.metrics[key][-n:]

        issues = []
        margin = sum(recent("reward_margin")) / len(recent("reward_margin"))
        accuracy = sum(recent("reward_accuracy")) / len(recent("reward_accuracy"))

        # 경고 조건
        if margin < 0:
            issues.append("CRITICAL: chosen < rejected (wrong direction)")
        elif margin < 0.1:
            issues.append("WARNING: reward margin too small (weak signal)")

        if accuracy < 0.5:
            issues.append("CRITICAL: accuracy < 50% (random chance)")
        elif accuracy > 0.99:
            issues.append("WARNING: accuracy too high (possible overfitting)")

        return {
            "reward_margin": margin,
            "reward_accuracy": accuracy,
            "issues": issues,
            "status": "unhealthy" if issues else "healthy"
        }
```

---

## 비교 요약

| 방법 | RM 필요 | 쌍 필요 | Ref 필요 | 단계 | 주요 특징 |
|------|---------|---------|---------|------|---------|
| PPO | O | O | O | 2 | 온라인 RL, 강력 |
| DPO | X | O | O | 2 | 단순, 안정 |
| IPO | X | O | O | 2 | 노이즈 강건 |
| KTO | X | X | O | 2 | 단독 레이블 |
| SimPO | X | O | X | 2 | 메모리 효율 |
| ORPO | X | O | X | 1 | SFT+정렬 통합 |
| RPO | X | O | O | 2 | Displacement 방지 |
| Step-DPO | X | O | O | 2 | 단계별 추론 |

---

## Further Questions

**Q. DPO가 PPO보다 항상 좋은가?**
> 아님. DPO는 정적 데이터, PPO는 동적 탐색. 복잡한 reasoning(수학, 코딩)에서는 PPO/GRPO 등 온라인 RL이 더 효과적. 단순한 instruction following에서는 DPO로 충분. 비용 제약 시: DPO 먼저 시도, 성능 부족 시 GRPO/PPO.

**Q. SimPO에서 길이 정규화가 중요한 이유?**
> DPO 없이 단순 로그 확률 비교 → 짧은 응답이 유리 (토큰 수가 적으면 log prob 합 절댓값 작음 → 더 높음). SimPO는 토큰 수로 나눠 "토큰당 평균 확률" 비교 → 길이 편향 제거. 더 공정한 품질 비교.

**Q. 왜 ORPO가 SFT 단계를 없애도 되는가?**
> L_SFT가 ORPO 손실에 포함되어 있음. 선호 응답의 NLL 최소화 = SFT 역할. 동시에 L_OR로 비선호 응답 억제. 하지만 SFT만큼 강하지 않을 수 있어 데이터 품질이 매우 중요.

**Q. Offline DPO vs Online DPO 언제 각각 쓰나?**
> Offline DPO: 고품질 정적 데이터셋 있을 때, 빠른 실험. Online DPO: 지속적 개선 시스템, 현재 policy에 맞는 데이터 필요 시. 실제 배포 시스템은 Online + Human feedback 루프가 이상적.

**Q. DPO에서 β 값 선택 기준은?**
> β는 KL divergence 제약 강도. 작은 β (0.01): Policy 자유도 높음, overfitting 위험. 큰 β (0.5): Reference 근처에 고정, 학습 신호 약함. 보통 0.05~0.2 시작, 선호 응답 reward margin 모니터링하며 조정. β 너무 크면 선호/비선호 reward 차이 벌어지지 않음.

**Q. Constitutional AI가 일반 RLHF와 다른 점은?**
> 일반 RLHF: 인간이 직접 응답 비교. Constitutional AI: AI Supervisor가 명시된 원칙(헌법)으로 자동 평가 → Human annotation 대체. 핵심: 원칙을 자연어로 명시 → 일관성 있는 피드백. 단점: 헌법 설계 quality가 결과를 결정, AI의 가치 판단에 의존.

**Q. DPO 학습 시 흔한 실패 패턴과 해결책은?**
> 1) Reward margin이 0에 수렴: β 줄이기, 데이터 품질 검토. 2) chosen reward가 계속 하락 (Likelihood displacement): RPO 사용 (NLL 추가). 3) 긴 응답 편향: SimPO 또는 length penalty 추가. 4) 분포 이동 심함: Iterative DPO로 데이터 갱신. 주요 모니터링: reward margin, accuracy, KL divergence.
