# 최적화 (Optimization)

## 옵티마이저

### AdamW (표준)

```
Adam + Weight Decay 분리

Adam:
  m_t = β₁ m_{t-1} + (1-β₁) g_t          # 1차 모멘트 (방향)
  v_t = β₂ v_{t-1} + (1-β₂) g_t²         # 2차 모멘트 (스케일)
  m̂_t = m_t / (1 - β₁ᵗ)                  # bias correction
  v̂_t = v_t / (1 - β₂ᵗ)
  θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)

AdamW (Loshchilov & Hutter, 2019):
  θ_t = θ_{t-1} - lr · (m̂_t / (√v̂_t + ε) + λ·θ_{t-1})
  ← weight decay를 Adam update와 분리 (L2 regularization과 다름)

하이퍼파라미터:
  β₁ = 0.9 (1차 모멘트 decay)
  β₂ = 0.95 (LLM에서는 0.999 대신 0.95 많이 사용)
       → 0.999: 더 긴 기억, 안정적이지만 적응 느림
       → 0.95: 더 빠른 적응, LLM 학습에 적합
  ε = 1e-8 (수치 안정성)
  λ (weight decay) = 0.01 ~ 0.1 (기본 0.1)

메모리 요구량:
  모델 파라미터: P
  모멘트 (m, v): 2P (FP32)
  마스터 가중치: P (FP32, mixed precision 시)
  총: ~4P (FP32) + 학습 파라미터의 2배
```

```python
import torch
from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),   # LLM 표준
    eps=1e-8,
    weight_decay=0.1,
)

# 파라미터 그룹별 weight decay 조정
# Embedding, LayerNorm, bias에는 weight decay 적용 안 함
def get_parameter_groups(model, weight_decay=0.1):
    decay = set()
    no_decay = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name or "embedding" in name:
            no_decay.add(name)
        else:
            decay.add(name)

    param_dict = {n: p for n, p in model.named_parameters()}
    return [
        {"params": [param_dict[n] for n in sorted(decay)], "weight_decay": weight_decay},
        {"params": [param_dict[n] for n in sorted(no_decay)], "weight_decay": 0.0},
    ]
```

---

## 메모리 효율 옵티마이저

### Adafactor [Shazeer & Stern, 2018]

```
Adam의 2차 모멘트 v_t를 행/열 인수분해로 근사:

행렬 파라미터 W (m×n):
  Full v: O(mn) 메모리 필요
  Adafactor: row factor R (m) + col factor C (n) = O(m+n)

  v̂ ≈ R · Cᵀ / (Rᵀ · 1) (외적 근사)

스칼라/벡터 파라미터:
  단일 스케일 값으로 근사

추가 특성:
  No first moment: momentum 없음 (메모리 추가 절약)
  Relative step size: 파라미터별 상대적 학습률

메모리:
  Adam: 2P (float) 모멘트
  Adafactor: O(P^(1/2)) 수준 (행렬의 경우)

단점:
  학습 불안정성 증가 가능
  큰 학습률 사용 어려움

사용: T5, PaLM, Gemini 학습
```

### SOAP (Simultaneous Orthogonalization and Preconditioning, 2024)

```
Shampoo (2차 최적화) + Adam 결합:

Shampoo:
  각 행렬 파라미터에 대해 왼쪽/오른쪽 preconditioner 유지
  L_t = Σ G_t G_t^T,  R_t = Σ G_t^T G_t
  업데이트: W = -lr · L^{-1/2} G R^{-1/2}

SOAP:
  Adam을 "preconditioned 기저"에서 실행
  → Adam의 안정성 + Shampoo의 빠른 수렴

결과:
  Adam 대비 2-3× 빠른 수렴
  메모리: Adam의 2배 수준
  소형 모델(1B 이하)에서 특히 효과적
```

### Muon [Kosson et al., 2024]

```
직교화(orthogonalization) 기반 옵티마이저:
  행렬 파라미터의 gradient를 직교 업데이트로 변환

  Nesterov momentum:
    m_t = β m_{t-1} + g_t
    g̃_t = β m_t + g_t  (lookahead)

  Orthogonalization:
    O = NewtonSchulz(g̃_t)  # 직교 행렬에 가까운 업데이트
    θ_t = θ_{t-1} - lr · O

NewtonSchulz iteration (수렴 빠름):
  X_{k+1} = 1.5 X_k - 0.5 X_k X_k^T X_k

최신 트렌드:
  NanoGPT-speedrun에서 AdamW 대비 빠른 수렴 확인
  여러 언어모델 실험에서 효과적
```

### Lion (EvoLved Sign Momentum, Chen et al., 2023)

```
Sign gradient descent 기반 경량 옵티마이저:

업데이트:
  m_t = β₁ m_{t-1} + (1-β₁) g_t   (모멘텀)
  θ_t = θ_{t-1} - lr · sign(β₂ m_{t-1} + (1-β₂) g_t)
  m_t = β₁ m_{t-1} + (1-β₁) g_t   (모멘텀 업데이트)

특징:
  sign: 방향만 사용, 크기 무시
  1차 모멘트만 저장 → Adam 대비 1/2 메모리
  학습률: Adam의 1/10 수준 사용

성능:
  ViT, CLIP 학습에서 AdamW보다 좋은 경우 있음
  LLM 학습에서는 혼재된 결과
```

### Schedule-Free Optimizer (Defazio, 2024)

```
Learning rate schedule이 불필요:
  내부적으로 "running average" 유지
  외부 LR schedule 없이 비슷한 효과

주요 아이디어:
  θ = current parameters
  z = "scheduled average"
  x = θ + momentum term (평가/예측에 사용)

  매 스텝 후: x와 θ가 천천히 수렴

장점: LR schedule 튜닝 불필요
실용: 여러 NLP 태스크에서 AdamW+cosine에 필적
```

---

## 혼합 정밀도 훈련 (Mixed Precision)

```
FP32: 전체 정밀도, 메모리 4 bytes/param
  지수 8비트 + 가수 23비트 + 부호 1비트

FP16: 반정밀도, 2 bytes/param
  지수 5비트 + 가수 10비트 + 부호 1비트
  최대값: 65504 (overflow 쉬움)
  최솟값: 6e-8 (underflow 쉬움)

BF16: Brain Float 16, 2 bytes/param
  지수 8비트 + 가수 7비트 + 부호 1비트
  FP32와 동일한 지수 범위 (overflow 없음)
  정밀도는 낮음 (가수 23 → 7)

BF16가 FP16보다 LLM에 선호되는 이유:
  FP16은 최대값 65504 → 큰 gradient에서 overflow 빈번
  BF16은 지수 8비트 (FP32와 같음) → 안전한 범위 유지
  손실: FP16 정밀도 저하로 수치 오류

AMP (Automatic Mixed Precision):
  Forward pass: BF16 (빠름, 메모리 절약)
  Gradient: FP32 (underflow 방지)
  Master weights: FP32 (정밀한 업데이트)
  Optimizer state: FP32 (정밀도 필요)
```

```python
from torch.cuda.amp import autocast, GradScaler

# FP16 AMP (GradScaler 필요)
scaler = GradScaler()

for batch in dataloader:
    with autocast(dtype=torch.float16):
        loss = model(batch)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # gradient를 FP32로 변환
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

# BF16 AMP (Ampere GPU 이상, GradScaler 불필요)
for batch in dataloader:
    with autocast(dtype=torch.bfloat16):
        loss = model(batch)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

---

## FP8 학습 (최신)

```
FP8 (8비트 부동소수점):
  E4M3: 지수 4비트, 가수 3비트 → 순방향에 적합
  E5M2: 지수 5비트, 가수 2비트 → 기울기에 적합

DeepSeek-V3에서 최초 대규모 성공:
  Forward: FP8 행렬 곱
  Backward: FP8 gradient 연산
  Master weights: BF16 유지

스케일링 요령:
  Block-wise quantization: 작은 블록마다 별도 스케일 계수
  → 정밀도 손실 최소화

메모리 절약:
  BF16 → FP8: 2배 메모리 절약
  더 큰 배치 또는 더 긴 시퀀스 가능

하드웨어:
  NVIDIA H100 이상 (Hopper): FP8 tensor core 지원
  A100: 소프트웨어 FP8 (느림)
```

---

## Gradient Techniques

### Gradient Clipping

```python
# L2 norm clipping (표준)
total_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0  # 보통 1.0
)

# max_norm 초과 시:
# gradient 전체를 스케일링: g = g × (max_norm / ||g||)
# 방향은 유지, 크기만 제한

# 왜 필요한가:
# Loss spike 시 큰 gradient → 파라미터 폭발 방지
# 특히 초기 학습, 긴 시퀀스에서 중요

# 모니터링:
print(f"Gradient norm: {total_norm:.4f}")
# 정상: 0.1 ~ 2.0
# 위험: > 10 (불안정)
# 너무 작음: < 0.01 (vanishing gradient 가능)
```

### Gradient Checkpointing (Activation Checkpointing)

```python
import torch

# 표준 사용법
def forward_with_checkpointing(module, x):
    return torch.utils.checkpoint.checkpoint(module, x)

# 전체 모델에 적용 (Transformers 라이브러리)
model.gradient_checkpointing_enable()

# 메모리 vs 속도 트레이드오프:
# Forward: 중간 activation 버림
# Backward: 필요한 activation 재계산
# 메모리: O(n) → O(√n)
# 속도: ~30-40% 느려짐

# Selective checkpointing:
# 메모리가 많이 필요한 레이어만 적용
# 예: Attention layer만, FFN은 일반 forward
```

### Gradient Accumulation

```python
# 큰 배치 시뮬레이션
accumulation_steps = 8  # 8 step 누적 → 8배 큰 effective batch

optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    with autocast(dtype=torch.bfloat16):
        loss = model(batch) / accumulation_steps  # 스케일 조정

    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

# 효과: per_device_batch=4, accumulation=8 → effective batch=32
# GPU 메모리 제한에서 큰 배치 효과 달성
```

---

## 정규화 기법

### Weight Decay

```
L2 패널티 추가로 큰 가중치 억제:
  AdamW: θ_t = θ_{t-1} - lr · (...) - lr · λ · θ_{t-1}

일반 Adam + L2 vs AdamW 차이:
  Adam + L2: gradient = g + λθ → adaptive lr로 왜곡됨
  AdamW: weight decay를 adaptive lr과 독립적으로 적용
  → AdamW가 올바른 weight decay 구현

λ 선택:
  LLM pretraining: 0.1 (표준)
  Fine-tuning: 0.01 (더 보수적)
  LoRA: 0.0 (일반적으로 사용 안 함)
```

### Z-loss (PaLM, 2022)

```
Router/logit의 크기를 제한하는 정규화:

L_z = β · (1/n Σᵢ log Σⱼ exp(zᵢⱼ))²

  zᵢⱼ: 토큰 i의 expert j에 대한 logit

효과:
  Router collapse 방지 (MoE에서 중요)
  Softmax의 sharp 분포 완화
  더 안정적인 학습

Softmax 안정화:
  일반적 안정화: softmax(z - max(z))
  Z-loss: logit 자체가 너무 커지는 것 방지
```

---

## Learning Rate Scheduling

```python
import math

def get_cosine_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine Decay with Warmup (표준)"""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# 일반적 값:
# max_lr = 3e-4 (7B), 1e-4 (13B), 3e-5 (70B+)
# min_lr = max_lr / 10
# warmup_steps = 1000 ~ 4000 (전체의 1~2%)
# max_steps = 전체 학습 스텝
```

### WSD (Warmup-Stable-Decay) 스케줄

```
1. Warmup: 0 → max_lr (전체의 1~2%)
2. Stable: max_lr 유지 (전체의 70~80%)
3. Decay: max_lr → min_lr (전체의 10~20%, 급격히)

장점:
  학습 중간에 "stable" 체크포인트에서 continue 가능
  다른 방향으로 파인튜닝 분기 가능
  Cosine보다 후반 급격한 decay로 더 나은 최종 성능

MiniCPM, Qwen 학습에서 사용:
  if step < warmup_steps:
      lr = max_lr * step / warmup_steps
  elif step < stable_end_steps:
      lr = max_lr
  else:
      # 지수적 decay
      lr = max_lr * decay_factor^(step - stable_end)
```

### Cyclical LR & Warm Restarts

```python
# Cosine Annealing with Warm Restarts (SGDR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=1000,   # 첫 주기 길이
    T_mult=2,   # 각 주기를 2배로 늘림
    eta_min=1e-6
)

# 효과: 여러 개의 local minima 탐색
# LR이 높아질 때 새 minima로 이동 가능
```

---

## μP (Maximal Update Parametrization)

```
문제: 모델이 커질수록 학습률 재튜닝 필요
  7B에서 최적 lr=3e-4가 70B에서는 맞지 않음

μP (Yang et al., 2022) 핵심 아이디어:
  파라미터 초기화와 학습률을 크기에 따라 조정
  → 다른 크기의 모델에서 동일한 lr 사용 가능

규칙:
  Width m (hidden size):
    Embedding 행렬: 초기화 std ∝ 1 (크기 무관)
    Hidden 행렬: 초기화 std ∝ 1/√m
    Attention logit: 1/m² (기존 1/√d_head 대신)
    Output 행렬: 학습률 ∝ 1/m

  → 작은 proxy 모델에서 찾은 lr을 큰 모델에 적용 가능

실용적 흐름:
  1. 작은 모델(125M)에서 lr 탐색
  2. μP 규칙에 따라 큰 모델(7B)에 같은 lr 적용
  3. 재튜닝 없이 좋은 성능

Cerebras, 일부 스타트업에서 활용
```

---

## 수렴 모니터링

```python
import wandb

# 주요 모니터링 지표
def log_training_metrics(step, loss, model, optimizer):
    # Gradient norm
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = total_norm ** 0.5

    # 실제 학습률
    current_lr = optimizer.param_groups[0]['lr']

    # Perplexity
    import math
    perplexity = math.exp(min(loss, 20))  # overflow 방지

    wandb.log({
        "train/loss": loss,
        "train/perplexity": perplexity,
        "train/grad_norm": grad_norm,
        "train/lr": current_lr,
        "train/step": step,
    })

# MFU (Model FLOP Utilization):
# 이론적 최대 FLOP 대비 실제 달성 FLOP 비율
# A100 80GB: ~312 TFLOP/s (BF16)
# LLaMA-3 학습 목표: 38-45% MFU
def compute_mfu(model_params, seq_len, batch_size, iter_time, gpu_peak_flops):
    # Transformer forward pass FLOP 추정
    # 6 × num_params × seq_len × batch_size (approximate)
    flops_per_step = 6 * model_params * seq_len * batch_size
    actual_flops = flops_per_step / iter_time
    return actual_flops / gpu_peak_flops
```

---

## Loss Spike 처리

```
Loss spike: 학습 중 갑자기 loss가 급증하는 현상
  원인: 비정상적인 데이터 배치, 큰 gradient
  빈도: 수천 step에 1회 정도 발생 가능

처리 방법:
  1. Gradient skip: loss가 threshold 초과 시 step 건너뜀
     if loss > prev_loss * 3:  # 3배 이상 증가
         optimizer.zero_grad()
         continue

  2. Checkpoint rollback: 이전 안정 체크포인트로 복원
     마지막 100 step 이전 checkpoint로 복원
     문제 배치 제거 후 재학습

  3. Dynamic gradient clipping: clip 기준을 낮춤
     if grad_norm > 10: clip to max_norm=0.1

  4. Batch skip: 문제 배치 탐지 및 제거
     극단적 loss를 일으키는 데이터 필터링

LLaMA-3 (Meta): 자동 rollback 메커니즘 구현
  3M step에서 수 회 spike 처리
```

---

## Further Questions

**Q. Adam vs SGD 언제 사용?**
> LLM은 거의 항상 AdamW. SGD: vision 모델에 좋지만 희소 gradient 많은 NLP에 불리, 모멘텀 없으면 수렴 느림. AdamW의 적응적 lr이 다양한 파라미터 스케일(embedding 크기 차이) 처리에 효과적. SGD+Momentum: 일부 image classification에서 AdamW보다 좋지만 LLM에서는 불리.

**Q. Weight decay를 Adam에 별도로 적용해야 하는 이유?**
> 일반 Adam에 L2 regularization을 더하면: g' = g + λθ → Adam의 적응적 lr이 weight decay에도 적용 → 큰 가중치에 낮은 lr → weight decay 효과 왜곡. AdamW는 weight decay를 gradient 업데이트와 분리: θ -= lr × adam_update + lr × λ × θ → 진정한 weight decay 효과.

**Q. BF16이 FP16보다 선호되는 이유?**
> BF16은 FP32와 동일한 지수 범위(8비트) → overflow/underflow 없음. FP16은 최대 65504 → 큰 gradient에서 overflow 빈번 → GradScaler 필요. 정밀도(가수 비트)는 BF16이 낮지만 LLM 학습에서는 충분. Ampere(A100) 이상 GPU에서 BF16 tensor core 지원 → 속도도 같음.

**Q. μP가 왜 실용적인가?**
> LLM 학습의 주요 비용: hyperparameter 탐색. 매번 큰 모델 실험 불가. μP: 작은 모델(125M)에서 최적 lr 탐색 → 큰 모델(7B)에 바로 적용. Transfer가 잘 됨 (Width scaling). 단, depth scaling은 여전히 실험 필요. 실용적으로 탐색 비용 10-100배 절약.
