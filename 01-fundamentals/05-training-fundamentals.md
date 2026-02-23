# 학습 기초 개념 (Training Fundamentals)

> Transformer를 학습할 때 반드시 이해해야 하는 핵심 개념들.

---

## 1. 역전파 (Backpropagation) in Transformer

### Gradient Flow 이해
```
Forward: x → Transformer Block × N → logits → loss
Backward: ∂L/∂x ← Block_N ← ... ← Block_1

Residual Connection의 역할:
  x_out = x_in + F(x_in)
  ∂L/∂x_in = ∂L/∂x_out · (1 + ∂F/∂x_in)
              ↑                  ↑
           gradient          gradient
           직접 전달          through F

  "1"이 항상 있으므로 gradient가 직접 전달
  → 깊은 네트워크에서도 gradient vanishing 방지
```

### Pre-LN vs Post-LN Gradient 분석
```
Post-LN (원 논문):
  x → [Attn] → Add → LN → [FFN] → Add → LN

  초기화 직후: LN이 각 블록 출력을 정규화
  → 출력의 분산 = 1
  → Residual 경로의 기여가 "Add" 에서 묻힘
  → 초기 gradient vanishing → warmup 필수

Pre-LN (현대 표준):
  x → LN → [Attn] → Add → LN → [FFN] → Add

  Residual 경로: normalized 되지 않음
  → 정보가 직접 전달 (residual stream 개념)
  → 더 안정적 gradient flow
  → warmup 없이 학습 가능

수학적으로:
  Post-LN: 출력 분산 ∝ 1/√L (레이어 수)
  Pre-LN: 출력 분산 ∝ L (발산 가능 → 해결: init scale 조정)
```

### 수치 안정성 문제
```
Attention Score Overflow:
  score = QK^T / sqrt(d_k)
  d_k=64일 때 √d_k = 8 로 나눔
  그래도 score 값이 크면 exp(score) → inf

  해결: Online softmax (Flash Attention)
    m = max(score)
    stable_score = score - m
    softmax = exp(stable_score) / Σexp(stable_score)

Gradient Explosion (FFN):
  SwiGLU: gate * up
  두 경로의 gradient가 곱해짐
  → 큰 activation 시 gradient 폭발

  해결: Gradient Clipping + 적절한 초기화
```

---

## 2. 초기화 전략 (Initialization)

### 왜 초기화가 중요한가
```
나쁜 초기화:
  너무 큰 weight → activation 폭발 → NaN
  너무 작은 weight → activation 소실 → 학습 안됨

이상적:
  각 레이어의 activation 분산이 일정하게 유지
  gradient 분산도 일정하게 유지
```

### Xavier / Glorot 초기화
```
가정: Linear layer, Tanh/Sigmoid activation

분산 유지 조건:
  Var[output] = Var[input]
  → W ~ Uniform(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
  또는 N(0, 2/(fan_in + fan_out))

fan_in: 입력 차원
fan_out: 출력 차원

직관: fan_in이 크면 작은 weight (합이 일정하도록)
```

### He / Kaiming 초기화
```
가정: ReLU activation (절반이 0)

W ~ N(0, 2/fan_in)  (ReLU의 경우)
W ~ N(0, 2/fan_out) (SwiGLU 등의 경우)

PyTorch 기본:
  nn.Linear: Kaiming uniform (fan_in)
  nn.Embedding: N(0, 1) → 보통 너무 큼
```

### LLM에서의 실제 초기화
```
표준 LLM 초기화:
  Embedding: N(0, 1) → 학습 중 조정
  Linear: N(0, std) where std = 0.02 (GPT) 또는 1/√(d_model)
  Output projection: N(0, 0.02/√(2L)) (depth 스케일링)
    → 깊은 모델에서 residual 기여 감소
  LM head: embedding과 공유 (weight tying)

μP (maximal update parameterization):
  fan_in에 따른 체계적 스케일링
  작은 모델에서 찾은 LR → 큰 모델에 적용 가능
```

---

## 3. 정규화 기법

### LayerNorm 내부 동작
```
LayerNorm(x):
  μ = mean(x)  # 평균
  σ² = var(x)  # 분산
  x_norm = (x - μ) / √(σ² + ε)
  output = γ * x_norm + β  # 학습 가능한 scale, shift

특징:
  - Batch 크기에 독립 (단일 예시에서도 동작)
  - Sequence 길이에 독립
  - 각 position에서 d_model 차원에 대해 정규화

BatchNorm vs LayerNorm:
  BatchNorm: 배치 내 같은 feature 정규화
    → 작은 배치에서 불안정
    → RNN, Transformer에 부적합
  LayerNorm: 각 예시 내 feature 정규화
    → 배치 크기 독립
    → Autoregressive에 적합
```

### RMSNorm (LLaMA, Qwen 등)
```
LayerNorm의 단순화:
  RMS(x) = √(1/d · Σ xᵢ²)
  RMSNorm(x) = γ * x / RMS(x)

  평균 빼는 연산(centering) 제거 → 속도 향상
  β (shift) 제거

이유:
  실험적으로 centering이 성능에 거의 영향 없음
  연산량 감소 (약 7-8% 속도 향상)

수식:
  γ * x / √(mean(x²) + ε)
```

### Pre-LN의 스케일 발산 문제
```
Pre-LN에서:
  x_{L+1} = x_L + Block_L(LN(x_L))

  |x_L| ~ √L (레이어가 깊을수록 커짐)
  → 출력 logits이 매우 커질 수 있음

해결책:
  1. Output projection scaling: 1/√(2L) 스케일
     Residual 기여를 레이어 수에 반비례로 감소
  2. QK-Norm (DeepSeek): Q, K를 normalize
     Attention score 안정화
  3. Logit soft-capping (Gemma-2):
     logits = tanh(logits/cap) * cap
     출력 범위 제한
```

---

## 4. 손실 함수 (Loss Functions)

### Cross-Entropy Loss (언어 모델)
```
L = -Σ_t log P(x_t | x_{<t})

각 토큰 위치에서 정답 토큰의 로그 확률 합
평균: L_avg = L / T (총 토큰 수)

Perplexity:
  PPL = exp(L_avg) = exp(-1/T · Σ log P(xₜ|x<t))
  → 모델이 평균적으로 얼마나 많은 선택지 중 고민하는지
  → 낮을수록 좋음 (2.0: 완벽한 이진 예측, 100: 100 중 하나)

구현:
  logits: (B, T, vocab_size)
  labels: (B, T) [각 위치의 다음 토큰]
  loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1))
```

### Label Smoothing
```
Hard label: [0, 0, 1, 0, ...] (정답만 1)
Smooth label: [ε/(V-1), ..., 1-ε, ..., ε/(V-1)]

L_smooth = (1-ε) · CE(logits, y) + ε · H(uniform)

이점:
  과적합 방지
  모델이 너무 확신하지 않도록
  특히 작은 데이터셋에서 유효

LLM에서:
  대부분 label smoothing 없이 사용 (데이터가 충분히 큼)
  일부: ε = 0.1
```

### Instruction Tuning에서의 Loss Masking
```
문제: Prompt 부분에 대해서도 loss 계산하면
      → 모델이 prompt 재현 학습 (불필요)

해결: User/System 부분 마스킹
  labels: [-100, -100, ..., answer_tokens, ...]
  F.cross_entropy: label=-100이면 무시

구현:
  for i, message in enumerate(conversation):
    if message['role'] == 'assistant':
      # 이 부분만 loss 계산
      labels[start:end] = input_ids[start:end]
    else:
      labels[start:end] = -100  # 마스킹
```

---

## 5. 최적화 알고리즘 심화

### Adam & AdamW
```
Adam:
  m_t = β₁ m_{t-1} + (1-β₁) g_t       (1st moment, momentum)
  v_t = β₂ v_{t-1} + (1-β₂) g_t²      (2nd moment, RMS)
  m̂_t = m_t / (1 - β₁ᵗ)               (bias correction)
  v̂_t = v_t / (1 - β₂ᵗ)
  θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)

β₁ = 0.9, β₂ = 0.95 or 0.999, ε = 1e-8

AdamW:
  θ_t = θ_{t-1} - η · (m̂_t / (√v̂_t + ε) + λθ_{t-1})
  Weight decay를 gradient가 아닌 weight에 직접 적용
  L2 regularization과 다름! (adaptive LR의 영향 없음)

  Adam + L2: gradient에 λθ 추가 → adaptive LR에 의해 weight decay 효과 약화
  AdamW: weight에 직접 λθ → 항상 일정한 decay 효과
```

### Gradient Flow와 Weight Decay
```
Weight Decay (λ):
  목적: 가중치를 0 방향으로 당겨 일반화 향상
  값: 0.01 ~ 0.1 (LLM: 보통 0.1)

  Embedding에는 보통 weight decay 적용 안 함
  LM head에도 보통 제외 (weight tying 시 embedding과 같음)

  분리 설정:
    param_groups = [
      {'params': decay_params, 'weight_decay': 0.1},
      {'params': no_decay_params, 'weight_decay': 0.0}  # bias, norm
    ]
```

### Gradient Accumulation의 정확한 동작
```
목적: 작은 GPU 메모리로 큰 effective batch size 달성

올바른 구현:
  accum_steps = 4  # 4번 accumulate
  for i, batch in enumerate(dataloader):
    loss = model(batch) / accum_steps  # 스케일 중요!
    loss.backward()
    if (i + 1) % accum_steps == 0:
      optimizer.step()
      optimizer.zero_grad()

왜 loss를 나누나?:
  4번 backward → gradient 합산
  나누지 않으면 effective LR이 4× 커지는 효과
  → 불안정 학습

DDP + Gradient Accumulation:
  gradient sync를 마지막 step에만 (no_sync() context)
  → 통신 비용 감소
```

### Adafactor (메모리 효율)
```
Adam의 메모리 문제:
  m, v를 저장해야 함 → 2 × model_params (FP32)
  70B 모델: 560GB 추가 메모리

Adafactor:
  2D matrix의 v를 row vector + col vector로 근사
  r_t ≈ v_t^{row} ⊗ v_t^{col}
  → 메모리 O(max(n,m)) 대신 O(n+m)

  1st moment도 제거 (momentum = 0)
  → 메모리 추가: O(√params) 정도

단점:
  불안정할 수 있음
  warmup이 더 중요해짐
  LR 설정이 Adam보다 까다로움

사용: T5, 메모리 제한 학습에서
```

---

## 6. 토큰 예측의 이해

### Temperature와 샘플링
```
Logits → Temperature → Softmax → 샘플링

Temperature T:
  logits_scaled = logits / T
  P_T(x) = softmax(logits / T)

  T → 0: argmax (greedy, deterministic)
  T = 1: 원래 분포
  T > 1: 더 균등한 분포 (창의적, 다양)

Top-k Sampling:
  상위 k개 토큰만 유지, 나머지 -∞
  k=50: 다양성 vs 품질 균형

Top-p (Nucleus) Sampling:
  누적 확률이 p를 넘는 최소 토큰 집합
  p=0.9: 동적 크기 (확률 집중 시 작은 집합)

Repetition Penalty:
  이미 생성한 토큰의 logit를 감소
  반복 방지

Min-P Sampling:
  최고 확률의 min_p 이상인 토큰만 허용
  동적 filtering (최근 유망한 대안)
```

### Beam Search vs Sampling
```
Greedy Search:
  매 step 최고 확률 토큰 선택
  빠르지만 지역 최적 (local optima)

Beam Search:
  B개의 후보 시퀀스 유지
  각 step에서 B × vocab 조합 평가
  최종: 가장 높은 시퀀스 확률 선택
  단점: 반복 문제, 다양성 낮음

Sampling (확률적):
  분포에서 무작위 샘플
  다양성 높음, 품질 가변
  현대 LLM 대화에 주로 사용

Sampling + Best-of-N:
  N번 샘플링 → 가장 좋은 것 선택
  품질 향상 (Test-time compute scaling)
```

---

## 7. 긴 시퀀스 학습

### Position Interpolation 원리
```
문제: RoPE로 max_len=4096 학습 후 8192 토큰 처리?
  position 4097~8192은 학습 시 본 적 없음 → 성능 저하

Position Interpolation:
  기존: pos θ_i(pos) = pos / 10000^{2i/d}
  PI: θ_i(pos) = (pos/scale) / 10000^{2i/d}
  → 위치를 압축하여 학습 범위 내로
  scale = max_new_len / max_train_len

  예: 4096 → 8192 확장, scale=2
  → 8192 토큰이 0~4096 위치로 압축
  → 학습 시 봤던 범위

YaRN (Yet another RoPE extension):
  짧은 거리: interpolation 없음 (local 정확도 유지)
  긴 거리: NTK 방식으로 외삽
  LLaMA-3: YaRN 기반 컨텍스트 확장 적용
```

### Flash Attention과 긴 시퀀스
```
N = 128K 토큰, d = 128, h = 8 heads:
  기존 attention matrix: 128K × 128K = 16G 요소 × 2B = 32TB (FP16)
  → 불가능

Flash Attention:
  블록 크기: SRAM 맞게 (보통 64 또는 128)
  HBM 접근: O(N × d) (Q, K, V 각 1번)
  SRAM 사용: O(block_size × d)
  → N=128K도 처리 가능

Ring Attention (매우 긴 컨텍스트):
  N/P 토큰씩 P GPU에 분산
  GPU 간 K, V ring 방식으로 순환
  → 사실상 무한 컨텍스트 가능 (통신 오버헤드 있음)
```

---

## Further Questions

**Q1. Pre-LN에서 레이어가 깊어질수록 발생하는 문제는?**
```
Pre-LN: x_{L+1} = x_L + F(LN(x_L))
  각 레이어가 동일한 크기의 residual 추가
  → |x_L| ~ O(√L) (랜덤 워크)

문제:
  매우 깊은 모델에서 마지막 레이어 activation이 매우 큼
  → logit scale 폭발 → softmax 포화 → gradient 소실

해결:
  1. Output projection scaling: Residual 기여를 1/√(2L) 스케일
  2. 최종 layer norm (LLaMA 스타일): 출력 정규화
  3. QK-Norm: attention score 안정화
```

**Q2. AdamW의 weight decay와 L2 regularization이 다른 이유는?**
```
L2 reg (Adam + L2):
  gradient = ∇L + λθ
  Adam에서 adaptive scale 적용:
  update = Adam_scale(∇L + λθ)
  → λθ도 adaptive scaling → decay 강도가 adaptive lr에 의존

AdamW:
  gradient = ∇L만 사용
  update = Adam_scale(∇L) + λθ
  → weight decay는 항상 동일한 비율로 적용
  → 이론적으로 더 올바른 regularization

실제 차이:
  high lr + Adam + L2: weight decay 효과 과장됨
  AdamW: lr과 독립적인 consistent weight decay
  → 파인튜닝에서 더 안정적
```

**Q3. Temperature가 0에 가까울 때 greedy와 동일한 이유는?**
```
P_T(x_i) = exp(logit_i / T) / Σ_j exp(logit_j / T)

T → 0:
  최대 logit의 exp가 나머지에 비해 무한히 커짐
  → softmax → [0, ..., 0, 1, 0, ..., 0] (argmax)

T → ∞:
  모든 logit/T → 0
  → exp(0) = 1 모두 같음
  → softmax → uniform distribution

실용적:
  T=0.1: 거의 greedy, 변동 작음
  T=0.7: 품질/다양성 균형
  T=1.2: 창의적이지만 일관성 낮을 수 있음
```

**Q4. Gradient Clipping이 없으면 어떤 일이 발생하나?**
```
Loss spike 시나리오:
  나쁜 배치 → 큰 gradient → 큰 weight update
  → 다음 스텝 loss 더 큼 → 더 큰 gradient
  → 수렴 발산 (divergence)

수학적으로:
  gradient norm이 threshold보다 크면:
  g ← g × (max_norm / ||g||)

  방향은 유지, 크기만 제한 → 안전한 업데이트

모니터링:
  gradient norm을 로그에 기록
  norm이 max_norm에 가깝게 유지되면 → LR 낮추기
  norm이 갑자기 튀면 → 데이터 문제 의심
```
