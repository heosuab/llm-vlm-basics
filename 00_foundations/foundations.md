# Section 0: Overall Foundations

> LLM/VLM을 공부하기 전에 반드시 알아야 할 수학적·정보이론적·최적화 기초 개념들을 다룹니다.

---

## 0.1 Probability Basics

### Maximum Likelihood Estimation (MLE)

MLE는 주어진 데이터를 가장 잘 설명하는 파라미터 θ를 찾는 방법입니다.
데이터셋 D = {x₁, ..., x_N}이 주어졌을 때, log-likelihood를 최대화합니다.

```
log L(θ) = Σᵢ log p_θ(xᵢ)
θ* = argmax_θ Σᵢ log p_θ(xᵢ)
```

언어 모델에서는 autoregressive factorization을 사용합니다:

```
p_θ(x) = Π_t p_θ(x_t | x_1, ..., x_{t-1})
```

즉, 각 token의 조건부 확률의 곱으로 전체 시퀀스의 확률을 나타냅니다. MLE 목표는 이 log probability의 합을 최대화하는 것이고, 이는 next token prediction loss를 최소화하는 것과 동일합니다.

---

### Cross-Entropy Loss

참 분포 p(c)와 모델 분포 q(c) 사이의 cross-entropy:

```
H(p, q) = -Σ_c p(c) · log q(c)
```

언어 모델 학습에서는 레이블이 one-hot 벡터(정답 token만 1, 나머지 0)이므로:

```
L_CE = -log q(y_true)
```

즉 정답 token에 대한 모델의 log probability에 음수를 취한 값입니다.
Cross-entropy를 최소화하는 것 = NLL(Negative Log-Likelihood)을 최소화하는 것 = MLE를 수행하는 것이 모두 동일합니다.

---

### KL Divergence

KL Divergence는 두 분포가 얼마나 다른지를 측정합니다:

```
D_KL(p ‖ q) = Σ_x p(x) · log(p(x) / q(x))
```

중요한 성질:
- 항상 ≥ 0 (Jensen's inequality)
- p = q일 때만 0
- **비대칭**: D_KL(p ‖ q) ≠ D_KL(q ‖ p)

Cross-entropy와의 관계:
```
H(p, q) = H(p) + D_KL(p ‖ q)
```
즉, cross-entropy = entropy(p) + KL divergence.
p가 고정되어 있을 때 cross-entropy 최소화 = KL divergence 최소화.

**LLM에서의 활용**: RLHF에서 KL penalty로 policy가 너무 많이 벗어나지 못하게 제약합니다:

```
reward = r(x, y) - β · D_KL(π_θ(y|x) ‖ π_ref(y|x))
```

---

## 0.2 Information Theory

### Entropy

Entropy는 분포의 불확실성(information content)을 측정합니다:

```
H(p) = -Σ_x p(x) · log₂ p(x)
```

- 균등 분포일 때 최대 (가장 불확실)
- 한 값에 확률 1이 집중될 때 0 (완전히 확실)
- 단위: bits (log₂ 사용 시), nats (ln 사용 시)

---

### Perplexity

Perplexity(PPL)는 언어 모델이 test set을 얼마나 잘 예측하는지의 지표입니다:

```
PPL = exp( -1/N · Σ_t log p_θ(x_t | x_{<t}) )
    = exp(cross-entropy loss)
```

직관적으로, 모델이 다음 token을 고를 때 "몇 개 중에 고르는 것처럼 느끼는가"를 나타냅니다.
PPL = 10이면 평균적으로 10개의 equally likely한 선택지 중에서 고르는 것과 같습니다.

**주의사항**:
- 낮을수록 좋은 모델
- **같은 tokenizer를 사용한 모델끼리만** 비교 가능 (tokenization이 다르면 PPL 값 자체가 달라짐)
- 특정 domain의 PPL이 낮다고 해서 다른 task에서도 좋다는 보장 없음

---

## 0.3 Optimization Basics

### SGD (Stochastic Gradient Descent)

```
θ_{t+1} = θ_t - η · ∇_θ L(θ_t)
```

전체 데이터 대신 mini-batch로 gradient를 근사합니다. 기본적이지만 LLM 학습에는 단독으로 거의 사용되지 않습니다 (momentum, adaptive learning rate가 없어 수렴이 느림).

---

### AdamW

Adam(Adaptive Moment Estimation)에 **decoupled weight decay**를 결합한 옵티마이저로, 현재 LLM 학습의 표준입니다:

```
g_t   = ∇_θ L(θ_t)                              # gradient
m_t   = β₁ · m_{t-1} + (1 - β₁) · g_t          # 1st moment (mean)
v_t   = β₂ · v_{t-1} + (1 - β₂) · g_t²         # 2nd moment (variance)
m̂_t  = m_t / (1 - β₁ᵗ)                         # bias correction
v̂_t  = v_t / (1 - β₂ᵗ)                         # bias correction
θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε) - η·λ·θ_t  # update + weight decay
```

**하이퍼파라미터 기본값**:
| 파라미터 | 값 | 의미 |
|---------|-----|------|
| β₁ | 0.9 | gradient의 exponential moving average 계수 |
| β₂ | 0.95~0.999 | squared gradient의 EMA 계수 |
| ε | 1e-8 | 수치 안정성을 위한 작은 값 |
| λ | 0.01~0.1 | weight decay 강도 |

**Adam vs AdamW 차이**: 일반 Adam은 weight decay를 gradient에 더해서 적용하므로 adaptive learning rate의 영향을 받습니다. AdamW는 weight decay를 파라미터에 직접(별도로) 적용합니다. LLM 학습에서는 AdamW가 표준입니다.

---

### Learning Rate Schedule

**Warmup**:
학습 초반에 learning rate를 0에서 목표값까지 선형적으로 증가시킵니다.
필요한 이유:
1. 학습 초반에 가중치가 랜덤 초기화 상태라 gradient가 불안정함
2. Attention의 softmax가 초반에 sharp한 분포를 만들어 gradient explode 위험
3. Warmup을 통해 모델이 먼저 데이터 분포를 "감"을 잡게 함

보통 전체 학습 step의 1~5%를 warmup에 사용합니다.

**Cosine Decay**:
```
η_t = η_min + ½ · (η_max - η_min) · (1 + cos(π · t / T))
```
학습이 진행될수록 learning rate가 cosine curve를 따라 부드럽게 감소합니다. 현재 LLM pretraining의 표준입니다.

**Linear Decay**: 단순히 선형으로 감소. Fine-tuning에서 자주 사용됩니다.

---

## 0.4 Scaling Laws

### Chinchilla Scaling

2022년 DeepMind의 Chinchilla 논문은 LLM 학습에 대한 핵심적인 통찰을 제공했습니다.

**핵심 발견**: Compute budget C가 고정되어 있을 때, 모델 크기 N과 학습 데이터 양 D를 **균등하게** 늘려야 compute-optimal합니다.

```
N_opt ∝ √C
D_opt ∝ √C
→ D_opt ≈ 20 × N_opt  (tokens per parameter)
```

**이전(GPT-3 시대)의 관행**:
모델은 크게 만들되, 데이터는 상대적으로 적게 사용했습니다.
GPT-3 (175B params, ~300B tokens) → **심하게 undertrained** 상태

**Chinchilla 실험 결과**:
Chinchilla (70B params, 1.4T tokens)가 Gopher (280B params, 300B tokens)를 대부분의 벤치마크에서 이겼습니다. 더 작은 모델이지만 충분한 데이터로 학습했기 때문입니다.

**이후의 흐름**:
LLaMA 시리즈는 "inference-optimal" 관점을 도입했습니다. Chinchilla-optimal 이상으로 데이터를 사용해 더 작은 모델을 더 잘 학습시키면, 추론 시 비용이 낮으면서도 성능이 뛰어난 모델을 만들 수 있습니다.

---

## 0.5 Compute vs Data Tradeoff

### FLOP 계산

Transformer 학습의 대략적인 FLOP 수:
```
C ≈ 6 · N · D
```
- C: 총 FLOPs
- N: 모델 파라미터 수
- D: 학습 token 수
- 6 = 2(forward) + 4(backward, gradient는 2× forward)

**실용적 의미**:
- GPU 대수 × FLOPs/GPU/sec × 학습 시간 = C
- C가 정해지면, N과 D의 최적 비율이 존재 (Chinchilla)

### Compute-Optimal Training

| 모델 | 파라미터 | 데이터 | 특성 |
|------|---------|-------|------|
| GPT-3 | 175B | 300B tokens | Model-heavy |
| Chinchilla | 70B | 1.4T tokens | Compute-optimal |
| LLaMA-3 | 8B | 15T tokens | Data-heavy (inference-optimal) |

---

## 0.6 Emergent Abilities

### 정의

"Emergent abilities"는 작은 모델에서는 나타나지 않다가 모델이 일정 규모를 넘어서면 갑자기 나타나는 능력들을 말합니다. GPT 계열 모델들을 관찰하면서 발견되었습니다.

### 대표적인 사례

| Ability | 출현 규모 |
|---------|---------|
| Chain-of-Thought reasoning | ~100B params |
| Multi-step arithmetic | ~10B params |
| Word unscrambling | ~10B params |
| Instruction following | ~7B params (with RLHF) |
| In-context learning (few-shot) | ~few billion |

### 논쟁: 진짜 Emergence인가?

**주장 1 (진짜 emergence)**:
Scale에 따른 phase transition이 있으며, 작은 모델은 실제로 그 capability가 없다.

**반론 (metric artifact)**:
비선형적인 평가 metric(예: exact match accuracy)을 사용하면, 모델이 실제로는 점진적으로 나아지고 있어도 갑자기 뛰는 것처럼 보입니다. Continuous metric(예: log probability)으로 보면 점진적인 향상이 있다는 연구도 있습니다.

**현재 시각**: 둘 다 일부 맞습니다. 일부 능력은 진짜 창발적(emergent)이고, 일부는 metric 선택의 artifact입니다.

---

## 0.7 In-Context Learning vs Fine-Tuning

### In-Context Learning (ICL)

파라미터를 전혀 업데이트하지 않고, **프롬프트에 예시를 넣어주는 것만으로** 모델이 새로운 task를 수행하는 능력입니다.

```
Prompt:
  "Translate to French: 'Hello' → 'Bonjour'
   Translate to French: 'Cat' → 'Chat'
   Translate to French: 'Dog' → ?"
```

- **Zero-shot**: 예시 없이 instruction만으로 수행
- **Few-shot**: 2~10개 예시 포함

**ICL이 왜 작동하는가?**:
정확한 메커니즘은 아직 연구 중이지만, pretraining 중에 "맥락을 보고 패턴을 추출하는" 능력이 학습된다고 봅니다. 파라미터를 "암묵적 meta-learner"로 볼 수 있습니다.

---

### Fine-Tuning

파라미터를 직접 업데이트하여 특정 task/domain에 맞게 모델을 조정합니다.

```
θ* = argmin_θ L_task(θ; D_task)
```

**Full fine-tuning**: 모든 파라미터 업데이트 (비용 高)
**PEFT (LoRA 등)**: 일부 파라미터만 업데이트 (비용 低)

---

### 비교 정리

| 항목 | In-Context Learning | Fine-Tuning |
|------|--------------------|----|
| 파라미터 변경 | ❌ | ✅ |
| 필요 데이터 | 수~수십 개 예시 | 수백~수백만 개 |
| 일반화 | 유연하지만 불안정 | task에 특화, 안정적 |
| 비용 | Context token만큼만 | GPU + 시간 필요 |
| 언제 쓰나 | 빠른 프로토타이핑, API 접근 | 지속적 서비스, 도메인 특화 |

**결론**: ICL과 fine-tuning은 경쟁 관계가 아니라 보완 관계입니다. Production 환경에서는 fine-tuned model에 few-shot ICL을 함께 사용하는 경우도 많습니다.
