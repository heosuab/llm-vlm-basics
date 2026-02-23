# Mechanistic Interpretability (기계적 해석가능성)

## 핵심 질문

```
"LLM 내부에서 실제로 무슨 일이 일어나는가?"

기존 해석가능성: 입출력 관계 (Behavioral)
  → "모델이 성별 편향을 보인다"
기계적 해석가능성: 내부 메커니즘 (Mechanistic)
  → "어떤 circuit이 성별 편향을 만드는가?"

목표:
  - 알고리즘 이해: 모델이 어떻게 추론하는가
  - Safety: 위험한 내부 표현 탐지/제거
  - 개선: 알고리즘 이해로 모델 설계 개선
```

---

## Residual Stream & Circuit

### Residual Stream 관점
```
Transformer = 잔차 스트림에 정보를 읽고 쓰는 컴포넌트들

x₀ = embedding(token)
x₁ = x₀ + Attn₁(x₀)    ← Attention Layer 1이 정보 추가
x₂ = x₁ + FFN₁(x₁)     ← FFN Layer 1이 정보 추가
x₃ = x₂ + Attn₂(x₂)
...
logits = LM_head(x_final)

핵심: 모든 레이어가 같은 차원의 잔차 스트림에 기여
  → 레이어들은 서로 "통신"하는 게 아니라
     공유 스트림에 쓰고 읽음

의미: 초기 레이어의 정보가 후기 레이어에 그대로 전달 가능
    → 특정 정보가 어느 레이어에서 처리되는지 분석 가능
```

### Attention Head 구조 재해석
```
각 Attention Head:
  Q(x): "무엇을 찾는가?" (Query)
  K(x): "나는 무엇인가?" (Key)
  V(x): "내가 기여할 정보는?" (Value)
  O: "정보를 residual stream 어디에 쓸까?"

수식:
  head_output = softmax(QK^T/√d)·V
  contribution_to_residual = O · head_output

Head의 기능 분류:
  Attention pattern (QK 행렬): 어디를 볼지 결정
  Value output (VO 행렬): 무엇을 전달할지 결정

  두 요소 분리 분석 가능!
```

---

## 어텐션 헤드 종류

### Induction Heads (가장 중요)
```
기능: [A][B]...[A] → 다음 [B] 예측
  즉, 이전에 봤던 패턴 반복 탐지

메커니즘 (2-head circuit):
  Head 1 (Previous Token Head): 각 token이 이전 token에 attend
    → token t의 value = token t-1의 표현

  Head 2 (Induction Head): token t에서 Q=현재 token, K=이전 token
    → 현재 token과 같은 token이 과거에 나왔던 위치 찾기
    → 그 다음 토큰(predecessor's value)에 attend

In-context learning의 핵심 메커니즘!
  예: "Paris is in France. London is in ____"
  → "London is in" 패턴이 앞에 있으면 "England" 예측
```

### Name Mover Heads
```
IOI (Indirect Object Identification) 태스크:
  "John gave Mary the ball. He gave it to ____"
  → "Mary" 예측

Name Mover Heads:
  주어(John)와 간접 목적어(Mary)를 구분
  "gave to" 패턴을 인식하여 올바른 이름 선택

  발견: GPT-2에서 특정 head (9.9, 10.0 등)가 이 역할

Negative Name Mover:
  틀린 이름에 negative 기여 (억제)
  → 앙상블로 더 정확한 예측
```

### Positional Heads
```
특정 위치 패턴에 attend하는 head
  - Previous token head: 항상 이전 token에 attend
  - Beginning-of-sentence head: BOS 토큰에 attend
  - Current position head: diagonal attention pattern

분석 방법:
  Attention pattern 시각화 → 특정 패턴 발견
  Ablation: head 제거 후 성능 변화 측정
```

---

## Superposition & Features

### Linear Representation Hypothesis
```
모델의 표현:
  각 "feature" = 활성화 공간에서의 방향 벡터
  여러 feature가 동일 공간에 중첩 (superposition)

이유: d_model (예: 4096) < 실제 필요 feature 수 (수백만)
→ 모델이 많은 feature를 작은 공간에 압축

Superposition:
  feature_a: [1, 0, 0.5, ...]
  feature_b: [0.5, 1, 0, ...]
  → 거의 직교하지만 약간 겹침
  → 동시에 활성화 안 되면 (sparse) 문제없음
```

### Polysemanticity (다의성)
```
문제: 하나의 뉴런이 여러 의미 없어 보이는 개념에 반응
  예: GPT-2의 특정 뉴런이
      "뇌" "기차역" "마이클 조던"에 모두 활성화

원인: Superposition → 여러 feature가 같은 뉴런에 인코딩
  뉴런은 기저(basis) 역할 아님, feature가 비선형 방향

해석 어려움:
  "이 뉴런 = 이 개념" 분석 불가
  → Sparse Autoencoder (SAE) 필요
```

### Sparse Autoencoder (SAE)
```
목적: superposition을 풀어 monosemantic feature 추출

구조:
  encoder: x (d_model) → z (d_sae, d_sae >> d_model)
    z = ReLU(W_enc · x + b_enc)
  decoder: z → x_recon (d_model)
    x_recon = W_dec · z + b_dec

학습:
  L = ||x - x_recon||² + λ · ||z||₁
  재구성 오차 최소화 + L1 sparsity 강제

결과:
  z의 각 차원 = interpretable feature (거의)
  예: z_42 = "Python 코드" feature
      z_1024 = "부정적 감정" feature

Anthropic Claude 3에서 수백만 feature 발견 (Claude Monet, DNA 구조 등)
→ AI safety 연구의 핵심 도구
```

---

## In-Context Learning (ICL) 메커니즘

### ICL이 하는 일
```
Few-shot: [x1→y1][x2→y2]...[xn→?]
  → 모델이 패턴을 "학습" 없이 예측

메커니즘 가설들:
1. Induction Head 가설:
   → 패턴 반복 탐지 → (x1→y1) 패턴을 xn에 적용

2. Implicit Gradient Descent 가설:
   → Attention이 암묵적 gradient descent 수행
   → 각 forward pass = gradient step

3. Task Vector 가설:
   → 특정 레이어 activation에 "task"가 인코딩
   → "translate to French" task vector 존재
```

### ICL의 한계 이해
```
ICL은 언제 실패하나:
  1. Label flipping: 틀린 label 줘도 어느 정도 작동
     → 정확한 예시보다 형식(format)이 중요
  2. 분포 밖 task: 사전학습 분포 밖 예시는 효과 없음
  3. 긴 ICL: 예시 수 증가에 따른 성능 포화

실험 결과 (Min et al., 2022):
  - Label 반전: 성능 거의 유지
  - No examples: 상당한 성능
  → ICL = task format 인식 + label space 인식
     (정확한 매핑 학습은 아님)
```

---

## 지식 저장 위치

### MLP as Key-Value Memory
```
Geva et al. (2021): "Transformer Feed-Forward Layers are Key-Value Memories"

FFN: FFN(x) = W₂ · GELU(W₁ · x)
  W₁: d_model → d_ffn (key matrix)
  W₂: d_ffn → d_model (value matrix)

해석:
  W₁의 각 행 = key (특정 패턴에 반응)
  W₂의 각 열 = value (residual stream에 기여)

사실적 지식 저장:
  "Eiffel Tower는 ___에 있다" → 특정 key가 활성화
  → 해당 value가 "Paris" 방향으로 기여

증거:
  특정 지식 관련 factual recall에서
  중간 레이어 FFN 뉴런 ablation → 성능 저하
  초기 레이어: 문법적/구문적 패턴
  후기 레이어: 의미적/사실적 지식
```

### Locating Factual Associations (Rome et al.)
```
Causal Tracing: 어떤 component가 특정 사실에 기여하는지

방법:
  1. 정상 forward pass 실행, 모든 activation 저장
  2. 입력 부패 (corrupted run): 주제 토큰에 noise 추가
  3. 부패 run에서 하나씩 정상 activation으로 복원
  4. 복원 시 성능이 회복되는 지점 = 중요 component

발견:
  Subject 마지막 토큰의 중간 레이어 MLP가 결정적
  → "Eiffel Tower" 입력 시 특정 MLP가 "Paris" 인코딩

응용: ROME, MEMIT → 특정 사실만 편집 (fine-tuning 없이)
```

---

## Emergent Abilities

### 정의 및 논쟁
```
Emergent: 작은 모델에서는 없다가 특정 크기 이상에서 갑자기 나타나는 능력

예시 (Wei et al., 2022):
  - Multi-step arithmetic: 100B+ 모델에서 갑자기 가능
  - BIG-Bench tasks: 특정 임계점 후 성능 급등
  - Chain-of-thought reasoning: 100B 이상에서 효과적

논쟁 (Schaeffer et al., 2023):
  "Emergent"는 불연속적 측정 지표의 착시일 수 있음
  → 연속 지표(expected calibration error)로 보면 점진적
  → 이진 정확도(맞/틀)로 보면 갑작스러워 보임

실용적 의미:
  특정 능력을 얻으려면 특정 규모 이상이 필요할 수 있음
  Instruction following, complex reasoning → 대형 모델 유리
```

### Grokking
```
현상: 훈련 데이터 암기 후 갑자기 일반화 능력 획득

예: 97 modular arithmetic 학습
  - Early training: train acc 100%, val acc 50% (암기)
  - Late training: 갑자기 val acc 100% (이해)

해석:
  1. 처음에 shortcut (암기) 해 사용
  2. 정규화 압력으로 점점 더 일반적 알고리즘 선호
  3. 임계점에서 알고리즘 해로 "전환"

의미: 충분히 오래 학습하면 암기→이해 전환 가능
  → 과적합으로 보이는 단계를 지속해야 할 수 있음
```

---

## 분석 도구 & 방법론

### TransformerLens
```python
import transformer_lens

# 모델 로드 (내부 activation 접근 가능)
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

# 특정 activation에 hook 걸기
logits, activations = model.run_with_cache("The Eiffel Tower is in")

# Layer 특정 attention pattern
attn_pattern = activations["blocks.5.attn.hook_attn"]
# shape: (batch, heads, seq, seq)

# 특정 head의 value output
value_out = activations["blocks.5.attn.hook_z"]

# Patching: 특정 activation을 다른 run에서 교체
def patch_hook(value, hook):
    value[:, :, 5:7] = clean_value[:, :, 5:7]  # position 5-7 교체
    return value

patched_logits = model.run_with_hooks(
    corrupted_tokens,
    fwd_hooks=[("blocks.5.attn.hook_z", patch_hook)]
)
```

### Probing Classifiers
```python
# Linear Probe: activation에 특정 개념이 있는지 분석
from sklearn.linear_model import LogisticRegression

# 특정 레이어 activation 수집
activations = []
labels = []
for text, label in dataset:
    act = model.get_activation(text, layer=10)
    activations.append(act)
    labels.append(label)

# 선형 분류기 학습
probe = LogisticRegression()
probe.fit(activations_train, labels_train)
accuracy = probe.score(activations_test, labels_test)

# 높은 accuracy → 해당 레이어가 해당 개념 인코딩
# 낮은 accuracy → 해당 개념은 다른 레이어나 비선형적으로 인코딩
```

### Activation Patching / Causal Tracing
```python
# 깨끗한 입력과 부패 입력 비교
clean_tokens = tokenize("The Eiffel Tower is in")
corrupt_tokens = tokenize("The ___ ___ ___ is in")  # 주제 부패

# 깨끗한 run의 activation 저장
_, clean_cache = model.run_with_cache(clean_tokens)

# 부패 run + 특정 위치 복원
for layer in range(model.n_layers):
    for pos in range(seq_len):
        def hook(value, hook, l=layer, p=pos):
            value[:, p:p+1, :] = clean_cache[f"blocks.{l}.hook_resid_post"][:, p:p+1, :]
            return value

        patched_logits = model.run_with_hooks(
            corrupt_tokens,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook)]
        )
        # 성능 회복 = 해당 (layer, pos)가 중요
```

---

## Further Questions

**Q1. Induction Head가 In-Context Learning과 어떻게 관련되나?**
```
Induction Head: [A][B]...[A] → [B] 예측 (패턴 반복)

ICL 연결:
  [prompt][answer]...[similar_prompt] → [answer] 예측
  = Induction Head의 일반화된 버전

증거:
  - ICL 능력이 나타나는 시점 = Induction Head 형성 시점
  - Induction Head ablation → ICL 성능 급락
  - 두 레이어 모델에서도 Induction Head 관찰 (Circuit 단순)
```

**Q2. Sparse Autoencoder로 무엇을 할 수 있나?**
```
1. Feature 발견: polysemantic 뉴런을 monosemantic feature로 분해
2. Safety: 위험한 feature (예: "자살 방법") 탐지 및 조작
3. 지식 편집: 특정 feature 억제로 정보 제거
4. 이해: 모델이 어떤 concept을 표현하는지 분석

한계:
  - d_sae가 매우 커야 함 (수백만)
  - 모든 feature가 monosemantic인 보장 없음
  - 계산 비용 큼
  - 발견한 feature의 실제 기능 검증 어려움
```

**Q3. "LLM은 추론을 하는가"를 어떻게 검증하나?**
```
1. Process 분석: 올바른 답에 올바른 중간 단계 사용?
   Causal scrubbing: 중간 단계 activation 교란 → 성능 변화?

2. Counterfactual: 주어진 CoT와 다른 답 유도 가능?
   CoT가 faithful하면 CoT 변경 → 답 변경 (대부분 안 됨)

3. Faithfulness vs Accuracy:
   Lanham et al.: CoT에서 중간 단계 삭제해도 최종 답 유지
   → CoT가 실제 추론이 아닐 수 있음

결론: LLM의 "사고"는 인간 추론과 다르며
      Chain-of-Thought의 기능적 역할은 아직 연구 중
```

**Q4. 지식 편집(Knowledge Editing)이란?**
```
목적: "Eiffel Tower는 어디에 있나?" → "Rome" 대신 "Paris" 예답
  특정 사실만 바꾸고 나머지는 유지

방법:
  ROME (2022): Rank-1 Model Editing
    - Causal tracing으로 사실 저장 레이어 찾기
    - 해당 MLP weight를 closed-form으로 수정
    - 단점: 하나씩 편집, 다수 편집 시 성능 저하

  MEMIT (2023): Mass-Editing Memory in Transformers
    - 여러 사실 동시 편집
    - 여러 레이어에 분산

  Fine-tuning 기반:
    - LoRA로 특정 사실 파인튜닝
    - 간단하지만 catastrophic forgetting 위험

한계: 진정한 세계 모델 업데이트인지 vs 표면적 패턴 변경인지 불확실
```
