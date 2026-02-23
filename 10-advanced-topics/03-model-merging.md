# 모델 병합 (Model Merging)

## 개념과 동기

```
목적: 여러 파인튜닝 모델을 하나로 합쳐 능력 결합
      재학습 없이 (zero training cost)

예시:
  Chat 모델 + Code 모델 → 코딩 잘하는 채팅 모델
  여러 도메인 전문가 → 멀티도메인 모델

전제: 같은 base 모델에서 파인튜닝된 모델들
  (base 다르면 병합 효과 불확실)

실용성:
  - A100 8장 학습 vs 병합: 병합이 100× 이상 빠름
  - 오픈소스 생태계에서 활발 (mergekit 등)
```

---

## Task Vector

### 개념 (Ilharco et al., 2022)
```
Task Vector τ = θ_finetuned - θ_base

해석: 파인튜닝이 base model에 더한 "방향"
  Base model + τ_code = Code 모델
  Base model + τ_chat = Chat 모델

덧셈:
  θ_new = θ_base + τ_code + τ_chat
  → 두 능력 모두 보유

스케일링:
  θ_new = θ_base + λ·τ  (λ로 능력 강도 조절)
  λ > 1: 능력 강화 (overshoot 위험)
  λ < 1: 능력 약화

빼기:
  θ_new = θ_base - τ_harmful  (특정 능력 제거)
  → "Detoxification" 가능

놀라운 점:
  벡터 덧셈만으로 독립 학습된 능력 결합
  → 선형 표현 공간 지지 증거
```

```python
import torch
from safetensors.torch import load_file, save_file

def load_model_weights(path: str) -> dict[str, torch.Tensor]:
    """safetensors 또는 pytorch bin 로드"""
    if path.endswith(".safetensors"):
        return load_file(path)
    return torch.load(path, map_location="cpu")

def compute_task_vector(base_path: str, finetuned_path: str) -> dict[str, torch.Tensor]:
    """τ = θ_finetuned - θ_base"""
    base = load_model_weights(base_path)
    finetuned = load_model_weights(finetuned_path)

    task_vector = {}
    for key in base:
        if key in finetuned:
            task_vector[key] = finetuned[key].float() - base[key].float()
    return task_vector

def apply_task_vectors(base_path: str, task_vectors: list[dict],
                       lambdas: list[float]) -> dict[str, torch.Tensor]:
    """Base model + weighted sum of task vectors"""
    base = load_model_weights(base_path)
    merged = {k: v.float().clone() for k, v in base.items()}

    for tau, lam in zip(task_vectors, lambdas):
        for key in merged:
            if key in tau:
                merged[key] += lam * tau[key]

    return merged

# 사용 예시
tau_code = compute_task_vector("llama3-base", "llama3-code")
tau_chat = compute_task_vector("llama3-base", "llama3-chat")
tau_math = compute_task_vector("llama3-base", "llama3-math")

merged = apply_task_vectors(
    base_path="llama3-base",
    task_vectors=[tau_code, tau_chat, tau_math],
    lambdas=[0.6, 0.8, 0.5]  # 각 능력의 강도 조절
)
```

---

## Spherical Linear Interpolation (SLERP)

### 선형 보간 vs SLERP
```
선형 보간 (LERP):
  θ_new = (1-t) · θ_A + t · θ_B
  문제: 중간값에서 norm 감소 (벡터가 안으로 당겨짐)
  → 능력 감소 현상

SLERP (구면 선형 보간):
  구 표면 위를 따라 보간 (norm 유지)
  θ_new = sin((1-t)Ω)/sinΩ · θ_A + sin(tΩ)/sinΩ · θ_B
  Ω = arccos(θ_A · θ_B / (||θ_A|| · ||θ_B||))

장점:
  - norm이 일정하게 유지
  - 보간 경로가 더 "자연스러움"
  - 실제로 LERP보다 좋은 경우 많음
```

### 구현
```python
import torch
import numpy as np

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """Spherical Linear Interpolation"""
    # 정규화
    v0_norm = v0 / torch.norm(v0, dim=-1, keepdim=True)
    v1_norm = v1 / torch.norm(v1, dim=-1, keepdim=True)

    # cos(Ω) 계산
    dot = (v0_norm * v1_norm).sum(dim=-1, keepdim=True)
    dot = dot.clamp(-1, 1)

    # 거의 같은 방향이면 LERP 사용 (수치 안정성)
    if (dot.abs() > DOT_THRESHOLD).all():
        return (1 - t) * v0 + t * v1

    # SLERP
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    s0 = torch.sin((1 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta
    return s0 * v0 + s1 * v1

def slerp_models(model_a: dict, model_b: dict, t: float = 0.5) -> dict:
    """두 모델의 SLERP 병합"""
    merged_state = {}
    for key in model_a.keys():
        if model_a[key].dtype in (torch.float32, torch.float16, torch.bfloat16):
            a = model_a[key].float()
            b = model_b[key].float()
            merged_state[key] = slerp(t, a, b)
        else:
            merged_state[key] = model_a[key]  # non-float layers는 그냥 복사
    return merged_state
```

---

## TIES-Merging (2023)

### 문제: 파라미터 간섭 (Parameter Interference)
```
단순 task vector 덧셈의 문제:
  τ_A와 τ_B가 같은 파라미터에 반대 방향 기여
  → 서로 상쇄 → 성능 저하

예:
  τ_code[weight_42] = +0.1
  τ_chat[weight_42] = -0.08
  합산: +0.02 → 두 모델의 능력 모두 저하
```

### TIES 3단계
```
1단계: TRIM (트리밍)
  각 task vector에서 작은 값 제거 (0으로 설정)
  → 노이즈 감소, 중요 변화만 유지
  방법: |τ_i| 상위 k% (예: 상위 20%)만 유지

2단계: ELECT (선출)
  파라미터별로 "지배적 방향" 결정
  → 모든 task vector에서 해당 파라미터의 부호 비교
  → 다수결 (절대값 합이 큰 방향 선택)

  γ_p = sign(Σ_i τ_i[p])  (단순)
  또는 γ_p = sign(Σ_i |τ_i[p]| · sign(τ_i[p]))  (가중 다수결)

3단계: DISJOINT MERGE (비겹침 병합)
  지배적 방향과 같은 부호의 task vector만 평균
  반대 방향은 제외

  θ_merged = θ_base + Σ_i [τ_i[p] · 1(sign(τ_i[p]) == γ_p)] / count

결과: 간섭 감소, 성능 향상
```

```python
def ties_merge(base_weights: dict, task_vectors: list[dict],
               density: float = 0.2, weights: list[float] = None) -> dict:
    """
    TIES-Merging: Trim, Elect, Disjoint Merge
    density: 유지할 파라미터 비율 (상위 density%)
    """
    if weights is None:
        weights = [1.0 / len(task_vectors)] * len(task_vectors)

    merged = {}

    for key in base_weights:
        base = base_weights[key].float()
        vectors = [tv[key].float() for tv in task_vectors if key in tv]

        if not vectors:
            merged[key] = base
            continue

        # 1. TRIM: 작은 값 제거
        trimmed = []
        for v in vectors:
            flat = v.abs().flatten()
            threshold = flat.kthvalue(int(len(flat) * (1 - density))).values.item()
            trimmed.append(v * (v.abs() >= threshold).float())

        # 2. ELECT: 지배적 방향 결정 (가중 다수결)
        weighted_sum = sum(w * t for w, t in zip(weights, trimmed))
        dominant_sign = torch.sign(weighted_sum)

        # 3. DISJOINT MERGE: 같은 방향만 평균
        aligned = []
        for t, w in zip(trimmed, weights):
            # 지배적 방향과 같은 부호의 요소만 유지
            aligned_t = t * (torch.sign(t) == dominant_sign).float()
            aligned.append(w * aligned_t)

        # 평균 (유효 기여 수로 나눔)
        count = sum((torch.sign(t) == dominant_sign).float() for t in trimmed)
        count = count.clamp(min=1)  # 0 나눗셈 방지

        delta = sum(aligned) / count
        merged[key] = base + delta

    return merged
```

---

## DARE (Drop and REscale, 2023)

### 아이디어
```
많은 파인튜닝 델타(τ)는 실제로 중요하지 않음
→ 랜덤하게 제거해도 성능 유지

근거: 뉴럴 네트워크의 희소성
  대부분의 파라미터 변화는 노이즈에 가까움

방법:
  1. 랜덤 마스킹: 확률 p로 τ의 요소를 0으로 설정
  2. 재스케일링: 1/(1-p)로 나머지 값 보상

  DARE(τ, p) = mask(τ, p) / (1-p)

  이후 TIES와 결합: DARE-TIES

효과:
  - 각 model의 독립성 증가 (간섭 감소)
  - 더 많은 모델 병합 가능 (노이즈 감소)
  - p = 0.9: 90% 제거해도 성능 유지 (놀라운 희소성)
```

```python
def dare_sparsify(task_vector: dict, p: float = 0.9,
                  seed: int = 42) -> dict:
    """DARE: Drop And REscale task vector"""
    torch.manual_seed(seed)
    sparsified = {}

    for key, v in task_vector.items():
        # 랜덤 마스크 생성 (확률 p로 드롭)
        mask = torch.bernoulli(torch.ones_like(v) * (1 - p))
        # 드롭 후 재스케일링 (기대값 유지)
        sparsified[key] = v * mask / (1 - p)

    return sparsified

def dare_ties_merge(base_weights: dict, task_vectors: list[dict],
                    drop_rate: float = 0.9, density: float = 0.2) -> dict:
    """DARE + TIES 결합"""
    # Step 1: DARE로 각 task vector 희소화
    sparsified = [dare_sparsify(tv, p=drop_rate) for tv in task_vectors]
    # Step 2: TIES로 병합
    return ties_merge(base_weights, sparsified, density=density)
```

---

## Frankenmerging (레이어 스와핑)

### 개념
```
서로 다른 모델의 레이어를 직접 교환/조합

예시 (Goliath-120B):
  LLaMA-2-70B (Layer 0-39) + LLaMA-2-70B fine-tuned (Layer 40-79)
  → 120B "Franken" 모델 (실제 작동!)

왜 작동하나?:
  같은 base 모델의 레이어 표현 공간이 호환
  → 레이어 경계에서 representation이 이어짐

위험:
  다른 base 모델 간 → 표현 공간 불일치 → 실패
  레이어 순서 중요 (초기/중기/후기 레이어 역할 다름)
```

### Passthrough 레이어
```
특정 레이어를 여러 번 반복:
  Layer: 0,1,...,20, 15,16,17,18, 21,...,40
  → 중간 레이어를 두 번 통과

효과:
  특정 기능 강화 (반복 레이어가 그 기능 담당 시)
  전체 레이어 수 증가 (파라미터 증가)

실험적:
  정확한 이론적 이해 없음
  Empirical로 작동하는 경우 발견
```

---

## Evolutionary Merging (2024)

### 개념
```
병합 하이퍼파라미터를 진화 알고리즘으로 최적화

탐색 공간:
  - 레이어별 혼합 비율
  - 어떤 레이어를 스왑할지
  - Task vector 스케일

적합도 함수:
  - 특정 벤치마크 성능
  - Perplexity
  - Human preference (expensive)

장점:
  - 수동 설정보다 더 나은 병합 발견 가능
  - GPU 사용 안 함 (평가만)

단점:
  - 탐색 공간이 넓음
  - 많은 평가 필요 (수백~수천)
```

---

## AdaMerging (2024)

```
Test-Time Adaptation for Model Merging

문제: 고정된 병합 계수 → 입력/태스크에 따라 최적값 다름

방법:
  1. 각 레이어/태스크에 학습 가능한 스케일 계수 도입
  2. Test time에 소량 unlabeled 데이터로 최적화
     → Entropy minimization objective

  θ_merged = θ_base + Σ_i α_i · τ_i

  α_i를 task entropy 최소화로 학습:
  min_α H(f(x; θ_merged(α)))

장점:
  태스크/입력에 맞는 동적 병합
  추가 데이터 없이 (unlabeled) 적응
  Task vector보다 일관되게 좋음

단점:
  테스트 시 추가 최적화 비용
  일부 환경에서 overfitting 위험
```

---

## Model Stock (2024)

```
Projection-based Merging (NAVER)

핵심 아이디어:
  파인튜닝된 모델들이 "하이퍼스피어" 표면에 존재
  최적 병합점 = 이 하이퍼스피어 위의 중심

방법:
  1. 각 모델의 task vector를 정규화
  2. task vector들의 평균 방향 계산
  3. base 모델에서 평균 방향으로 이동

수식:
  τ̄ = normalize(Σ_i τ_i / ||τ_i||)
  θ_merged = θ_base + α · τ̄
  α: 이동 크기 (평균 task vector norm 기반)

효과:
  단순 평균보다 일관된 성능
  레이어별로 최적 이동 거리 자동 계산
```

---

## 실전: mergekit 사용

### 설치 및 기본 사용
```bash
pip install mergekit

# 설정 파일 기반 병합
mergekit-yaml merge_config.yaml merged_model/ \
  --cuda --low-cpu-memory
```

### 설정 파일 예시
```yaml
# SLERP 병합
merge_method: slerp
base_model: mistralai/Mistral-7B-v0.1
models:
  - model: WizardLM/WizardLM-7B-V1.0  # 지시 따르기
    parameters:
      t: 0.5
  - model: teknium/OpenHermes-2.5-Mistral-7B  # 코딩+지시
    parameters:
      t: 0.5
dtype: bfloat16

---
# TIES 병합
merge_method: ties
base_model: meta-llama/Meta-Llama-3-8B
models:
  - model: meta-llama/Meta-Llama-3-8B-Instruct
    parameters:
      density: 0.5   # DARE: 50% 파라미터 유지
      weight: 1.0
  - model: deepseek-ai/deepseek-coder-7b-instruct-v1.5
    parameters:
      density: 0.5
      weight: 0.8
parameters:
  normalize: true  # weight 합계를 1로 정규화
dtype: bfloat16

---
# 레이어 교환 (Frankenmerge)
merge_method: passthrough
slices:
  - sources:
    - model: mistralai/Mistral-7B-v0.1
      layer_range: [0, 16]
  - sources:
    - model: teknium/OpenHermes-2.5-Mistral-7B
      layer_range: [16, 32]
dtype: bfloat16
```

---

## Model Soup (앙상블 없이)

```
Weight averaging (단순 평균):
  N개 체크포인트의 weight 평균
  → 일반화 성능 향상 (sharp minima → flat minima)

  θ_soup = (1/N) Σ_i θ_i

Greedy Soup:
  하나씩 추가하며 검증 성능 확인
  성능 향상 시만 추가

효과:
  - 단일 체크포인트보다 일반적으로 성능 향상
  - 검증 없이 사용 가능 (uniform soup)
  - CLIP, ViT에서 최초 증명 (Wortsman et al., 2022)

LLM 파인튜닝:
  다른 시드 or 다른 LR로 학습한 여러 체크포인트 평균
  → 더 안정적이고 좋은 성능
```

---

## 병합 전략 비교 및 선택

```
태스크별 권장:

같은 능력 강화 (예: coding 두 모델):
  → SLERP (t=0.5) 또는 Model Soup

다른 능력 결합 (예: code + chat):
  → Task Vector + 스케일 조정 또는 TIES

3개 이상 모델:
  → TIES-DARE (간섭 감소)
  → 각 모델 density=0.5~0.7 권장

성능 최적화 필요:
  → Evolutionary Merging + 벤치마크 평가
  → AdaMerging (테스트 타임 최적화)

빠른 실험:
  → Task Vector 덧셈 (가장 단순)
  → mergekit SLERP (툴 사용)
```

```python
def benchmark_merge_configs(base_path: str, model_paths: list[str],
                              eval_fn, n_configs: int = 20) -> dict:
    """여러 병합 설정을 랜덤 탐색하여 최적 찾기"""
    import random

    best_config = None
    best_score = -float('inf')
    results = []

    task_vectors = [compute_task_vector(base_path, mp) for mp in model_paths]

    for _ in range(n_configs):
        # 랜덤 lambda 값 탐색
        lambdas = [random.uniform(0.1, 1.0) for _ in model_paths]

        # 병합
        merged = apply_task_vectors(base_path, task_vectors, lambdas)

        # 평가
        score = eval_fn(merged)
        results.append({"lambdas": lambdas, "score": score})

        if score > best_score:
            best_score = score
            best_config = lambdas

    return {
        "best_lambdas": best_config,
        "best_score": best_score,
        "all_results": sorted(results, key=lambda x: -x["score"])
    }
```

---

## 병합의 한계

```
실패 케이스:
  1. Base 모델이 다름: 표현 공간 불호환
  2. 너무 다른 파인튜닝: 파라미터 간섭 심함
  3. 크기 차이: 7B + 13B 직접 병합 불가

한계:
  이론적 이해 부족 (경험적 방법)
  최적 병합 방법 모델마다 다름
  벤치마크에서 잘 나와도 실제 사용에서 다를 수 있음

적합한 상황:
  같은 base 모델 + 유사한 파인튜닝
  빠른 실험 (학습 없이)
  리소스 제한 환경
```

---

## Further Questions

**Q1. Task Vector 덧셈이 왜 작동하나?**
```
선형 표현 가설 (Linear Representation Hypothesis):
  LLM의 개념이 선형 방향으로 인코딩
  → 벡터 연산이 의미 있음

경험적 증거:
  "왕 - 남자 + 여자 ≈ 여왕" (Word2Vec)
  "프랑스 + 수도 ≈ 파리 방향" (representation)
  Task vector 덧셈 → 능력 결합

한계:
  완전히 이해되지 않음
  비선형 상호작용 무시
  큰 파인튜닝 차이에서는 실패
```

**Q2. TIES가 단순 Task Vector 평균보다 나은 이유는?**
```
단순 평균의 문제:
  τ_A[p] = +0.5, τ_B[p] = -0.4 → 평균 +0.05
  → 두 모델 모두의 의도가 소거

TIES:
  1. Trim: 노이즈 제거 (작은 값 0으로)
  2. Elect: 다수결로 방향 결정 (+인지 -인지)
  3. Merge: 같은 방향만 평균

결과: 간섭 감소 → 각 모델의 강점 유지
실험: 7개 태스크에서 TIES가 단순 평균 대비 2-5% 향상
```

**Q3. 모델 병합 vs 멀티태스크 파인튜닝 차이는?**
```
멀티태스크 파인튜닝:
  여러 태스크 데이터를 동시에 학습
  → 망각(Catastrophic forgetting) 위험
  → 학습 비용 큼
  → 최적 데이터 혼합 비율 탐색 필요

모델 병합:
  각 태스크를 독립 학습 후 병합
  → 학습 비용 = 각 모델 학습 비용 (병렬 가능)
  → 망각 없음
  → 추후 새 모델 추가 가능

언제 무엇을 쓰나?:
  데이터 있고 리소스 충분: 멀티태스크
  기존 모델 활용, 빠른 실험: 병합
  리소스 없음: 병합 (상업용 모델 병합도 가능)
```

**Q4. DARE에서 90% drop해도 성능이 유지되는 이유는?**
> Neural network의 "lottery ticket" 특성과 유사. Task vector의 대부분이 "노이즈" (실제 태스크 능력에 불필요한 파라미터 변화). 핵심 10%가 태스크 능력의 대부분을 담당 (희소성). Drop후 1/(1-p) 재스케일링으로 기대값 보존. 간섭 감소 효과: 희소한 벡터들이 덜 겹침.

**Q5. Frankenmerging에서 어떤 레이어를 어디에 배치해야 하나?**
> 레이어 역할 특성: 초기 레이어 (0~10): 기본 문법/구문 처리. 중간 레이어 (10~25): 의미/지식 처리. 후기 레이어 (25+): 태스크별 특화/출력 결정. 일반적 전략: Base 모델의 초기/중기 레이어 + Fine-tuned 모델의 후기 레이어. 태스크 특화 기능은 후기 레이어에 많이 저장.
