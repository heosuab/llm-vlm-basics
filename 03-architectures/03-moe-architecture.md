# MoE (Mixture of Experts) 아키텍처

## 핵심 아이디어

```
Dense 모델의 한계:
  모든 입력에 동일한 FFN을 적용
  파라미터 수 ↑ → 연산량(FLOP) 비례 증가
  → 크게 만들수록 추론 비용이 선형 증가

MoE의 혁신:
  여러 "전문가(Expert)" FFN 중 입력에 맞는 것만 선택
  파라미터 수(용량) ↑ + 연산량(FLOP) 일정
  → 파라미터 수 ≠ 연산량 분리

비교:
  Dense 모델: active params = total params
  MoE 모델: active params << total params
  예: Mixtral-8x7B
    Total: 47B params
    Active: ~13B params (top-2/8 선택)
    → 70B Dense 수준 성능, 13B 수준 추론 속도
```

---

## MoE 수식과 구현

### Router 수식

```
입력 토큰 x에 대해:

1. Router score 계산:
   s(x) = x · W_g  [d_model → n_experts]

2. TopK 선택:
   TopK_indices = argsort(s(x))[-K:]
   G(x) = Softmax(s(x)[TopK_indices])

3. MoE 출력:
   h(x) = Σₖ∈TopK G(x)ₖ · FFNₖ(x)

   G(x)ₖ = exp(sₖ) / Σⱼ∈TopK exp(sⱼ)

TopK 이외 Expert는 -∞로 마스킹 → Softmax 후 0
```

### PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ffn, n_experts, top_k, dropout=0.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        # Router (Gate)
        self.gate = nn.Linear(d_model, n_experts, bias=False)

        # Expert FFNs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.SiLU(),
                nn.Linear(d_ffn, d_model),
            ) for _ in range(n_experts)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # [B*T, D]

        # Router scores
        router_logits = self.gate(x_flat)  # [B*T, n_experts]

        # TopK 선택
        topk_logits, topk_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        # 선택된 expert만 Softmax
        topk_weights = F.softmax(topk_logits, dim=-1)  # [B*T, K]

        # Expert 실행 및 가중 합산
        output = torch.zeros_like(x_flat)

        # 각 expert별로 처리 (효율적 구현)
        for expert_idx in range(self.n_experts):
            # 이 expert를 선택한 토큰 찾기
            mask = (topk_indices == expert_idx)  # [B*T, K]
            expert_mask = mask.any(dim=-1)         # [B*T]

            if expert_mask.sum() == 0:
                continue

            expert_input = x_flat[expert_mask]
            expert_output = self.experts[expert_idx](expert_input)

            # 가중치 적용
            weights = topk_weights[mask].unsqueeze(-1)  # 해당 토큰들의 가중치
            output[expert_mask] += (weights * expert_output).sum(dim=...)

        return output.view(B, T, D), router_logits
```

### SwiGLU Expert (LLaMA 스타일)

```python
class SwiGLUExpert(nn.Module):
    """
    DeepSeek, Mixtral에서 사용하는 Expert FFN 형식
    """
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)   # gate
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)   # down
        self.w3 = nn.Linear(d_model, d_ffn, bias=False)   # up

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

---

## 주요 MoE 모델

### Switch Transformer (Google, 2021)

```
최초 대규모 Transformer MoE:
  Top-1 routing (각 토큰이 1개 Expert만 선택)
  Capacity Factor: Expert 당 최대 토큰 수 제한

  capacity = (tokens_per_batch / n_experts) × capacity_factor

  Overflow: capacity 초과 토큰은 다음 레이어로 그냥 통과

1.6T 파라미터, Top-1 routing
학습: T5 대비 7× 빠른 속도 달성
```

### GLaM (Google, 2021)

```
1.2T parameters, 64 experts, Top-2 routing
GPT-3 대비 0.1× FLOP으로 더 좋은 성능
```

### Mixtral-8x7B (Mistral AI, 2024)

```
구조:
  8개 Expert, Top-2 선택
  각 Expert: FFN (d_model=4096, d_ffn=14336)
  총 파라미터: 47B (attention + 8 FFN)
  Active 파라미터: ~13B (attention + 2 FFN)

Mixtral-8x22B (2024):
  8 Experts, 141B total, ~39B active
  성능: GPT-4 수준

사용 방법:
  from transformers import AutoModelForCausalLM
  model = AutoModelForCausalLM.from_pretrained(
      "mistralai/Mixtral-8x7B-v0.1",
      torch_dtype=torch.bfloat16,
      device_map="auto"
  )
```

### DeepSeek-MoE (2024)

```
두 가지 혁신:

1. Fine-grained Expert:
   Mixtral: 8 experts, top-2 (각 expert = 14B FFN)
   DeepSeek: 64 experts, top-6 (각 expert = 1.75B FFN)
   → 더 작고 전문화된 Expert가 더 효과적
   → 선택의 유연성 향상

2. Shared Expert:
   모든 토큰이 항상 사용하는 1-2개 공통 Expert
   → 공통 지식은 Shared Expert에 집중
   → Routed Expert는 도메인 특화 지식만 학습
   → 지식 중복 최소화, 전문화 극대화

수식:
  h(x) = Expert_shared(x) + Σₖ∈TopK G(x)ₖ · Expert_routedₖ(x)
```

### DeepSeek-V3 (2024)

```
671B 총 파라미터, 37B active parameters (추론 시)

Expert 구조:
  256개 Routed Experts
  1개 Shared Expert
  Top-8 routing

추가 혁신:
  Multi-Token Prediction (MTP):
    다음 1개 토큰 외에 여러 토큰 동시 예측
    보조 학습 신호 → 더 빠른 수렴

  Loss-free Load Balancing:
    Auxiliary loss 대신 Expert bias로 균형 조정
    학습 목표 함수 깔끔하게 유지

FP8 학습:
  최초로 대규모 FP8 mixed precision 훈련 성공
```

### Grok-1 (xAI, 2024)

```
314B 총 파라미터
8 experts, Top-2 routing
오픈소스 공개 (weights)
```

### Phi-3.5-MoE (Microsoft, 2024)

```
41.9B total, 6.6B active
16 experts, Top-2
데이터 품질 중심 학습 (합성 데이터)
소형 디바이스 최적화
```

### Qwen2-MoE (Qwen 팀, 2024)

```
57B total, 14.3B active
64 routed + 4 shared experts, Top-8
```

### OLMoE (Allen AI, 2024)

```
7B total, 1.3B active
8 experts, Top-1
완전 오픈소스 (데이터, 학습 코드 포함)
```

---

## 라우팅 전략

### Token Choice (표준)

```
각 토큰이 스스로 Expert 선택 (Top-K gate score)

문제: Load Imbalance
  일부 Expert에 토큰 집중 → 나머지 Expert 낭비
  GPU 계산 불균형 → 병목

완화 방법:
  1. Auxiliary Loss (다음 섹션)
  2. Expert Capacity Factor 설정
  3. Noise 추가 (탐색성 증가)
```

### Expert Choice [Zhou et al., 2022]

```
각 Expert가 처리할 토큰 선택 (Top-K 토큰)

  각 Expert가 "내가 가장 잘 처리할 수 있는 K개 토큰 선택"

장점:
  완벽한 Load Balance (각 Expert가 고정 K개 처리)
  Expert 전문화 촉진

단점:
  각 토큰이 반드시 Expert를 갖지 않을 수 있음
  가변적인 처리 보장 없음 → 실용적 어려움
  Auto-regressive 생성에서 비효율
```

### Expert Capacity (Capacity Factor)

```
각 Expert가 처리할 최대 토큰 수 = (tokens/n_experts) × C

C (Capacity Factor):
  C=1.0: 균등 분배 가정, overflow 토큰 버림
  C=1.25: 25% 여유 (더 많은 토큰 수용)
  C=2.0: 2배 여유 (높은 메모리, 낮은 overflow)

Overflow 처리:
  방법 1: 토큰을 다음 Expert로 넘기기
  방법 2: 잔차 연결 (Expert 건너뜀)
  방법 3: 가장 점수 낮은 토큰 drop
```

---

## Load Balancing

### Auxiliary Loss

```
Expert들의 부하가 균등하도록 추가 loss:

fᵢ = (Expert_i를 선택한 토큰 비율) = Σ_t 1[i ∈ TopK(t)] / T
pᵢ = (Expert_i 평균 router 점수) = (1/T) Σ_t softmax(s_t)[i]

L_aux = α · N · Σᵢ₌₁ᴺ fᵢ · pᵢ

  최솟값: 균등 분배 (fᵢ = 1/N, pᵢ = 1/N)
  α: 작은 값 사용 (0.001~0.01)

문제:
  Auxiliary loss가 기본 성능 저하 가능
  α 튜닝 어려움
```

### DeepSeek-V3의 혁신: Loss-free Load Balancing

```
Auxiliary loss 없이 bias term으로 균형:

수정된 라우팅:
  s'ᵢ(x) = s(x)ᵢ + bᵢ  (bias 추가)
  TopK: 수정된 s'로 선택

하지만 실제 MoE output은 원래 s로 계산:
  h(x) = Σₖ Softmax(s(x)[topk_from_s'])ₖ · Expertₖ(x)

Bias 업데이트 (배치 완료 후):
  if expert_i_load > target: bᵢ -= γ  (덜 선택되도록)
  if expert_i_load < target: bᵢ += γ  (더 선택되도록)

장점:
  학습 목표 함수에 aux loss 없음 → 깔끔한 학습
  동적 균형 조정 (adaptive)
  성능 저하 없이 균형 달성
```

---

## Expert Parallelism (EP)

```
분산 학습/추론에서 Expert를 여러 GPU에 분산:

  GPU0: Expert 0, Expert 1
  GPU1: Expert 2, Expert 3
  ...
  GPUₙ: Expert 2K-2, Expert 2K-1

All-to-All 통신:
  토큰들이 선택된 Expert가 있는 GPU로 이동
  Expert 처리 후 원래 GPU로 복귀

  GPU_token → 통신 → GPU_expert → 처리 → 통신 → GPU_token

통신 오버헤드:
  각 forward pass에 2번 All-to-All 통신
  네트워크 대역폭 병목 가능

최적화:
  DeepSeek: 통신을 compute와 overlap
  Node-limited routing: 각 노드 내 expert만 선택
    → Node 간 통신 없음 (NVLink만 사용)
```

---

## MoE 학습 과제

### Expert Collapse

```
일부 Expert만 계속 선택됨 → 나머지 Expert 학습 안 됨
초기화 단계에서 발생하기 쉬움

해결:
  1. Auxiliary Loss (부하 균형 강제)
  2. Expert 초기화 다양성 (다른 random seed)
  3. Z-loss: router logit의 크기를 제한
     L_z = β · (1/n · Σ log Σⱼ exp(sⱼ))²
  4. 학습 초기에 높은 aux_loss_weight 사용
```

### Communication Overhead

```
분산 학습 시 All-to-All 통신 필요
  토큰 → Expert GPU 이동 → 처리 → 원래 GPU

병목 발생:
  Inter-node 통신 (InfiniBand 100Gbps)
  배치 크기 크면 통신량 증가

해결:
  1. Top-1 routing (통신 감소)
  2. Node-local Expert (노드 간 통신 제거)
  3. Expert Parallelism 최소화
  4. 통신과 계산 overlap
```

### 메모리 vs 속도 트레이드오프

```
MoE: 메모리에 모든 Expert 로드 필요
  Mixtral-8x7B: 47B params → ~90GB (FP16)
  BUT active params = 13B → FP ops는 13B 수준

vs Dense 13B:
  메모리: ~26GB (FP16)
  FP ops: 13B (동일)

→ MoE: 더 많은 메모리 필요, 같은 추론 속도
  (메모리 있다면) Dense 13B 속도로 47B 성능

서빙 최적화:
  Expert offloading: 사용 안 되는 Expert를 CPU로
  Expert caching: 자주 사용되는 Expert를 GPU에
```

---

## MoE 효율적 구현 (MegaBlocks)

```python
# MegaBlocks: 희소 MoE의 효율적 GPU 커널
# torch.scatter, gather 대신 전문화된 CUDA 커널 사용

# Dropped Tokens 방지하는 효율적 구현
# 각 expert가 가변 수의 토큰 처리 (padding 없음)
# Block-sparse matrix multiplication 활용

# 표준 구현의 문제:
# for expert_idx in range(n_experts):  # 순차 처리
#     tokens_for_expert = ...
#     output = expert(tokens_for_expert)  # 비효율

# MegaBlocks 스타일:
# 모든 expert를 병렬로 처리
# Variable-size batch per expert
# 전문화된 sparse matmul 커널
```

---

## MoE vs Dense 모델 비교

```
같은 FP 예산에서:
  MoE: 더 많은 파라미터(용량) → 더 많은 지식 저장
  Dense: 더 적은 파라미터 → 더 깊거나 넓은 모델

Chinchilla Scaling에서 MoE:
  같은 compute로 Dense 대비 N× 파라미터 사용 가능
  더 나은 성능 가능

MoE 장점:
  동일 추론 비용으로 더 강한 모델
  전문화된 Expert (특정 도메인 지식 분리)
  학습 효율 (더 빠른 수렴)

MoE 단점:
  메모리 요구량 큼 (모든 Expert 로드)
  분산 학습/추론 복잡
  소형 디바이스 배포 어려움
  Expert Collapse 위험
  Load Balancing 필요
```

| 특성 | Dense 7B | MoE-47B (top-2) |
|------|---------|-----------------|
| 추론 FLOP | 7B 수준 | 13B 수준 |
| 메모리 필요 | ~14GB | ~90GB |
| 성능 | LLaMA-2-7B 수준 | LLaMA-2-70B 수준 |
| 서빙 비용 | 낮음 | 높음 |

---

## Further Questions

**Q. MoE에서 Top-2가 Top-1보다 선호되는 이유?**
> Top-2는 두 Expert의 출력을 가중 합산 → 더 안정적이고 표현력 높음. Expert들이 다양한 관점에서 처리 가능. 학습 안정성 향상. 하지만 연산량 2배. Top-1: 속도 빠름, 통신 적음 (Switch Transformer). Top-2: 성능/안정성 더 좋음 (Mixtral, DeepSeek). Mixtral 연구: Top-2가 실제로 서로 다른 전문성 선택하는 것 확인.

**Q. Expert load balance가 왜 중요한가?**
> 일부 Expert에만 집중되면: 나머지 파라미터 완전히 낭비, GPU 계산 불균형(병목), 전문화 감소(특정 expert만 다양한 태스크 처리). Auxiliary loss: 균등 분배 유도하지만 성능에 영향. DeepSeek-V3의 Loss-free: 성능 저하 없이 균형 달성 → 향후 표준이 될 가능성.

**Q. DeepSeek-MoE의 Fine-grained Expert + Shared Expert 아이디어란?**
> Fine-grained: 64개 작은 Expert 중 6개 선택 → 더 세밀한 조합으로 다양한 태스크 처리 가능. Shared Expert: 모든 토큰이 항상 사용하는 공통 지식 Expert → 중복 지식 저장 방지. 조합: Shared Expert = 일반 지식, Routed = 전문 지식 → 효율적 파라미터 사용.

**Q. MoE를 서빙할 때 주요 도전은?**
> 1) 메모리: 47B MoE → ~90GB 필요 (Dense 13B의 ~3배). 2) 분산 필요: 여러 GPU에 Expert 분산 시 All-to-All 통신. 3) Expert Offloading: 사용 안 되는 Expert를 CPU로, 필요 시 GPU로 → 지연 발생. 4) 배치 처리: 다른 토큰이 다른 Expert → 배치 내 비효율성. 해결: Expert caching, 효율적 커널(MegaBlocks).
