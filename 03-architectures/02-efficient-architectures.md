# 효율적 아키텍처 (SSM, Hybrid, etc.)

## Transformer의 한계

```
1. Quadratic Attention: O(n²d)
   → n=128K: 128K² × d 연산 = 사실상 불가능
   → Flash Attention으로 완화 but 메모리/연산 근본 한계

2. KV Cache 폭발:
   → 시퀀스 길이에 비례 메모리 증가
   → n=128K, LLaMA-3-8B: ~4GB/sequence (GQA 기준)
   → 배치 처리 시 메모리 병목

3. Autoregressive 추론의 순차성:
   → 각 토큰 생성 시 전체 KV cache 접근
   → Memory bandwidth bottleneck
   → O(n) decode per token

4. 극단적 긴 시퀀스 (1M+):
   → Transformer는 실용적으로 한계
   → 다른 아키텍처 필요
```

---

## State Space Models (SSM) 수학

```
연속 시간 선형 동적 시스템:
  x'(t) = Ax(t) + Bu(t)   (상태 방정식)
  y(t) = Cx(t) + Du(t)    (출력 방정식)

  x(t): 숨겨진 상태 (state) ∈ R^N
  u(t): 입력 시퀀스 ∈ R
  y(t): 출력 ∈ R
  A: N×N 상태 전이 행렬 (Long-range dependency 핵심)
  B, C: 입력/출력 투영

이산화 (실제 시퀀스에 적용):
  Zero-Order Hold (ZOH):
    Ā = exp(ΔA)
    B̄ = (ΔA)⁻¹(exp(ΔA) - I) · ΔB
    (Δ: time step, 실수 양수)

이산 순환 (Recurrent - 추론):
  h_t = Āh_{t-1} + B̄u_t   (O(1) per step)
  y_t = Ch_t + Du_t

이산 컨볼루션 (Parallel - 학습):
  ȳ = u * K̄,  K̄_i = CA^i B  (Cauchy kernel)
  FFT 활용: O(n log n) 병렬 처리

장점 비교:
  학습: 컨볼루션으로 Transformer와 동일한 병렬성
  추론: 순환으로 O(1) 메모리 (고정 크기 상태 h)
```

---

## S4 (Structured State Space Sequences) [Gu et al., 2021]

```
핵심 기여: HiPPO 이론 기반 A 행렬 초기화

문제: 일반 A 행렬로는 gradient vanishing/exploding
해결: HiPPO (High-order Polynomial Projection Operator)
  A_{nk} = {
    -√(2n+1)√(2k+1)  if n > k  (off-diagonal)
    -(n+1)             if n = k  (diagonal)
    0                  if n < k
  }
  → 과거 입력을 다항식으로 최적 근사
  → Long-range dependency를 안정적으로 학습

S4의 구조적 특성 (DPLR):
  A = Λ - PQᵀ
  Λ: 대각 행렬 (복소수 eigenvalues)
  P, Q: low-rank 수정 (rank 1)
  → 행렬 역산/지수를 O(N log N)으로 계산

성능:
  Long Range Arena (LRA): Transformer 압도 (특히 Path-X 문제)
  1D 신호(음성), DNA 시퀀스, 등 긴 시퀀스에서 강점
```

---

## Mamba [Gu & Dao, 2023]

### S4의 한계 및 Mamba의 혁신

```
S4 한계: 고정된 A, B, C → 내용(content)에 무관한 처리
  → 중요한 정보와 불필요한 정보를 동등하게 처리
  → Transformer의 selective attention 능력 없음

Mamba 핵심: Selective SSM
  B, C, Δ를 입력 u에 따라 동적으로 결정!

  B_t = Linear_B(u_t) ∈ R^N   [input-dependent]
  C_t = Linear_C(u_t) ∈ R^N   [input-dependent]
  Δ_t = softplus(Linear_Δ(u_t)) ∈ R  [input-dependent time step]

직관:
  Δ_t 큰 경우: 해당 입력에 "주목" → 불연속적 상태 업데이트
  Δ_t 작은 경우: 해당 입력 "무시" → 이전 상태 유지
  → 내용 기반 선택적 처리!

이산화:
  Ā_t = exp(Δ_t A)   [input-dependent!]
  B̄_t = (Δ_t A)⁻¹(Ā_t - I) · Δ_t B_t
```

### Mamba 구조 및 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    """Mamba SSM Block"""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: int = None):
        super().__init__()
        d_inner = d_model * expand  # 내부 차원
        if dt_rank is None:
            dt_rank = max(1, d_model // 16)  # Δ rank

        # 입력 투영
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)  # x와 z 분리
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv,
                                 padding=d_conv-1, groups=d_inner)  # depthwise conv

        # SSM 파라미터
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)  # Δ, B, C
        self.dt_proj = nn.Linear(dt_rank, d_inner)  # Δ 투영

        # SSM 고정 파라미터 (학습 가능하지만 input-independent)
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))  # log space for stability
        self.D = nn.Parameter(torch.ones(d_inner))  # skip connection

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # 입력 분리 (x: SSM 입력, z: gate)
        x_z = self.in_proj(x)  # [B, L, d_inner*2]
        x_ssm, z = x_z.chunk(2, dim=-1)

        # Depthwise convolution (로컬 컨텍스트)
        x_ssm = self.conv1d(x_ssm.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_ssm = F.silu(x_ssm)

        # Selective SSM 파라미터 계산
        x_dbc = self.x_proj(x_ssm)  # [B, L, dt_rank + 2*d_state]
        dt, B_mat, C_mat = x_dbc.split([self.dt_proj.in_features,
                                         self.A_log.shape[1],
                                         self.A_log.shape[1]], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # [B, L, d_inner]
        A = -torch.exp(self.A_log)  # [d_inner, d_state]

        # Selective Scan (핵심 연산, 실제로는 CUDA 커널)
        y = self._selective_scan(x_ssm, dt, A, B_mat, C_mat)

        # 게이팅 + skip
        y = y + x_ssm * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)

        return self.out_proj(y)

    def _selective_scan(self, u, dt, A, B, C):
        """
        Selective scan (교육용 Python 구현, 실제는 CUDA parallel scan)
        실제 Mamba는 torch_extensions의 selective_scan_cuda 사용
        """
        B_batch, L, D = u.shape
        N = A.shape[-1]  # d_state

        h = torch.zeros(B_batch, D, N, device=u.device)
        ys = []

        for t in range(L):
            # input-dependent 파라미터
            A_t = torch.exp(dt[:, t, :, None] * A[None, :, :])  # [B, D, N]
            B_t = dt[:, t, :, None] * B[:, t, None, :]          # [B, D, N]

            # 상태 업데이트
            h = A_t * h + B_t * u[:, t, :, None]

            # 출력
            y_t = (h * C[:, t, None, :]).sum(dim=-1)  # [B, D]
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # [B, L, D]
```

### Mamba 성능 특성

```
일반 LM 벤치마크 (2024):
  1.4B Mamba ≈ 2.8B Transformer (파라미터 효율 ~2×)
  학습 속도: Transformer 대비 5× 빠름 (긴 시퀀스)
  추론 속도: 5× 빠름 (KV cache 없음 → 고정 상태)

Mamba의 취약점:
  In-context Learning: Transformer가 더 강함
  복잡한 추론: attention의 "정확한" 관계 파악 어려움
  Association recall: "A is B" 형태의 사실 기억 약함

적합한 태스크:
  생물학적 시퀀스 (DNA, 단백질 - 매우 긴 시퀀스)
  오디오/비디오 처리 (스트리밍)
  스트리밍 추론 (고정 메모리 필요)
```

---

## Mamba-2 [Dao & Gu, 2024]

### State Space Duality (SSD)

```
핵심 발견: SSM = Structured Masked Attention의 특수 케이스

SSM 행렬 형태:
  h_t = A_t h_{t-1} + B_t u_t
  y_t = C_t h_t

  → 전체 출력: Y = (C ⊙ A_cumul) × (B ⊙ u)ᵀ
     = L_A ⊙ (CBᵀ) × u  (구조화된 attention으로 표현!)

  L_A: semi-separable (특수한 하삼각 행렬)

핵심 통찰:
  SSM = A-weighed, B-input, C-output attention
  → Flash Attention 최적화를 SSM에 적용 가능!
  → Attention을 SSM처럼 O(1) 메모리로 표현 가능

Mamba-2 장점:
  더 큰 d_state 지원 (d_state ≥ 64, Mamba-1은 16)
  Head dimension 구조: Multi-head SSM → Tensor Parallelism 지원
  더 빠른 CUDA 커널 (SSD 형태로 최적화)

성능:
  Mamba-2 2.7B > Mamba-1 2.8B (동일 파라미터, 더 좋은 성능)
  Transformer와의 성능 격차 더욱 감소
```

---

## RWKV (Receptance Weighted Key Value)

```
선형 Transformer와 RNN의 결합:

핵심: WKV (Weighted Key Value) 메커니즘
  RNN과 Attention을 하나의 수식으로 통합

Time Mixing (시간 차원):
  r_t = sigmoid(W_r · x_t + U_r · x_{t-1})  (receptance gate)
  k_t = W_k · x_t + U_k · x_{t-1}           (key)
  v_t = W_v · x_t + U_v · x_{t-1}           (value)

  WKV_t = (Σ_{i<t} exp(-(t-1-i)w + k_i) v_i + exp(u+k_t) v_t)
          / (Σ_{i<t} exp(-(t-1-i)w + k_i)   + exp(u+k_t))
  w: 채널별 decay (학습됨)
  u: "current position" 특별 가중치

  x_t = r_t ⊙ sigmoid(W_o · WKV_t)  (receptance gating)

Channel Mixing (FFN 대체):
  r_t = sigmoid(W_r · x_t)
  k_t = relu(W_k · x_t)²  (squared ReLU)
  x_t = r_t ⊙ W_v · k_t

학습: 커스텀 CUDA 커널로 병렬화
추론: RNN으로 O(1) 메모리 per step

RWKV v4: 14B까지 확장, GPT-2 수준
RWKV v5 (Eagle): Multi-headed WKV
RWKV v6 (Finch): 더 표현적인 time mixing
```

```python
class RWKV_TimeMixing(nn.Module):
    """RWKV Time Mixing (v5/v6 간략 구현)"""

    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        d_head = d_model // num_heads

        # 시간 가중치 (채널별)
        self.time_decay = nn.Parameter(torch.ones(d_model))
        self.time_first = nn.Parameter(torch.ones(d_model))  # u (current)

        # 토큰 믹싱 파라미터
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))

        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # 이전 토큰과 현재 토큰 믹싱
        x_prev = torch.cat([torch.zeros(B, 1, C, device=x.device), x[:, :-1]], dim=1)
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_prev * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))

        # WKV 계산 (교육용 sequential, 실제는 parallel CUDA)
        wkv = self._compute_wkv(k, v)

        return self.output(r * wkv)

    def _compute_wkv(self, k, v):
        """WKV sequential computation (추론용)"""
        B, T, C = k.shape
        w = -torch.exp(self.time_decay)  # negative decay
        u = self.time_first

        output = []
        num = denom = torch.zeros(B, C, device=k.device)

        for t in range(T):
            kk = k[:, t, :]
            vv = v[:, t, :]

            # Current token (special weight u)
            wkv_t = (num + torch.exp(u + kk) * vv) / (denom + torch.exp(u + kk))
            output.append(wkv_t)

            # Update state
            num = torch.exp(w) * num + torch.exp(kk) * vv
            denom = torch.exp(w) * denom + torch.exp(kk)

        return torch.stack(output, dim=1)
```

---

## Hybrid 아키텍처

### Jamba (AI21 Labs, 2024)

```
Mamba + Transformer + MoE 결합:

구성 (52B 모델):
  총 레이어: 32개
  Mamba:Attention 비율 = 7:1
  MoE: 일부 레이어 (16 expert, top-2 선택)
  Active Parameters: ~12B (52B 중 라우팅 후 활성화)

레이어 구성 예:
  [Mamba, Mamba, Mamba, Mamba, Mamba, Mamba, Mamba, Attention-MoE] × 4

효과:
  128K 컨텍스트 처리 (Transformer 대비 8× 메모리 효율)
  Jamba 7:1 vs pure Mamba: ICL 성능 크게 향상
  Mixtral-8x7B 대비 메모리 8× 절약
  성능: Mixtral 수준

후속:
  Jamba-1.5-Mini (12B), Jamba-1.5-Large (94B)
  256K 컨텍스트, 향상된 성능
```

### Zamba2 (Zyphra, 2024)

```
Mamba2 + Single Shared Attention:

핵심 혁신:
  N개 Mamba2 레이어마다 1개 Shared Attention 재사용
  → 모든 Mamba 레이어가 같은 Attention 파라미터 공유

구조:
  [Mamba2] × 6 → [Shared Attention] → [Mamba2] × 6 → [Shared Attention] → ...
  Shared Attention: 한 번만 정의, 모든 위치에서 재사용

Zamba2-7B:
  디바이스 배포에 최적화
  Mistral-7B와 유사한 성능
  추론 속도 2× 향상
  메모리 1.5× 절약

장점: 파라미터 효율 (공유 Attention)
단점: Shared Attention의 다양성 제한
```

### Griffin / Hawk (Google DeepMind, 2024)

```
선형 순환 + 로컬 Attention 결합:

Hawk (Recurrent only):
  선형 순환 레이어만 (Attention 없음)
  Linear Recurrence:
    h_t = Λ_t ⊙ h_{t-1} + C ⊙ (u_t W_x)
    Λ_t = sigmoid(u_t W_Λ)  (입력 의존적 gate)

  특성:
    O(1) 추론 메모리
    Mamba와 다른 접근 (diagonal transition matrix)
    14B Hawk > Mamba-2 이상

Griffin (Hawk + Local Attention):
  교번 구조: [Recurrence] → [Local Attention (window)] → ...
  Local Attention: window 내에서만 attention (전역 X)
  window size: 1024 tokens

성능:
  14B Griffin > Mamba-2 > 7B RWKV
  Gemma 아키텍처와 통합 가능
  Google의 실용적 hybrid 연구 방향
```

---

## RetNet (Microsoft, 2023)

### Retention 메커니즘

```python
class RetentionLayer(nn.Module):
    """RetNet Retention (세 가지 계산 등가 모드)"""

    def __init__(self, d_model: int, num_heads: int = 8, gamma: float = 0.9):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.gamma = gamma  # decay rate

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel mode (학습 - Transformer처럼)"""
        B, L, D = x.shape
        Q = self.W_Q(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)

        # Decay matrix D_{mn} = γ^{m-n} if m >= n, else 0
        positions = torch.arange(L, device=x.device)
        D = torch.pow(self.gamma, (positions.unsqueeze(1) - positions.unsqueeze(0)).clamp(min=0))
        D = D * torch.tril(torch.ones(L, L, device=x.device))  # causal masking

        # Retention = (QKᵀ ⊙ D) V
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = attn * D.unsqueeze(0).unsqueeze(0)
        return self.W_O(torch.matmul(attn, V).transpose(1, 2).reshape(B, L, D))

    def forward_recurrent(self, x_t: torch.Tensor, state: torch.Tensor) -> tuple:
        """Recurrent mode (추론 - O(1) 메모리)"""
        q = self.W_Q(x_t)  # [B, D]
        k = self.W_K(x_t)  # [B, D]
        v = self.W_V(x_t)  # [B, D]

        # State 업데이트: s_n = γ · s_{n-1} + kᵀv
        new_state = self.gamma * state + k.unsqueeze(-1) * v.unsqueeze(-2)
        # Output: Ret(X_n) = q · s_n
        y = (q.unsqueeze(-2) @ new_state).squeeze(-2)

        return self.W_O(y), new_state

    def forward_chunkwise(self, x: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
        """Chunkwise mode (긴 시퀀스 학습 효율화)"""
        B, L, D = x.shape
        chunks = x.split(chunk_size, dim=1)
        outputs = []
        state = torch.zeros(B, self.num_heads, self.d_head, self.d_head, device=x.device)

        for chunk in chunks:
            # 청크 내: Parallel 방식
            # 청크 간: Recurrent 방식으로 state 전달
            out = self.forward_parallel(chunk)
            outputs.append(out)
            # state 업데이트는 생략 (교육용)

        return torch.cat(outputs, dim=1)

세 가지 계산 등가:
  학습: Parallel (GPU 친화적)
  추론: Recurrent (O(1) 메모리)
  긴 시퀀스: Chunkwise (메모리 효율)
```

---

## 비교 요약

| 아키텍처 | 학습 복잡도 | 추론 메모리 | 추론 속도 | ICL 능력 | 긴 시퀀스 |
|---------|-----------|-----------|---------|---------|---------|
| Transformer | O(n²d) | O(n) KV | O(n)/step | 최강 | 한계 |
| S4 | O(n log n) | O(d_state) 고정 | O(1) | 약함 | 강함 |
| Mamba | O(n) | O(d_state) 고정 | O(1) | 중간 | 강함 |
| Mamba-2 | O(n) | O(d_state) 고정 | O(1) | 중간+ | 강함 |
| RWKV | O(n) | O(d_model) 고정 | O(1) | 약-중간 | 강함 |
| RetNet | O(n) / O(n²) | O(d²) 고정 | O(1) | 중간 | 강함 |
| Griffin | O(n) 혼합 | 혼합 | O(1) 근사 | 중간+ | 강함 |
| Jamba | O(n) 대부분 | 혼합 | 빠름 | 강함 | 매우 강함 |

---

## 실전 선택 가이드

```
순수 Transformer가 최선인 경우:
  - 표준 NLP 태스크 (Q&A, 요약, 생성)
  - In-context learning 중요
  - 사전학습 모델 생태계 활용 필요
  - 컨텍스트 < 32K

SSM/Hybrid를 고려할 경우:
  - 컨텍스트 길이 1M+ (DNA, 오디오, 비디오)
  - 엣지/모바일 배포 (고정 메모리)
  - 스트리밍 추론 (infinite sequence)
  - 생물정보학 (DNA, 단백질 분석)

Hybrid가 균형적:
  - 긴 컨텍스트 + 일반 성능 동시 필요
  - Jamba: 128K+ 컨텍스트 + Transformer 수준
  - Griffin: 효율 + 품질 균형

2025 트렌드:
  Mamba + Transformer Hybrid가 주류
  Mamba-2 (SSD): Flash Attention 최적화 적용
  새 아키텍처보다 Scaling + 데이터가 더 중요한 경향
```

---

## Further Questions

**Q. Mamba가 Transformer를 완전히 대체할 수 있나?**
> 현재는 아님. 한계: 1) Association Recall 약함 (in-context "A is B" 기억 어려움). 2) 복잡한 추론에서 Transformer보다 약함. 3) ICL 성능 격차. 현실적 방향: Hybrid (Mamba + 일부 Attention)가 최적 균형. 긴 시퀀스(1M+)에서는 Mamba가 유일한 실용적 선택.

**Q. SSM의 핵심 장점과 한계는?**
> 장점: 추론 O(1) 메모리 (고정 크기 state), O(1) 시간/step, 학습 O(n) 병렬성, 긴 시퀀스 이론적 처리. 한계: State 크기 고정 → 정보 손실 가능 (Transformer는 전체 KV 유지). Input-dependent 파라미터 → 병렬화 어려움 (병렬 scan으로 해결). ICL, 추론 약함.

**Q. Mamba-2의 SSD가 중요한 이유는?**
> SSM과 Attention이 같은 프레임워크라는 것을 보임 → Flash Attention 최적화를 SSM에 적용 가능. 더 큰 d_state (16→64+) → 표현력 향상. Multi-head 구조 → Tensor Parallelism 자연스럽게 지원. 이론적으로 Attention의 특정 구조화된 형태가 SSM과 동일.

**Q. 새로운 아키텍처를 실제로 채택해야 할 때는?**
> 명확한 use case가 있을 때: 1) 긴 시퀀스(1M+) 처리 필요, 2) 고정 메모리로 스트리밍 추론(IoT/엣지), 3) 의료/바이오 시퀀스. 일반 LLM 앱: 여전히 Transformer(LLaMA 계열)가 최선 (생태계, fine-tuned 모델). Hybrid: Jamba처럼 긴 컨텍스트 + 일반 성능 둘 다 필요 시.
