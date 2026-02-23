# Positional Encoding

## 왜 위치 인코딩이 필요한가

```
Transformer Attention의 문제:
  "The dog bit the man" vs "The man bit the dog"
  → Attention 자체는 토큰 순서 정보 없음
  → 위치 정보를 별도로 주입 필요

위치 인코딩 요구사항:
  1. 각 위치에 고유한 값
  2. 서로 다른 위치 간 거리 일관성
  3. 훈련보다 긴 시퀀스로 일반화 가능 (외삽)
  4. 연산 효율성

두 가지 방식:
  Additive: token_emb + pos_emb
  Multiplicative/Rotational: token_emb에 위치 변환 적용 (RoPE)
```

---

## 1. Sinusoidal PE (원 Transformer, 2017)

```
수식:
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

  pos: 위치 (0, 1, 2, ...)
  i: 차원 인덱스 (0, 1, ..., d_model/2 - 1)

직관:
  각 차원이 다른 주파수를 가진 sin/cos 파동
  낮은 i (낮은 인덱스): 고주파 (가까운 위치 구분)
  높은 i (높은 인덱스): 저주파 (멀리 있는 위치 구분)

특성:
  학습 파라미터 없음 (고정)
  PE(pos+k)는 PE(pos)의 선형 변환
  → 상대 위치를 내적으로 표현 가능

한계:
  훈련 길이 이상 외삽 불안정
  절대 위치만 인코딩
```

### 구현
```python
import torch
import math

def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)

    # 1 / 10000^(2i/d_model) 계산
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 차원
    pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 차원

    return pe  # (max_len, d_model)

# 시각화: 위치별 인코딩 다름을 확인
import matplotlib.pyplot as plt

def visualize_sinusoidal_pe():
    pe = sinusoidal_positional_encoding(100, 128)
    # pe[i, j]: position i의 dimension j
    # 고주파 (j=0,1): 빠르게 변함 (인접 위치 구분)
    # 저주파 (j=126,127): 느리게 변함 (먼 위치 구분)
    return pe
```

---

## 2. Learned Positional Embedding

```
각 위치에 학습 가능한 임베딩 벡터 할당:
  pos_embed = nn.Embedding(max_position, d_model)

사용:
  BERT: max_position=512, learned PE
  GPT-2: max_position=1024, learned PE
  ViT: 2D patch position, learned PE

장점:
  태스크에 맞는 위치 정보 학습
  구현 단순

단점:
  훈련 시 max_position 이상 외삽 불가
  → BERT를 512 이상에서 쓰면 성능 급락
  파라미터 수 증가 (max_position × d_model)

확장 기법:
  Position Interpolation: 새 위치 → 훈련 범위 내 위치로 선형 보간
```

---

## 3. ALiBi (Attention with Linear Biases) [Press et al., 2022]

```
아이디어: Attention score에 상대 거리 기반 penalty 직접 추가

수식:
  Attn_ij = (Qᵢ · Kⱼ^T / √d_k) - m_h · (i - j)

  m_h: head별 slope (고정, 학습 아님)
    head 1: slope = 2^(-8/H)
    head 2: slope = 2^(-8·2/H)
    ...
    H개 head에서 기하급수적으로 증가하는 slope

특성:
  멀리 있는 토큰: 강한 negative bias → 자연스러운 decay
  학습 파라미터 없음
  Extrapolation 우수: 훈련보다 긴 시퀀스에서도 안정

장점:
  외삽 성능 탁월 (512 훈련 → 2048+ 추론 가능)
  구현 단순 (attention bias만 추가)
  추가 연산 최소

단점:
  절대 위치 인코딩 없음
  RoPE 대비 일부 태스크에서 성능 열세
  최신 대형 모델에서 RoPE로 대체됨

사용: BLOOM, MPT, BioMedLM
```

```python
def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """ALiBi head slopes 계산"""
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(n_heads))
    else:
        # 2의 거듭제곱이 아닌 경우 보간
        closest_power = 2 ** math.ceil(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_power)[:n_heads]
        return torch.tensor(slopes)

def compute_alibi_bias(seq_len: int, n_heads: int) -> torch.Tensor:
    """ALiBi attention bias 행렬 계산"""
    slopes = get_alibi_slopes(n_heads)  # (n_heads,)

    # 상대 거리 행렬: (seq_len, seq_len)
    positions = torch.arange(seq_len)
    distances = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
    distances = distances.tril()  # Causal: 미래 위치는 0

    # ALiBi bias: (n_heads, T, T)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * distances.unsqueeze(0)
    return alibi  # Negative values → attention decay with distance
```

---

## 4. RoPE (Rotary Position Embedding) [Su et al., 2021]

현대 LLM의 표준 위치 인코딩.

### 핵심 아이디어
```
목표: Q·K 내적이 상대 위치 (m-n)만의 함수가 되도록

절대 위치 m의 Query:
  f(q, m) = q · e^{imθ}  (회전 변환)

절대 위치 n의 Key:
  f(k, n) = k · e^{inθ}

내적:
  f(q,m)ᵀ · f(k,n) = q · e^{i(m-n)θ} · k
                   = g(q, k, m-n)
  → 상대 위치 (m-n)에만 의존!

직관:
  각 토큰 벡터를 위치에 따른 각도로 회전
  두 벡터의 내적 = 두 벡터 간 각도 차이에 의존
  = 상대 위치 정보
```

### 수식 (2D 예시)
```
2D 벡터 [q₁, q₂]를 위치 m으로 회전:
  [q₁]   [cos(mθ)  -sin(mθ)] [q₁]
  [q₂] = [sin(mθ)   cos(mθ)] [q₂]

d_model차원 벡터 (d/2 쌍으로 처리):
  각 2D 쌍 (q_{2i-1}, q_{2i})을 θᵢ로 회전
  θᵢ = 10000^{-2i/d}  (각 쌍마다 다른 각도)

전체:
  q_rot[2i-1] = q[2i-1] cos(m·θᵢ) - q[2i] sin(m·θᵢ)
  q_rot[2i]   = q[2i-1] sin(m·θᵢ) + q[2i] cos(m·θᵢ)
```

### 구현
```python
import torch

def precompute_freqs_cis(d_head: int, max_seq_len: int, base: float = 10000.0):
    """RoPE 주파수 계산 (LLaMA-3 방식)"""
    # θᵢ = 10000^(-2i/d)
    theta = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    # t: 위치 인덱스
    t = torch.arange(max_seq_len, device=theta.device)
    # 외적으로 (seq_len, d_head/2) 계산
    freqs = torch.outer(t, theta)
    # 복소수로 변환: cos + i*sin
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs_cis: torch.Tensor):
    """Q, K에 RoPE 적용"""
    # x: (B, T, H, d_head)
    # freqs_cis: (T, d_head/2) 복소수

    # 복소수로 변환
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 회전 적용 (복소수 곱셈)
    freqs_cis = freqs_cis[:xq_.shape[1]]  # 시퀀스 길이에 맞게 자름
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)
```

### RoPE 주요 모델별 base 값
```
모델               | base      | 컨텍스트 길이 | 특이사항
LLaMA-2-7B        | 10,000    | 4K          | 기본
LLaMA-3-8B        | 500,000   | 8K (128K)   | 긴 컨텍스트 지향
Mistral-7B        | 10,000    | 8K (32K)    | SWA 사용
Qwen2-72B         | 1,000,000 | 128K        | 매우 큰 base
Gemma-2-27B       | 10,000    | 8K          | MQA 사용
Phi-3-Mini        | 10,000    | 4K (128K)   | LongRoPE로 확장

base 값 ↑ → 더 긴 시퀀스에서 고주파 성분의 회전 속도 ↓
→ 훈련 분포 내 위치 표현 가능 (외삽 개선)
```

### RoPE의 장점
```
1. 상대 위치 자연스럽게 인코딩 (수학적 보장)
2. 추가 파라미터 없음
3. 구현 효율적 (복소수 곱셈)
4. 제한적 외삽 가능 (YaRN 등으로 확장)
5. 현재 표준: LLaMA, Mistral, Qwen, Gemma, Falcon, GPT-NeoX, 등
```

---

## 5. RoPE 컨텍스트 확장

### 왜 RoPE는 훈련 이상 길이에서 실패하는가?
```
RoPE θᵢ = 10000^(-2i/d)

저주파 성분 (큰 i): θ 작음 → 긴 시퀀스에도 충분히 다양한 각도
고주파 성분 (작은 i): θ 큼 → 짧은 시퀀스 내 위치 구분

훈련 시 최대 길이 L_train에서 고주파 성분은
  최대 회전 각도: θ_i × L_train

훈련 이상 길이 L > L_train:
  고주파 성분이 "배운 적 없는" 큰 회전 각도에 도달
  → Out-of-distribution → 성능 저하

해결 방향:
  고주파 성분의 회전 속도를 늦추거나
  훈련 범위 내로 "압축"하는 기법
```

### Linear Scaling (단순 방법)
```
위치를 스케일 팩터 s로 나눔:
  원래: position m → RoPE(m)
  새로운: position m → RoPE(m/s)

  L_train = 4K, L_target = 128K → s = 32

문제:
  모든 주파수 성분을 동일하게 스케일
  저주파는 괜찮지만 고주파 성분 품질 저하
  짧은 시퀀스 성능도 하락
```

### NTK-aware Scaling
```
아이디어: Neural Tangent Kernel 관점에서 base 조정

base를 늘리면 고주파 성분의 θ 감소:
  new_base = base × (s^(d/(d-2)))  where s = target_len/train_len

  예: base=10000, d=128, s=4
    new_base = 10000 × 4^(128/126) ≈ 40960

효과:
  고주파 성분 조정 → 훈련 범위 내로 수렴
  저주파 성분 거의 영향 없음
  별도 파인튜닝 없이 즉시 사용 가능 (Dynamic NTK)
```

```python
def ntk_aware_rope(d_head: int, max_seq_len: int,
                   original_max_len: int = 4096, base: float = 10000.0):
    """NTK-aware scaling: base 자동 조정"""
    scale = max_seq_len / original_max_len
    if scale <= 1:
        return precompute_freqs_cis(d_head, max_seq_len, base)

    # NTK 공식으로 새 base 계산
    new_base = base * (scale ** (d_head / (d_head - 2)))
    return precompute_freqs_cis(d_head, max_seq_len, new_base)

def dynamic_ntk_rope(d_head: int, current_seq_len: int,
                     original_max_len: int = 4096, base: float = 10000.0):
    """Dynamic NTK: 현재 시퀀스 길이에 따라 자동 조정"""
    if current_seq_len <= original_max_len:
        return precompute_freqs_cis(d_head, current_seq_len, base)
    else:
        scale = current_seq_len / original_max_len
        new_base = base * (scale ** (d_head / (d_head - 2)))
        return precompute_freqs_cis(d_head, current_seq_len, new_base)
```

### YaRN (Yet another RoPE extensioN) [Peng et al., 2023]
```
핵심: 주파수에 따라 다른 스케일링 전략

세 영역으로 분류:
  1. 저주파 (큰 i): 선형 스케일링 → 충분히 다양한 각도 확보
  2. 고주파 (작은 i): 스케일링 없음 → 기존 위치 유지
  3. 중간: 보간 (interpolation)

구체적으로:
  저주파 threshold: λ_lo = d/(2π × L_old)^(2/d) 이상의 파장
  고주파 threshold: λ_hi = L_old 이하의 파장

스케일링 팩터 계산:
  s(i) = 0           if λᵢ > λ_hi (완전 보간)
  s(i) = 1           if λᵢ < λ_lo (스케일 없음)
  s(i) = interpolate  otherwise

추가: Temperature 스케일링
  attention_score / sqrt(t · d_k)
  t = 0.1 × ln(s) + 1.0 → 긴 컨텍스트에서 attention이 sharp해지는 것 보정

결과:
  4K 훈련 모델 → 128K 추론 (minimal fine-tuning)
  LLaMA-2-7B + YaRN fine-tuning: 128K 컨텍스트 달성

사용: LLaMA-2 Long, Mistral Long, 많은 오픈소스 long-context 모델
```

```python
def yarn_rope(d_head: int, max_seq_len: int,
              original_max_len: int = 4096,
              base: float = 10000.0,
              beta_fast: int = 32, beta_slow: int = 1):
    """YaRN: 주파수별 차별화된 스케일링"""
    scale = max_seq_len / original_max_len

    # 각 차원의 파장 계산
    freqs = base ** (torch.arange(0, d_head, 2).float() / d_head)
    wavelen = 2 * math.pi * freqs  # (d_head/2,)

    # 영역 분류
    lo_freq = original_max_len / beta_slow   # 저주파 임계값
    hi_freq = original_max_len / beta_fast   # 고주파 임계값

    # 보간 계수 계산
    # wavelen < hi_freq: 고주파 → 스케일 없음 (s=1)
    # wavelen > lo_freq: 저주파 → 완전 스케일 (s=scale)
    # 중간: 선형 보간
    smooth = (wavelen - hi_freq) / (lo_freq - hi_freq)
    smooth = smooth.clamp(0, 1)  # 0~1로 클램핑

    # 혼합 스케일링
    new_freqs = (1 - smooth) * freqs / scale + smooth * freqs

    # Temperature 스케일링 인자
    temperature = 0.1 * math.log(scale) + 1.0 if scale > 1 else 1.0

    # 주파수 계산
    t = torch.arange(max_seq_len)
    freqs_out = torch.outer(t, 1.0 / new_freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_out), freqs_out)

    return freqs_cis, temperature
```

### LongRoPE (Microsoft, 2024)
```
비균일 위치 보간:
  각 위치에서 다른 보간 계수 사용
  진화 알고리즘으로 최적 보간 계수 탐색

  position m → ŵ(m) × m  (비균일 가중치)

효과:
  더 정확한 위치 표현
  Phi-3-Mini에서 128K context 달성

Context Window Extension 비교:
  방법          | 외삽 성능 | Fine-tuning 필요 | 단점
  Linear Scale  | 낮음      | 필요 (많이)      | 짧은 시퀀스 성능 저하
  NTK-aware     | 중간      | 없음 (동적)      | 성능 불안정
  YaRN          | 높음      | 약간 필요        | θ 설정 복잡
  LongRoPE      | 매우 높음 | 필요             | 최적화 비용
```

---

## 6. T5 Relative Position Bias

```
T5 (2020): Relative position bias 방식

수식:
  Attn(i, j) = Q_i · K_j / √d_k + b(i-j)

  b(i-j): 상대 위치 (i-j)에 대한 학습된 스칼라 bias
  → 거리 버킷으로 그룹화 (32개 버킷)

버킷:
  작은 거리: 개별 버킷 (세밀)
  큰 거리: 로그 스케일로 그룹화 (거칠게)

장점: 외삽 가능 (어느 정도)
단점: 파라미터 필요 (버킷 수 × head 수)
사용: T5, UL2, Flan-T5
```

---

## 7. M-ROPE (Multimodal RoPE, Qwen2-VL)

```
이미지/비디오에 2D/3D 위치 인코딩 적용

텍스트: 1D RoPE (시퀀스 위치)
  position_ids: [0, 1, 2, ..., T]

이미지: 2D RoPE
  height_id: [0, 0, 1, 1, 2, 2, ...]  (행 위치)
  width_id:  [0, 1, 0, 1, 0, 1, ...]  (열 위치)
  채널의 절반: height_id로 회전
  나머지 절반: width_id로 회전

비디오: 3D RoPE
  time_id:   [0, 0, ..., 1, 1, ..., 2, 2, ...]  (프레임)
  height_id: 각 프레임 내 행 위치
  width_id:  각 프레임 내 열 위치

효과:
  이미지 패치의 공간적 관계를 위치 인코딩에 반영
  비디오 프레임의 시간적 순서 인식
  기존 1D RoPE와 호환 (텍스트 부분)
```

```python
def compute_mrope_position_ids(
    text_tokens: int,
    image_patches: list[tuple[int, int]],  # list of (height, width)
) -> torch.Tensor:
    """M-ROPE: 텍스트 + 이미지 패치의 위치 ID 계산
    Returns: (3, total_tokens) - (time, height, width) positions
    """
    positions = []

    # Text positions: all 3 dimensions same (1D)
    for i in range(text_tokens):
        positions.append((i, i, i))  # t=h=w=i

    pos_counter = text_tokens

    # Image positions: 2D (h, w), time stays same
    for h, w in image_patches:
        for row in range(h):
            for col in range(w):
                positions.append((pos_counter, row, col))
        pos_counter += h * w  # Advance text position counter

    pos_tensor = torch.tensor(positions).T  # (3, total_tokens)
    return pos_tensor
```

---

## 8. Attention Sink (주의 집중 싱크)

```
LLM의 흥미로운 현상: 초기 토큰(들)이 매우 높은 attention 받음

관찰 (Xiao et al., 2023):
  대부분의 attention이 첫 번째 토큰(들)에 집중
  해당 토큰의 내용과 무관

해석:
  Softmax normalization 특성:
    모든 attention weight 합 = 1
    → "불필요한" attention을 버릴 곳이 필요
    → 초기 토큰 = "휴지통" 역할

StreamingLLM (Xiao et al., 2023):
  KV cache에서 이 "sink tokens"는 항상 유지
  최근 window + sink tokens = 효과적인 무한 컨텍스트

  [sink_token_1, sink_token_2, ..., recent_window...]

효과:
  Sliding window가 없는 경우보다 훨씬 안정적
  고정 메모리로 임의 길이 처리 가능
```

---

## 9. NoPE (No Positional Encoding)

```
일부 모델: 위치 인코딩 없이도 동작

원리:
  Causal attention mask 자체가 순서 정보 제공
  → 현재 위치 이전만 attend 가능 → 순서 암묵적

연구 결과:
  짧은 시퀀스: NoPE 모델도 위치 정보 어느 정도 학습
  긴 시퀀스: 위치 인코딩 없으면 성능 저하

실제 사용:
  ALiBi처럼 일부 모델은 최소한의 위치 bias만 사용
  대부분은 RoPE 표준 채택
```

---

## 10. 비교 요약

| 방법 | 외삽 | 파라미터 | 상대위치 | 특징 | 사용 모델 |
|------|------|---------|---------|------|---------|
| Sinusoidal | 제한 | 없음 | 암묵적 | 원 Transformer | Transformer 원본 |
| Learned | 불가 | 있음 | 없음 | 단순 | BERT, GPT-2 |
| ALiBi | 우수 | 없음 | 명시적 bias | 단순, 외삽 좋음 | BLOOM, MPT |
| T5 RelPE | 보통 | 있음 | 명시적 | 버킷 기반 | T5, Flan |
| RoPE | 보통 | 없음 | 명시적 회전 | 현대 표준 | LLaMA, 대부분 최신 |
| RoPE+YaRN | 우수 | 없음 | 명시적 회전 | 장거리 확장 | Long Context 모델 |
| M-ROPE | 우수 | 없음 | 2D/3D 회전 | VLM용 | Qwen2-VL |

---

## Further Questions

**Q. RoPE가 Learned PE보다 나은 이유는?**
> 1) 상대 위치를 수학적으로 Q·K 내적에서 자연스럽게 표현 (절대 위치지만 내적은 상대 위치만 의존). 2) 외삽 가능 (YaRN 등으로 확장). 3) 추가 파라미터 없음 → 임의 길이에 적용 가능. Learned PE는 훈련 최대 길이에 종속.

**Q. 같은 모델을 더 긴 컨텍스트에서 사용하려면?**
> 1) YaRN 또는 NTK-aware scaling 적용 (코드 한 줄). 2) 긴 컨텍스트 데이터로 소량 파인튜닝 (Long SFT). 3) Sliding Window Attention으로 교체 (구조 변경 필요). 실용적: YaRN dynamic scaling 먼저 시도 → 성능 부족 시 파인튜닝.

**Q. ALiBi와 RoPE의 핵심 차이는?**
> ALiBi: attention score에 상대 거리 linear penalty 직접 더함 (- m × |i-j|). 명시적, 단순하지만 query/key 벡터 자체는 무관. RoPE: Q,K 벡터 자체를 위치에 따라 회전 → 내적이 상대 위치의 함수. 더 풍부한 위치 표현, 현대 LLM 표준.

**Q. RoPE에서 base(10000) 값이 컨텍스트 길이에 미치는 영향은?**
> base가 클수록 저주파 성분의 주기가 길어짐 → 더 긴 시퀀스에서 다양한 위치 표현. LLaMA-3: base=500,000 (기본 10,000에서 50배 증가) → 긴 컨텍스트를 위한 설계. base를 높이는 것만으로도 컨텍스트 확장 효과 (NTK-aware scaling의 핵심 원리).

**Q. YaRN이 단순 NTK scaling보다 나은 이유는?**
> NTK: 모든 주파수에 동일한 base 조정. 저주파/고주파 각각의 특성 무시. YaRN: 저주파는 선형 스케일링, 고주파는 그대로, 중간은 보간. 각 주파수 성분의 역할에 맞는 맞춤 처리. 추가로 Temperature 보정으로 긴 컨텍스트에서 attention 집중도 유지.

**Q. Attention Sink 현상이 실용적으로 중요한 이유는?**
> StreamingLLM: KV cache 무한 확장 대신 sink tokens + sliding window로 고정 메모리. 무한 컨텍스트 스트리밍 처리 가능. sink token이 없으면 최근 토큰만 유지할 때 성능 급락. 실제 LLM 서버에서 매우 긴 대화 처리에 활용 가능.
