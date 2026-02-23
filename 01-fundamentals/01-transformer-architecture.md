# Transformer Architecture

## 핵심 논문
- **"Attention is All You Need"** (Vaswani et al., 2017) — Transformer 원조
- **"BERT"** (Devlin et al., 2018) — Encoder-only, MLM
- **"GPT-2/3"** (Radford et al.) — Decoder-only, Autoregressive

---

## 전체 구조

```
Input Tokens
    ↓
Token Embedding + Positional Encoding
    ↓
[Transformer Block] × N
    ├─ (Multi-Head) Self-Attention
    ├─ Add & LayerNorm (Pre-LN or Post-LN)
    ├─ Feed-Forward Network (FFN)
    └─ Add & LayerNorm
    ↓
Output Head (LM Head / Classification Head)
```

---

## 핵심 컴포넌트

### 1. Multi-Head Self-Attention (MHSA)
```
Q = X·W_Q,  K = X·W_K,  V = X·W_V

Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V

MultiHead = Concat(head₁, ..., headₕ) · W_O
where headᵢ = Attention(QWᵢ_Q, KWᵢ_K, VWᵢ_V)
```
- `d_k` = head dimension (= d_model / num_heads)
- √d_k 로 나누는 이유: 내적값이 커지면 softmax gradient 소실
- 시간 복잡도: **O(n² · d)** — 시퀀스 길이에 이차

### 2. Feed-Forward Network (FFN)
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```
- Hidden dim은 보통 d_model × 4
- 현대 모델: **SwiGLU**, **GeGLU** 등 Gated activation 사용
  ```
  SwiGLU(x) = (xW₁ · SiLU(xW₃)) · W₂
  ```
- FFN이 모델 파라미터의 대부분 차지 (~2/3)

### 3. LayerNorm
- **Post-LN** (원 논문): Residual 연결 후 정규화 → 학습 불안정
- **Pre-LN** (현대): Residual 연결 전 정규화 → 학습 안정
- **RMSNorm** (LLaMA): 평균 없이 RMS만 사용 → 더 빠름
  ```
  RMSNorm(x) = x / RMS(x) · γ,   RMS(x) = √(1/n · Σxᵢ²)
  ```

---

## Encoder vs Decoder vs Encoder-Decoder

| 유형 | 모델 | Attention | 용도 |
|------|------|-----------|------|
| Encoder-only | BERT, RoBERTa | Bidirectional | 분류, NER, 임베딩 |
| Decoder-only | GPT, LLaMA | Causal (masked) | 생성, 범용 |
| Encoder-Decoder | T5, BART | Cross-Attention | 번역, 요약 |

### Causal Masking (Decoder)
```
미래 토큰을 볼 수 없도록 attention mask 적용
[1 0 0 0]
[1 1 0 0]  ← 하삼각 행렬
[1 1 1 0]
[1 1 1 1]
```

---

## 중요 하이퍼파라미터

| 파라미터 | 설명 | GPT-3 | LLaMA-2-7B |
|---------|------|-------|------------|
| d_model | 히든 차원 | 12288 | 4096 |
| num_layers | 레이어 수 | 96 | 32 |
| num_heads | 어텐션 헤드 수 | 96 | 32 |
| d_ff | FFN 히든 차원 | ~49152 | 11008 |
| vocab_size | 어휘 크기 | 50257 | 32000 |
| max_seq_len | 최대 시퀀스 길이 | 2048 | 4096 |

---

## 파라미터 수 계산

```
Embedding: vocab_size × d_model
Per Layer:
  Attention: 4 × d_model² (Q, K, V, O 행렬)
  FFN: 8 × d_model² (SwiGLU 기준 3개 행렬 + 약간)
  LayerNorm: 2 × d_model (γ, β)

Total ≈ num_layers × 12 × d_model²  (근사치)

예: LLaMA-2-7B
  32 × 12 × 4096² ≈ 6.4B (임베딩 제외, 실제는 7B)
```

---

## Scaling Laws

**Chinchilla Scaling Law** (Hoffmann et al., 2022):
```
N_opt = 20 × D  (모델 파라미터 수 ≈ 20 × 학습 토큰 수)

예: 7B 모델 → 최적 학습 토큰 = 140B
   LLaMA-2-7B는 2T 토큰 → over-trained (추론 효율 최적화)
```

- Compute-optimal: FLOP budget 주어졌을 때 모델 크기와 데이터 크기의 최적 비율
- **Inference-optimal**: 실제로는 작은 모델을 더 많이 학습시키는 게 추론 비용 절약

---

## Attention 수식 상세 & 수치 안정성

### √d_k 스케일링의 필요성
```
d_k 차원의 랜덤 벡터 q, k:
  E[qᵢ] = 0, Var[qᵢ] = 1 가정
  q·k = Σᵢ qᵢkᵢ 의 분산 = d_k

  d_k = 64: 표준편차 = 8 → softmax 입력이 매우 큰 값
  → softmax 포화 → gradient ≈ 0 (vanishing)

  √d_k 나눔: 분산이 1로 정규화
  → softmax가 합리적인 범위에서 작동
```

### Numerical Stability in Softmax
```
문제: softmax(x)에서 x가 크면 exp(x) → inf (overflow)

해결: softmax 수치 안정화
  max_x = max(x_i)
  softmax(x)_i = exp(x_i - max_x) / Σ_j exp(x_j - max_x)

  max 빼도 softmax 값 동일 (분자/분모 같이 약분)

코드:
  scores = Q @ K.T / sqrt(d_k)
  scores = scores - scores.max(dim=-1, keepdim=True)  # 수치 안정화
  weights = torch.softmax(scores, dim=-1)
  out = weights @ V
```

### Causal Masking 구현
```python
# 마스킹 방법: -inf 추가 (softmax 후 0이 됨)
def causal_mask(seq_len, device):
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device),
        diagonal=1  # 상삼각 (미래 위치)
    ).bool()
    return mask

# 적용
scores = Q @ K.T / sqrt(d_k)
scores = scores.masked_fill(causal_mask, float('-inf'))
attn = F.softmax(scores, dim=-1)
out = attn @ V

# -inf를 쓰는 이유:
# exp(-inf) = 0 → softmax 후 해당 위치 가중치 = 0
# 미래 토큰 정보가 완전히 차단
```

---

## 완전한 Transformer Block 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # 선형 변환 + head 분할
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        # Q, K, V: (B, num_heads, T, d_k)

        # Attention score
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ V  # (B, num_heads, T, d_k)

        # head 합치기
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up   = nn.Linear(d_model, d_ff, bias=False)
        self.W_down  = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.W_gate(x))  # SiLU = Swish
        up   = self.W_up(x)
        return self.W_down(gate * up)

class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block (LLaMA 스타일)"""
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn  = SwiGLU(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, mask=None):
        # Pre-LN: norm → attention → residual
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x
```

---

## Grouped Query Attention (GQA / MQA)

```
Multi-Head Attention의 KV cache 메모리 문제 해결

MHA (원래):
  Q heads: H개,  K heads: H개,  V heads: H개
  KV cache: H × d_k × seq_len × 2 bytes

MQA (Multi-Query Attention):
  Q heads: H개,  K heads: 1개,  V heads: 1개
  → KV cache H× 절약
  품질 약간 저하

GQA (Grouped-Query Attention, Ainslie et al., 2023):
  Q heads: H개,  K heads: G개,  V heads: G개  (G < H)
  Q를 G개 그룹으로 나눔 → 각 그룹이 하나의 KV 공유

  H=32 heads, G=8 groups:
    KV cache: 4× 절약 (MHA 대비)
    품질: MHA에 거의 근접 (MQA보다 훨씬 좋음)

사용:
  LLaMA-2-70B: GQA (G=8)
  LLaMA-3: GQA (G=8)
  Mistral: GQA (G=8)
  Gemma: MQA (G=1)
```

```python
class GroupedQueryAttention(nn.Module):
    """GQA: Q heads > K,V heads"""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads  # Q groups per KV head
        self.d_k = d_model // n_heads
        self.d_kv = d_model // n_kv_heads  # KV head dimension... actually same d_k

        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(n_heads * self.d_k, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, T, C = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # Q: (B, n_heads, T, d_k)

        K = self.W_k(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        # K, V: (B, n_kv_heads, T, d_k)

        # KV heads 확장: (B, n_kv_heads, T, d_k) → (B, n_heads, T, d_k)
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)

        # Standard attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ V

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)
```

---

## Embedding & 출력 레이어

### 임베딩 초기화와 weight tying
```python
class LLaMAModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, n_kv_heads, d_ff):
        super().__init__()
        # Token embedding
        self.embed = nn.Embedding(vocab_size, d_model)

        # Transformer blocks (with GQA)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model)

        # LM head: embedding과 weight 공유 (weight tying)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight 공유!

    def forward(self, input_ids):
        x = self.embed(input_ids) * math.sqrt(self.embed.embedding_dim)
        # sqrt 스케일링: embedding 크기 안정화

        # Causal mask 생성
        T = input_ids.shape[1]
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        mask = mask.to(input_ids.device)

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """Autoregressive generation with KV cache (conceptual)"""
        for _ in range(max_new_tokens):
            # Only process new context (in practice, use KV cache)
            logits = self.forward(input_ids)[:, -1, :]  # (B, vocab)

            # Temperature scaling
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k)
                min_val = topk_vals[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < min_val, float('-inf'))

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
```

### Weight Tying 이유
```
lm_head.weight = embed.weight

이유:
  입력 embedding: token → d_model 공간
  출력 embedding: d_model → token 예측
  두 행렬이 "같은 의미 공간"을 공유 → 파라미터 효율

효과:
  vocab_size × d_model 파라미터 절약
  LLaMA-2-7B (32K vocab, 4096 dim): 131M 파라미터 절약
  성능: 대부분의 경우 tying이 더 좋거나 동등
```

---

## 파라미터 수 정밀 계산

```
LLaMA-2-7B 실제 계산:

임베딩:
  Token embed: 32000 × 4096 = 131M
  (Position embed: RoPE → 파라미터 없음)

레이어당 (32개 레이어):
  Attention:
    Q: 4096 × 4096 = 16.7M
    K: 4096 × 1024 = 4.2M  (GQA: 8 KV heads × 128 d_k)
    V: 4096 × 1024 = 4.2M
    O: 4096 × 4096 = 16.7M
    소계: 41.9M

  FFN (SwiGLU):
    gate: 4096 × 11008 = 45.1M
    up:   4096 × 11008 = 45.1M
    down: 11008 × 4096 = 45.1M
    소계: 135.3M

  RMSNorm: 2 × 4096 ≈ 0M (무시)

  레이어당 총: 177.2M

전체:
  32 × 177.2M + 131M (embed) ≈ 5.8B + 131M ≈ 6.7B
  (실제 7B: 반올림 + 추가 구성요소)
```

---

## Cross-Attention (Encoder-Decoder)

```python
class CrossAttention(nn.Module):
    """Encoder-Decoder Cross Attention"""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)  # from decoder
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # from encoder
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # from encoder
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, decoder_state: torch.Tensor,
                encoder_output: torch.Tensor,
                encoder_mask: torch.Tensor = None) -> torch.Tensor:
        B, T_dec, _ = decoder_state.shape
        _, T_enc, _ = encoder_output.shape

        # Q from decoder, K/V from encoder
        Q = self.W_q(decoder_state).view(B, T_dec, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(encoder_output).view(B, T_enc, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(encoder_output).view(B, T_enc, self.num_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, H, T_dec, T_enc)

        if encoder_mask is not None:
            scores = scores.masked_fill(encoder_mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ V  # (B, H, T_dec, d_k)

        out = out.transpose(1, 2).contiguous().view(B, T_dec, -1)
        return self.W_o(out)
```

---

## Residual Stream 해석

```
Mechanistic Interpretability 관점:
  Transformer = 각 layer가 "residual stream"에 정보 추가

  x_{l+1} = x_l + Attn_l(x_l) + FFN_l(x_l)

  Attention layer:
    Query 위치에서 Value 위치로 정보 이동
    "Key-Query matching" = 패턴 감지

  FFN layer:
    위치별 독립적 처리
    "Fact recall": 지식을 KV 메모리처럼 저장
    W1 rows = keys (패턴 매칭)
    W2 cols = values (결과 기여)

  Residual connection:
    각 layer의 기여가 선형으로 추가
    → 각 head/layer의 역할 분석 가능 (Logit Lens)
```

---

## 생성 전략 (Decoding Strategies)

```python
def greedy_decode(logits: torch.Tensor) -> torch.Tensor:
    """Greedy: 가장 높은 확률 토큰 선택"""
    return logits.argmax(dim=-1)

def temperature_sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Temperature: 분포의 날카로움 조절"""
    # temperature < 1: 더 deterministic
    # temperature > 1: 더 diverse
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

def top_k_sample(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> torch.Tensor:
    """Top-K: k개 후보 중 샘플링"""
    # k개 이외는 -inf로 마스킹
    topk_vals, _ = torch.topk(logits, k)
    threshold = topk_vals[:, -1].unsqueeze(-1)
    filtered = logits.masked_fill(logits < threshold, float('-inf'))
    return temperature_sample(filtered, temperature)

def top_p_sample(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    """Top-P (Nucleus): 확률 합이 p가 될 때까지 포함"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # p 초과하는 위치 제거
    sorted_to_remove = cumulative_probs > p
    # 최소 1개는 포함 (shift right)
    sorted_to_remove[:, 1:] = sorted_to_remove[:, :-1].clone()
    sorted_to_remove[:, 0] = False

    # 원래 인덱스로 복원
    indices_to_remove = sorted_to_remove.scatter(1, sorted_indices, sorted_to_remove)
    filtered = logits.masked_fill(indices_to_remove, float('-inf'))
    return temperature_sample(filtered, temperature)

# 실용적 조합
def generate_token(logits: torch.Tensor,
                   temperature: float = 0.7,
                   top_k: int = 50,
                   top_p: float = 0.9) -> torch.Tensor:
    """Temperature + Top-K + Top-P 조합"""
    scaled = logits / temperature
    # Top-K 먼저
    topk_vals, _ = torch.topk(scaled, top_k)
    scaled = scaled.masked_fill(scaled < topk_vals[:, -1].unsqueeze(-1), float('-inf'))
    # Top-P 적용
    return top_p_sample(scaled, p=top_p, temperature=1.0)  # already scaled
```

---

## Further Questions

**Q. Attention의 시간·공간 복잡도는?**
> 시간: O(n²d), 공간: O(n²) — n은 시퀀스 길이. QK^T 행렬이 n×n이기 때문. Flash Attention으로 메모리 O(n)으로 감소 (HBM write/read 최소화).

**Q. Pre-LN vs Post-LN 차이와 장단점은?**
> Post-LN: 원 논문 방식, residual 후 LN, 학습 초기 불안정 (초기 gradient vanishing). Pre-LN: 현대 표준, residual 전 LN, gradient flow 안정, warmup 없어도 학습 가능. 단점: 표현력 약간 감소 (연구 중).

**Q. FFN이 하는 역할은?**
> 각 토큰 위치에서 독립적인 비선형 변환. Mechanistic interpretability 관점: "Key-Value Memory" — W1 rows = keys (패턴 감지), W2 columns = values (정보 기여). 사실적 지식을 MLP에 저장.

**Q. √d_k로 나누는 이유는?**
> d_k 차원 내적의 분산이 d_k → 표준편차 √d_k. 스케일링 없이는 softmax 입력이 너무 커서 포화 (gradient ≈ 0). 나눔으로 분산을 1로 정규화 → 적절한 attention weight 분포.

**Q. Weight Tying이 왜 효과적인가?**
> 입력 토큰 임베딩과 출력 LM head가 같은 의미 공간 사용. 파라미터 공유로 효율성 ↑. 수렴 속도 향상 (임베딩 학습 신호 증가). vocab × d_model만큼 파라미터 절약.

**Q. GQA는 MHA에서 무엇을 희생하는가?**
> KV 헤드 수를 줄여 각 그룹의 Q 헤드들이 같은 K/V를 공유. 희생: 헤드별 독립적인 KV 표현이 없어짐 → 표현력 약간 감소. 얻는 것: KV cache 크기 n_groups배 절약 → 더 큰 배치 처리 가능. 실험적으로 MHA 대비 품질 손실 < 1%.

**Q. Temperature 1.0 vs 0.0 vs >1.0의 효과는?**
> T=0: 완전 deterministic (greedy), 항상 같은 출력. T=1: 원래 분포에서 샘플링. T<1: 더 날카로운 분포 (높은 확률 토큰 더 선택), 일관성 ↑. T>1: 더 평평한 분포 (다양성 ↑), creative writing에 유용. 코딩/수학: T=0-0.2, 창작: T=0.7-1.2.
