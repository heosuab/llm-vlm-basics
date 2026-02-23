# Attention Mechanisms

## Scaled Dot-Product Attention 수학

```
입력:
  Q (Query):  (seq_len_q, d_k)
  K (Key):    (seq_len_k, d_k)
  V (Value):  (seq_len_k, d_v)

계산:
  Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

단계별:
  1. 유사도 계산: S = QKᵀ           (seq_q, seq_k)
  2. 스케일링:   S = S / √d_k
  3. 마스킹:    S[i,j] = -∞ if masked  (causal LM)
  4. 소프트맥스: A = softmax(S, dim=-1) (seq_q, seq_k)
  5. 가중합:    O = A · V             (seq_q, d_v)

√d_k 스케일링 이유:
  d_k가 크면 QKᵀ의 분산 = d_k
  → 분산이 크면 softmax에서 극단적 값 (기울기 소실)
  → √d_k로 나눠 분산 = 1 유지

  수식적으로: Var(q·k) = Σᵢ Var(qᵢ)Var(kᵢ) = d_k · 1 · 1 = d_k
  → (q·k)/√d_k의 분산 = 1
```

---

## Multi-Head Attention (MHA)

```
여러 "관점"으로 병렬 attention:
  head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

  W_i^Q: (d_model, d_k)  → d_k = d_model / num_heads
  W_i^K: (d_model, d_k)
  W_i^V: (d_model, d_v)  → d_v = d_model / num_heads

출력 결합:
  MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W^O
  W^O: (h·d_v, d_model)

파라미터 수:
  W^Q, W^K, W^V: 각 d_model × d_model (= h × d_k × d_model)
  W^O: d_model × d_model
  총: 4 × d_model²

효과:
  각 head가 다른 종류의 관계 학습
  Head 1: 인접 단어 관계
  Head 2: 장거리 의존성
  Head 3: 문법 구조
  등 전문화 발생
```

---

## MHA → MQA → GQA 진화

### MHA vs MQA vs GQA

```
문제: 디코딩 시 KV Cache가 메모리 대역폭 병목
  KV Cache 크기: 2 × num_heads × d_head × seq_len × num_layers × dtype

MHA (Multi-Head Attention):
  Q, K, V 모두 num_heads개
  KV Cache: 2 × num_heads × seq_len × d_head × layers

MQA (Multi-Query Attention) [Shazeer 2019]:
  Q: num_heads개
  K, V: 1개 (모든 Q head 공유)
  KV Cache: 2 × 1 × seq_len × d_head × layers → num_heads배 감소
  장점: 추론 속도 2~4× 향상
  단점: 표현력 약간 저하
  사용: PaLM, Falcon, Gemma (초기)

GQA (Grouped-Query Attention) [Ainslie et al. 2023]:
  Q: num_heads개
  K, V: G개 (= num_kv_heads)
  각 그룹의 Q head들이 K,V 공유

  G = 1:         MQA (하나 공유)
  G = num_heads: MHA (각각 독립)
  1 < G < num_heads: 중간 균형

  LLaMA-2-70B: 8개 head, G=8 (= num_heads/8=1개 KV head/group)
  LLaMA-3-8B: 32개 Q head, 8개 KV head (G=4)
  Mistral-7B: 32개 Q head, 8개 KV head
```

### GQA 구현
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,      # Q head 수 (예: 32)
        num_kv_heads: int,   # KV head 수 (예: 8)
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads  # 그룹당 Q head 수
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, num_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(num_heads * self.d_head, d_model, bias=False)

    def forward(self, x, attention_mask=None):
        B, T, D = x.shape

        # Projections
        q = self.q_proj(x)  # (B, T, num_heads * d_head)
        k = self.k_proj(x)  # (B, T, num_kv_heads * d_head)
        v = self.v_proj(x)  # (B, T, num_kv_heads * d_head)

        # Reshape
        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        # (B, num_heads, T, d_head)
        k = k.view(B, T, self.num_kv_heads, self.d_head).transpose(1, 2)
        # (B, num_kv_heads, T, d_head)
        v = v.view(B, T, self.num_kv_heads, self.d_head).transpose(1, 2)
        # (B, num_kv_heads, T, d_head)

        # KV를 Q head 수에 맞게 반복 (repeat_kv)
        # num_kv_heads → num_heads: 각 kv head를 num_groups번 반복
        k = k.unsqueeze(3).expand(-1, -1, -1, self.num_groups, -1)
        k = k.reshape(B, self.num_heads, T, self.d_head)
        v = v.unsqueeze(3).expand(-1, -1, -1, self.num_groups, -1)
        v = v.reshape(B, self.num_heads, T, self.d_head)

        # Scaled dot-product attention
        # (더 효율적: Flash Attention 또는 F.scaled_dot_product_attention 사용)
        scale = math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, h, T, T)

        if attention_mask is not None:
            attn = attn + attention_mask  # -inf 마스킹

        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, h, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)
```

---

## Flash Attention

### 문제: IO 병목
```
기존 Attention (Naive):
  1. S = Q @ K^T          [HBM 읽기: Q, K] [HBM 쓰기: S] → O(n²) IO
  2. P = softmax(S)       [HBM 읽기: S] [HBM 쓰기: P]  → O(n²) IO
  3. O = P @ V            [HBM 읽기: P, V] [HBM 쓰기: O] → O(n²) IO

  총 IO: ~O(n²) → n이 크면 SRAM ↔ HBM 왕복이 병목

H100 메모리 계층:
  SRAM (on-chip): ~250 KB, 19 TB/s
  HBM (off-chip): 80 GB, 3.35 TB/s
  비율: SRAM >> HBM 속도
  → 계산보다 메모리 이동이 훨씬 느림
```

### Flash Attention 핵심 아이디어
```
Tiling + Online Softmax:
  S 전체를 HBM에 저장하지 않고
  블록 단위로 SRAM에서 처리

알고리즘 개요:
  Q, K, V를 블록으로 나눔 (block_size에 맞게)
  For each Q block:
    For each K, V block:
      1. S_block = Q_block @ K_block^T / √d_k
      2. softmax 분모 누적 (running max, running sum)
      3. O_block += softmax_numerator * V_block
  마지막에 softmax 완료 (분모로 나눔)

Online Softmax Trick:
  softmax(x₁, x₂, ..., xₙ) = exp(xᵢ - max(x)) / Σⱼexp(xⱼ - max(x))

  블록별로 max와 sum을 누적:
    m_new = max(m_old, max_of_new_block)
    s_new = s_old * exp(m_old - m_new) + sum(exp(block - m_new))

결과:
  IO: O(n²) → O(n)  (HBM 접근 획기적 감소)
  메모리: O(n²) → O(n)  (n² attention matrix HBM 저장 불필요)
  속도: 2~4× 향상 (특히 긴 시퀀스)
  정확도: Exact (approximation 아님!)

Backward pass:
  gradient 계산 시 attention matrix 필요
  → 저장 대신 재계산 (recomputation)
  → 메모리 ↓, 연산 약간 ↑ (하지만 IO ↓↓↓)
```

### Flash Attention 2 & 3
```
Flash Attention 2 (2023):
  - 더 나은 작업 분배 (warps 간)
  - Backward pass 최적화
  - 2× 추가 속도 향상

Flash Attention 3 (2024):
  - H100 Hopper GPU 전용
  - WGMMA (Warp Group Matrix Multiply Accumulate)
  - TMA (Tensor Memory Accelerator) 활용
  - FP8 precision 지원
  - 1.5~2.0× Flash Attention 2 대비 향상
  - H100에서 ~750 TFLOPs/s (이론치 989 TFLOPs/s의 75%)
```

---

## Causal (Auto-regressive) Attention

```
LLM의 핵심: 미래 토큰을 보지 않음

Causal Mask:
  attention_mask[i, j] = 0    if j <= i (과거/현재만 attend)
                        = -∞  if j > i  (미래 attend 금지)

구현:
  # 상삼각 행렬 (대각선 위) = -inf
  mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1)
  attn_scores = attn_scores + mask
  attn_weights = softmax(attn_scores, dim=-1)

효과:
  j > i인 위치: softmax(-∞) = 0 → 미래 attention 없음
  Autoregressive 생성 가능
  각 위치가 자신과 이전 토큰만 보는 구조

Prefill (Prompt 처리):
  Causal mask로 한 번에 전체 처리 (병렬)
  Flash Attention으로 효율화

Decode (생성):
  새 토큰 1개만 생성
  이전 KV cache 재사용
  KV Cache 크기 = num_layers × 2 × num_heads × d_head × seq_len
```

---

## Sparse Attention

### Sliding Window Attention
```
각 토큰이 주변 w개 토큰만 attend
복잡도: O(n·w) → 긴 시퀀스에 효율적

Mistral-7B: window_size=4096

한계: 멀리 있는 토큰 정보는 간접적으로만 전달
     (레이어 깊어질수록 넓어지는 receptive field)
```

### Dilated Attention (LongNet, 2023)
```
거리에 따라 간격(dilation)이 다른 attention:
  가까운 토큰: 촘촘하게 (dilation=1)
  먼 토큰: 드문드문 (dilation=4, 8, ...)

복잡도: O(n log n)
장점: 극단적 긴 컨텍스트 처리 (1B 토큰)
```

### Longformer Attention
```
두 가지 attention 조합:
  1. Local window attention: 모든 토큰에 적용
  2. Global attention: 특별 [CLS] 토큰은 모든 토큰과 attend

복잡도: O(n)
사용: 긴 문서 분류, QA
```

### BigBird Attention
```
Longformer 확장:
  1. Local attention (window)
  2. Global tokens
  3. Random attention (임의 연결)

이론: 이 조합이 완전한 Transformer와 동등한 표현력 보장
```

---

## Attention in Long Context

### Ring Attention
```
시퀀스 병렬화 (여러 GPU):
  매우 긴 시퀀스를 여러 GPU에 분산

  GPU 0: tokens 0~N/4
  GPU 1: tokens N/4~N/2
  GPU 2: tokens N/2~3N/4
  GPU 3: tokens 3N/4~N

  Ring 통신:
    각 GPU가 다음 GPU에 KV 블록 전달 (ring 형태)
    자신의 Q와 모든 KV에 대해 attention 계산

Flash Attention + Ring Attention:
  각 GPU 내에서 Flash Attention
  GPU 간 Ring 통신으로 전체 시퀀스 처리
  → 1M+ 토큰 처리 가능

Ulysses (DeepSpeed):
  시퀀스를 head 차원으로 분산
  All-to-All 통신으로 재배치
  Ring Attention과 보완적 사용
```

### Chunked Attention
```
긴 컨텍스트를 청크로 나눠 점진적 처리:
  청크 크기 내에서 Flash Attention
  KV Cache로 이전 청크 정보 유지

xFormers, vLLM에서 구현
메모리 피크 감소 (특히 Prefill 단계)
```

---

## KV Cache 최적화

```
기본 KV Cache:
  생성 시 이전 K, V 재사용
  메모리: 2 × layers × heads × d_head × seq_len × precision
  예: LLaMA-3-8B, 32 layers, 8 kv heads, 128 d_head, BF16
    = 2 × 32 × 8 × 128 × seq_len × 2 bytes
    = 131,072 bytes × seq_len
    = 128KB/token × seq_len

MQA/GQA로 KV Cache 감소:
  MHA (32 heads): 기준
  MQA (1 kv head): 32× 감소
  GQA (8 kv heads): 4× 감소

자세한 내용: 04-inference-engineering/01-kv-cache.md
```

---

## Attention Sink (StreamingLLM, 2023)

```
관찰: 첫 번째 토큰(BOS, [CLS])이
      내용과 무관하게 높은 attention weight 받음

이유: "Attention Sink" 역할
  모든 위치에서 attention 합 = 1 (softmax 제약)
  → 유효하지 않은 위치에도 값 할당 필요
  → BOS 토큰이 "쓰레기통" 역할

영향:
  Sliding window attention + BOS 제거 → 성능 급락
  BOS 유지 → 안정

StreamingLLM 활용:
  처음 몇 개 sink 토큰 + 최근 W개 토큰만 유지
  → 무한 컨텍스트 스트리밍 처리 가능

AttentionSink 이론:
  Anchor 역할: 모든 레이어에서 공통 참조점 제공
  → 제거 시 attention distribution 불안정
```

---

## Softmax Alternatives

### Sigmoid Attention (Apple, 2024)
```
Attention = sigmoid(QKᵀ/√d) · V

특징:
  각 query-key 쌍 독립적 평가 (row sum = 1 제약 없음)
  병렬화에 유리

한계:
  정규화 없음 → 학습 초기 불안정 가능성
```

### Differential Attention (Microsoft, 2024)
```
두 softmax attention의 차이:
  Attn = softmax(Q₁K₁ᵀ/√d) - λ·softmax(Q₂K₂ᵀ/√d)

  λ: 학습 가능한 파라미터 (초기 0 근처)

직관:
  첫 번째: signal + noise
  두 번째: noise (다른 관점에서)
  차이: signal만 남음

효과:
  Attention noise 감소
  Hallucination 감소 경향
  긴 컨텍스트 "Lost in the Middle" 문제 완화
```

---

## Cross-Attention

```
Encoder-Decoder 구조:
  Q = Decoder hidden states    (target sequence)
  K, V = Encoder output        (source sequence)
  → Decoder가 Encoder를 참조하여 생성

  번역: 한국어 → 영어 생성 시 한국어 인코딩 참조
  요약: 긴 문서 → 요약 생성 시 원문 참조

VLM:
  Q = Text tokens
  K, V = Vision features (image patches)
  → 텍스트 생성 시 이미지 참조 (Flamingo, LLaMA-3.2 Vision)
```

---

## Attention Head 전문화

```
Induction Heads [Olsson et al., 2022]:
  패턴 복사: [A][B]...[A] → [B] 예측
  In-context learning의 핵심 메커니즘
  → 학습 중 갑자기 등장 (phase transition)

Name Mover Heads:
  특정 이름/개체 이동
  "Harry went to ...", "Harry" → attention to "Harry"

Negative Name Mover Heads:
  반대 역할 (Inhibitory heads)
  특정 토큰 억제

Previous Token Heads:
  직전 토큰에 attention
  copy, repetition 태스크

Positional Heads:
  상대 위치 정보 처리

→ 다양한 head들의 조합이 복잡한 언어 처리 가능
```

---

## Further Questions

**Q. MHA vs GQA vs MQA 각각 언제 사용?**
> 품질 최우선: MHA (연구용, 학술). 추론 효율 최우선: MQA (낮은 메모리 대역폭 환경). 균형: GQA (대부분의 현대 모델). LLaMA-3, Mistral 등 거의 모든 최신 오픈소스: GQA (num_kv_heads = num_heads / 4 또는 8).

**Q. Flash Attention의 핵심 아이디어는?**
> IO-bound 문제 해결. 기존: n² attention matrix를 HBM에 쓰고 읽는 왕복. Flash Attention: 블록 단위로 SRAM에서 처리 → HBM 접근 O(n)으로 감소. 수학적으로 동일 결과 (exact), 속도 2~4×, 메모리 O(n). Backward는 재계산으로 메모리 절약.

**Q. Causal mask를 어떻게 구현하나?**
> QKᵀ 행렬(상삼각 부분)에 -∞ 더한 후 softmax. torch.triu(torch.ones(T,T)*float('-inf'), diagonal=1). -∞를 softmax에 넣으면 exp(-∞)=0 → attention weight = 0. F.scaled_dot_product_attention(is_causal=True)로 자동 처리.

**Q. Sliding Window Attention의 한계는?**
> 로컬 window 밖 정보를 직접 attend 불가. 레이어를 쌓으면 receptive field 넓어지지만 모든 토큰이 서로를 직접 보지 못함. 문서 시작의 중요한 정보 (introduction, context)에 긴 응답 생성 시 접근 어려움 → Attention Sink로 보완.

**Q. GQA에서 K,V를 왜 repeat하는가?**
> 수학적으로 num_kv_heads개 K,V를 num_heads개 Q에 맞추려면 각 kv head를 num_groups = num_heads/num_kv_heads번 복사. 실제로는 repeat_kv 함수로 구현 (expand + reshape). 메모리: 원본 KV만 저장 → 계산 시만 expand → KV Cache는 num_kv_heads 기준으로 작게 유지.
