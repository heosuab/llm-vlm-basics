# Decoder-only 아키텍처 & 주요 모델

## Decoder-only의 핵심 구조

```
구성:
  Embedding → [Transformer Decoder Block] × N → LM Head

Decoder Block (LLaMA-3 기준):
  x → RMSNorm → Self-Attention (GQA + RoPE) → x + (residual)
    → RMSNorm → FFN (SwiGLU)               → x + (residual)

사전학습 목표: Next Token Prediction (Causal LM)
  L = -Σ log P(x_t | x_{<t})

장점:
  생성에 최적화 (Autoregressive)
  사전학습 목표가 단순하고 확장성 높음
  In-context learning 능력

현재 대부분의 LLM (GPT, LLaMA, Mistral, Gemini, Claude 등)
```

---

## GPT 계열 진화

```
GPT-1 (2018): 12L, 117M, BooksCorpus
  - 최초의 대규모 언어 모델 사전학습 + 파인튜닝
  - Pre-LN + causal attention 표준화

GPT-2 (2019): 48L, 1.5B, WebText 40GB
  - "언어 모델은 멀티태스크 학습자"
  - zero-shot transfer 첫 시연
  - 1.5B 모델 공개 거부 (초기 AI safety 논쟁)

GPT-3 (2020): 96L, 175B, CommonCrawl+BookCorpus+WebText2
  - 스케일링 법칙 입증 (더 크게 → 더 좋음)
  - In-context learning (few-shot, 1-shot, zero-shot)
  - 파인튜닝 없이 다양한 태스크 수행

InstructGPT (2022): GPT-3 + RLHF
  - 더 적은 파라미터로 더 유용한 모델 가능
  - "Alignment" 개념 실용화

GPT-4 (2023):
  - MoE 아키텍처 (추정, 비공개)
  - 멀티모달 (GPT-4V)
  - 1T+ 파라미터 추정

GPT-4o (2024):
  - 음성/이미지/텍스트 통합
  - 더 빠른 추론
  - 128K 컨텍스트

o1/o1-mini (2024):
  - Chain-of-Thought reasoning 내재화
  - 긴 "생각" 후 답 (test-time compute scaling)

o3 (2025):
  - o1보다 강력한 reasoning
  - 과학/수학/코딩 SOTA
```

---

## LLaMA 계열 상세

### LLaMA-1 (Meta, 2023)

```
혁신:
  오픈소스 공개 (연구 목적)
  Chinchilla-optimal 방향: 작은 모델을 더 오래 학습

주요 개선 사항 (GPT 대비):
  Pre-RMSNorm: 각 sub-layer 입력을 정규화
    → Post-LN보다 학습 안정
    g(x) = x / sqrt(mean(x²) + ε) × γ

  SwiGLU activation:
    FFN: Linear → SwiGLU → Linear
    SwiGLU(x) = (xW₁) · σ(xW₃)  (element-wise)
    → GELU보다 성능 개선

  RoPE: Learned PE 대신
    → 외삽 가능, 파라미터 없음

데이터: CommonCrawl, C4, Github, Wikipedia, Books, ArXiv, StackExchange
  → 1T 토큰 (7B), 1.4T 토큰 (65B)

크기: 7B, 13B, 33B, 65B
```

### LLaMA-2 (Meta, 2023)

```
LLaMA-1 대비 개선:
  컨텍스트: 2K → 4K
  70B에만 GQA 적용 (다른 크기는 MHA 유지)
  데이터: 2T 토큰 (40% 더 많음)

Chat 버전:
  RLHF (PPO) 적용 → LLaMA-2-Chat
  Helpfulness + Safety 균형
  Ghost Attention: 시스템 프롬프트 유지 기법

상업적 사용 허가 (일부 제한 있음)
크기: 7B, 13B, 34B, 70B
```

### LLaMA-3 (Meta, 2024)

```
주요 개선:
  Tokenizer:
    128K vocab (BPE, tiktoken 기반)
    GPT-4 cl100k와 유사한 방식
    다국어/코드 더 효율적

  Architecture:
    GQA 모든 크기에 적용
    컨텍스트 8K (기본), 128K (확장)
    RoPE base: 500,000 (기본 10,000에서 50배↑)

  Data:
    15T+ 토큰
    코드, 수학 데이터 강화
    다국어 데이터 비율 증가

LLaMA-3.1 (2024.07):
  128K 컨텍스트 공식 지원
  405B 모델 공개 (최대 오픈소스)
  다국어 강화 (8개 언어)

LLaMA-3.2 (2024.09):
  멀티모달 버전 (11B, 90B Vision)
  경량 버전 (1B, 3B) — Edge 배포용
  Pruning + Distillation로 소형화

LLaMA-3.3 (2024.12):
  70B 성능 향상
  LLaMA-3.1-405B 수준을 70B로
```

### LLaMA 아키텍처 세부

```python
class LLaMADecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LLaMAAttention(config)   # GQA + RoPE
        self.mlp = LLaMAMLP(config)               # SwiGLU
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # Pre-RMSNorm + Self-Attention + Residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        # Pre-RMSNorm + FFN (SwiGLU) + Residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class LLaMAMLP(nn.Module):
    """SwiGLU FFN"""
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        # SwiGLU: gate * silu(up)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

---

## Mistral 계열

### Mistral-7B (2023)

```
혁신:
  Sliding Window Attention (SWA):
    window_size = 4096
    각 토큰이 자신과 이전 4096개만 attend
    복잡도: O(n·w) → 긴 시퀀스 효율

  Rolling Buffer KV Cache:
    Cache 크기를 window_size로 고정
    오래된 KV 슬롯에 새 KV 덮어씀
    메모리 고정 (시퀀스 길이 무관)

  GQA: 32 Q heads, 8 KV heads

  성능:
    LLaMA-2-13B보다 성능 우수 (7B 모델로)
    특히 추론 속도 빠름
```

### Mixtral-8x7B MoE (2024)

```
Sparse MoE:
  FFN 대신 8개의 Expert FFN
  각 토큰마다 2개 expert 선택 (라우팅)
  활성 파라미터: 12.9B (전체 46.7B)

성능:
  LLaMA-2-70B 수준, 속도는 12.9B 수준

자세한 내용: 03-architectures/03-moe-architecture.md
```

### Mistral-Nemo-12B (2024)

```
12B 파라미터 (Mixtral과 Mistral AI 협력)
Tekken 토크나이저: 131K vocab, 다국어 강화
128K 컨텍스트
GQA, Sliding Window Attention
```

---

## Gemma 계열 (Google, 2024)

### Gemma-1 (2024)

```
아키텍처 특징:
  Multi-Head Attention (MHA, not GQA 초기 버전)
  GeGLU activation: GELU 대신
    GeGLU(x) = (xW₁) · GELU(xW₃)
  Pre-RMSNorm + Post-RMSNorm (각 sub-layer 앞뒤)
  Vocab: 256K (SentencePiece BPE)
  Logit soft-capping:
    logit = tanh(logit / cap) × cap  (발산 방지)

크기: 2B, 7B
특징: 동일 크기 오픈소스 모델 중 강력한 성능
```

### Gemma-2 (2024)

```
주요 혁신:
  Sliding Window Attention + Full Attention 교번:
    홀수 레이어: Sliding Window (window=4096)
    짝수 레이어: Full Attention
    → 로컬 + 글로벌 정보 통합

  Knowledge Distillation:
    27B → 9B → 2B 순서로 증류
    큰 모델 지식을 작은 모델로

  Grouped Query Attention (GQA)

성능:
  2B: 7B 클래스 성능
  9B: 70B 클래스 성능 (효율적)
  27B: 당시 오픈소스 최고 수준

크기: 2B, 9B, 27B
```

---

## DeepSeek 계열 (DeepSeek AI, 2024)

### DeepSeek-V2 (2024)

```
혁신적 아키텍처 1: MLA (Multi-head Latent Attention)

문제: MHA의 KV Cache가 너무 큼
  KV Cache = 2 × num_layers × num_heads × d_head × seq_len

MLA 해결책:
  K, V를 저차원 잠재 벡터로 압축
  c_KV = W_c^KV · x  (압축, d_model → d_c, d_c << d_model)
  K = W^K · c_KV     (복원)
  V = W^V · c_KV     (복원)

Cache 전략:
  c_KV (압축된 형태)만 Cache에 저장
  → KV Cache 5.4× 감소 (MHA 대비)
  추론 시 압축 벡터로 K,V 재생성

또한 RoPE를 위한 특수 처리:
  RoPE는 위치 의존적이므로 압축 불가
  별도의 RoPE key를 추가 (작은 차원)

혁신적 아키텍처 2: DeepSeekMoE
  세분화된 Expert (Fine-grained):
    기존 MoE: 큰 expert 소수 선택
    DeepSeekMoE: 작은 expert 다수 선택 (2개→많은 수)
  Shared Expert: 항상 활성화되는 공통 expert
  → 더 효율적 지식 분배

크기: 236B 총 파라미터, 21B active
```

### DeepSeek-V3 (2024)

```
671B 총 파라미터, 37B active

혁신:
  FP8 Mixed Precision 학습:
    FP8 GEMM (행렬 곱셈)
    BF16 optimizer (정밀도 유지)
    → 메모리 절약 + 속도 향상
    학습 비용: ~$5.5M (매우 효율적)

  Multi-Token Prediction (MTP):
    일반: 다음 1토큰 예측
    MTP: 다음 N토큰 동시 예측 (보조 목표)
    → 향상된 학습 신호, 추론 속도 향상 (Speculative)

  Auxiliary-loss-free load balancing:
    MoE expert 균형을 loss 없이 달성
    특정 expert에 쏠림 방지

성능: GPT-4o, Claude-3.5-Sonnet과 경쟁
```

### DeepSeek-R1 (2025)

```
Reasoning 특화 오픈소스 모델

학습 과정:
  1. Cold Start SFT: 소량 고품질 CoT 데이터
  2. Reasoning GRPO: 수학/코딩 검증 보상
  3. Rejection Sampling SFT: 고품질 응답 선별
  4. GRPO: Helpfulness + Safety

핵심: 순수 RL로 CoT reasoning 자발적 획득
성능: OpenAI o1과 경쟁 (오픈소스)
```

---

## Qwen 계열 (Alibaba)

### Qwen-2.5 (2024)

```
아키텍처:
  Transformer Decoder, Pre-RMSNorm
  SwiGLU FFN
  GQA
  RoPE
  Tied embedding (vocab embed = lm_head)

특징:
  128K 컨텍스트
  152K vocab (영어+중국어 최적화)
  수학, 코딩 데이터 강화 (7T+ 토큰)
  다국어 (중국어 특히 우수)

특화 버전:
  Qwen2.5-Math: 수학 전문 (수학 데이터 특화 SFT)
  Qwen2.5-Coder: 코딩 전문 (코드 데이터 특화)
  QwQ-32B: Reasoning 특화 (DeepSeek-R1 스타일)

크기: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
```

---

## Phi 계열 (Microsoft)

```
철학: 고품질 합성 데이터로 소형 모델 강화

Phi-1 (2023):
  1.3B, 코딩 특화
  "교과서 수준" 코딩 데이터

Phi-1.5 (2023):
  1.3B, 일반 reasoning
  합성 "교과서 데이터"

Phi-2 (2023):
  2.7B, 범용
  당시 같은 크기에서 최강

Phi-3 (2024):
  Mini (3.8B): 모바일/엣지 배포
  Small (7B): 균형
  Medium (14B): 고성능
  합성 데이터 + 고품질 웹 데이터

Phi-3.5 (2024):
  MoE 버전 (Phi-3.5-MoE): 16×3.8B, 6.6B active
  Vision (Phi-3.5-V): 이미지 처리 추가
  128K 컨텍스트
```

---

## 아키텍처 설계 비교 심화

### Normalization 위치

```
Post-LN (원 Transformer):
  Attention(x) + x → LayerNorm → FFN
  문제: 깊은 모델에서 불안정 (초기 훈련)

Pre-LN (GPT-2, LLaMA):
  LayerNorm → Attention → x + Attention(norm(x))
  장점: 안정적 gradient, 더 깊은 모델 학습 가능
  주의: Pre-LN은 gradient가 input 방향으로 바이패스

Pre-RMSNorm (LLaMA, Mistral, etc.):
  RMSNorm 사용:
    g(x) = x / RMS(x) × γ
    RMS(x) = sqrt(mean(x²) + ε)
  LayerNorm과 유사하지만:
    평균 제거 없음 (더 빠름)
    파라미터 절약 (β 없음)

Double Normalization (Gemma-2):
  Pre-Norm + Post-Norm 둘 다 사용
  → 학습 안정성 최대화 (대신 느림)
```

### Activation Functions

```
GELU (GPT 계열):
  GELU(x) ≈ x · Φ(x)  (Φ: 정규분포 CDF)
  ReLU보다 부드러운 비선형

SwiGLU (LLaMA, Mistral):
  SwiGLU(x, W, V) = Swish(xW) ⊙ (xV)
  Swish(x) = x · σ(x)
  중간 차원: 2/3 × 4 × d_model (파라미터 수 유지)

GeGLU (Gemma):
  GeGLU(x, W, V) = GELU(xW) ⊙ (xV)

비교:
  SwiGLU/GeGLU: Gate 메커니즘으로 더 표현력 높음
  → 대부분의 현대 모델 채택
```

### FFN 차원

```
기본 FFN: hidden → 4×hidden → hidden
SwiGLU FFN: hidden → (8/3)×hidden (× 2 projections)
  → 파라미터 수 유지하면서 gate 추가
  실제: hidden → 11008 (7B, 4096 hidden)

MoE FFN: hidden → K개 expert 중 top-2 선택
  각 expert: 일반 FFN
  활성 파라미터: top-2 expert만
```

---

## 아키텍처 비교 요약

| 모델 | 크기 | Attention | Norm | FFN | PE | Context |
|------|------|-----------|------|-----|----|----|
| GPT-2 | 1.5B | MHA | Post-LN | GELU | Learned | 1K |
| GPT-3 | 175B | MHA | Pre-LN | GeLU | Learned | 2K |
| LLaMA-1 | 7-65B | MHA | Pre-RMSNorm | SwiGLU | RoPE | 2K |
| LLaMA-2 | 7-70B | MHA/GQA | Pre-RMSNorm | SwiGLU | RoPE | 4K |
| LLaMA-3 | 8-405B | GQA | Pre-RMSNorm | SwiGLU | RoPE(500K) | 8K/128K |
| Mistral-7B | 7B | GQA+SWA | Pre-RMSNorm | SwiGLU | RoPE | 8K |
| Gemma-2 | 2-27B | GQA+SWA | Pre+Post-RMSNorm | GeGLU | RoPE | 8K |
| Qwen2.5 | 0.5-72B | GQA | Pre-RMSNorm | SwiGLU | RoPE | 128K |
| DeepSeek-V3 | 671B | MLA+GQA | Pre-RMSNorm | MoE | RoPE | 128K |

---

## Further Questions

**Q. LLaMA가 GPT-3보다 소형에서 효율적인 이유?**
> Chinchilla 법칙 적용: 모델 크기보다 데이터를 더 많이 학습 (inference-optimal). 아키텍처 개선 (SwiGLU, Pre-RMSNorm, RoPE) + 더 많은 고품질 데이터. 오픈소스 커뮤니티 파인튜닝 효과.

**Q. GQA가 MHA보다 추론에서 빠른 이유?**
> KV Cache 크기 비례 감소 (heads 수 감소 → seq_len × layers × dtype × 2 × kv_heads × d_head). 메모리 대역폭 요구 감소 → Memory bandwidth가 병목인 decode 단계에서 특히 효과적. LLaMA-3-8B: 32 Q heads, 8 KV heads → 4× KV Cache 감소.

**Q. Pre-LN이 Post-LN보다 학습 안정적인 이유?**
> Post-LN: LayerNorm이 residual connection 후 → 초기 훈련 시 gradient가 매우 클 수 있음 → 발산. Pre-LN: 각 sub-layer 입력을 정규화 → gradient 흐름 안정적 (gradient bypass via residual). 깊은 모델 (100+ layers)에서 Pre-LN이 필수.

**Q. SwiGLU가 일반 FFN보다 좋은 이유?**
> Gate 메커니즘: 두 경로 중 하나가 다른 하나를 선택적으로 게이팅 → 더 표현력 높음. "Hard" ReLU 대비 부드러운 활성화 → gradient 흐름 원활. 경험적으로 비슷한 파라미터에서 SwiGLU > GELU > ReLU.

**Q. DeepSeek-V2의 MLA가 GQA보다 나은 이유?**
> GQA: K,V head 수만 줄임 (정보 손실 가능). MLA: K,V 전체를 저차원 잠재 벡터로 압축 → 학습을 통해 최적 압축. KV Cache를 압축 벡터 형태로 저장 → 5.4× 더 작음. 단점: 복원 연산 추가.
