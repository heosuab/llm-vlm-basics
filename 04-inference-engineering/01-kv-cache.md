# KV Cache & 메모리 최적화

## KV Cache 기본

### 왜 필요한가?
```
Autoregressive 생성: 토큰을 하나씩 생성
각 step에서 전체 시퀀스의 K, V 재계산 → 중복 연산 O(n²)

KV Cache:
  이전 step의 K, V 저장 → 재사용
  각 step: 새 토큰의 K, V만 계산 후 캐시에 추가
  → O(n²) → O(n) per step

메모리 사용:
  KV cache 크기 = 2 × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes

  예: LLaMA-3-8B (32 layers, 8 kv_heads, head_dim=128):
    = 2 × 32 × 8 × 128 × 1000 × 2 bytes = ~131MB per sequence (1000 tokens)
    긴 시퀀스 (32K tokens): 131MB × 32 = ~4.2GB per sequence

  LLaMA-2-7B (MHA, 32 kv_heads):
    = 2 × 32 × 32 × 128 × 1000 × 2 = ~524MB per sequence
    → GQA로 4× 감소!
```

### KV Cache 구현

```python
class KVCache:
    """단순 KV Cache 구현"""
    def __init__(self, max_seq_len: int, num_layers: int,
                 num_kv_heads: int, head_dim: int, dtype=torch.float16):
        self.max_seq_len = max_seq_len
        self.seq_len = 0

        # [num_layers, 2, num_kv_heads, max_seq_len, head_dim]
        self.cache = torch.zeros(
            num_layers, 2, num_kv_heads, max_seq_len, head_dim, dtype=dtype
        )

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple:
        """새 K, V를 캐시에 추가하고 전체 캐시 반환"""
        seq_len = k.shape[2]  # 현재 토큰 수 (Prefill: n, Decode: 1)

        # 새 K, V 저장
        self.cache[layer_idx, 0, :, self.seq_len:self.seq_len + seq_len] = k
        self.cache[layer_idx, 1, :, self.seq_len:self.seq_len + seq_len] = v

        if seq_len > 1:
            self.seq_len += seq_len  # Prefill 완료

        # 전체 캐시 반환 (현재까지 생성된 시퀀스)
        cached_k = self.cache[layer_idx, 0, :, :self.seq_len + (1 if seq_len == 1 else 0)]
        cached_v = self.cache[layer_idx, 1, :, :self.seq_len + (1 if seq_len == 1 else 0)]

        if seq_len == 1:
            self.seq_len += 1  # Decode 단계

        return cached_k, cached_v

def attention_with_cache(q, k, v, cache: KVCache, layer_idx: int):
    """KV Cache를 활용한 Attention"""
    # Cache 업데이트
    k_full, v_full = cache.update(layer_idx, k, v)

    # Attention 계산 (q: 새 토큰, k_full/v_full: 전체 시퀀스)
    d_head = q.shape[-1]
    attn_scores = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(d_head)
    attn_probs = torch.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_probs, v_full)
```

---

## Prefill vs Decode 단계

```
Prefill (Prompt 처리):
  - 입력 전체를 한번에 병렬 처리
  - Attention: 시퀀스 내 모든 토큰 간 (full causal attention)
  - Compute-bound: 배치 행렬 곱 → GPU FLOP 활용
  - 속도: tokens/sec 높음, 하지만 TTFT에 직결

Decode (생성):
  - 하나씩 토큰 생성 (autoregressive)
  - Attention: 새 토큰 q ↔ 캐싱된 K, V
  - Memory-bandwidth-bound: 매 step마다 KV cache 전체 읽기
  - 속도: tokens/sec 낮음 (HBM 대역폭 병목)

측정 지표:
  TTFT (Time To First Token): Prefill 완료 시간 → 사용자 체감 반응성
  TPOT (Time Per Output Token): 각 토큰 생성 시간 → 생성 속도
  E2E Latency: TTFT + TPOT × output_length

Decode bottleneck 분석:
  LLaMA-3-8B BF16 (A100 80GB):
  KV Cache 읽기 per step = 2 × 32 layers × 8 heads × 128 × seq_len × 2 bytes
  seq_len=4096: ~134MB/step
  A100 HBM 대역폭: 2TB/s
  → 최대 ~15,000 steps/sec 이론치 (실제 절반 수준)
```

---

## PagedAttention (vLLM, 2023)

### 문제: Memory Fragmentation

```python
# 기존 방식 (연속 메모리 사전 할당)
max_seq_len = 4096
# 각 요청에 4096 토큰 공간 예약 (실제 사용: 50~2000)
# → 내부 단편화: 예약 메모리의 50~90% 낭비
# → 외부 단편화: 남은 연속 공간 없어 새 요청 거절

# GPU 메모리 이용률: 20~40% (낭비 심각)
```

### PagedAttention 해결책

```python
class PagedKVCache:
    """블록 기반 비연속 KV Cache (PagedAttention 간략 구현)"""

    def __init__(self, num_blocks: int, block_size: int,
                 num_layers: int, num_kv_heads: int, head_dim: int):
        self.block_size = block_size  # 블록당 토큰 수 (보통 16)
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # 물리 메모리 풀: [num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim]
        self.cache = torch.zeros(
            num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim
        )
        self.free_blocks = list(range(num_blocks))  # 사용 가능한 블록 목록

        # 각 시퀀스의 논리→물리 블록 매핑
        self.block_table: dict[int, list[int]] = {}  # seq_id → [physical_block_ids]

    def allocate(self, seq_id: int, num_tokens: int) -> None:
        """시퀀스에 필요한 블록 할당"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError("OOM: 사용 가능한 블록 없음")

        self.block_table[seq_id] = [
            self.free_blocks.pop() for _ in range(num_blocks_needed)
        ]

    def append_token(self, seq_id: int, layer_idx: int,
                     k: torch.Tensor, v: torch.Tensor, token_pos: int) -> None:
        """토큰을 해당 시퀀스의 논리 위치에 저장"""
        block_idx = token_pos // self.block_size
        offset = token_pos % self.block_size
        physical_block = self.block_table[seq_id][block_idx]

        self.cache[physical_block, 0, layer_idx, :, offset] = k
        self.cache[physical_block, 1, layer_idx, :, offset] = v

    def free(self, seq_id: int) -> None:
        """시퀀스 완료 후 블록 반환"""
        if seq_id in self.block_table:
            self.free_blocks.extend(self.block_table.pop(seq_id))
```

### Copy-on-Write (빔 서치/병렬 샘플링)

```python
# 두 시퀀스가 같은 물리 블록 공유 (prefix 공유)
# 한쪽이 새 토큰 추가 시 → 새 블록에 복사 후 쓰기

def copy_on_write(paged_cache: PagedKVCache, src_seq_id: int, dst_seq_id: int):
    """prefix 공유: 같은 블록 테이블 복사"""
    # reference counting으로 공유 블록 추적
    shared_blocks = paged_cache.block_table[src_seq_id][:-1]  # 마지막 빼고 공유
    last_block = paged_cache.free_blocks.pop()  # 새 블록 할당
    # 마지막 블록 내용 복사
    paged_cache.cache[last_block] = paged_cache.cache[paged_cache.block_table[src_seq_id][-1]]

    paged_cache.block_table[dst_seq_id] = shared_blocks + [last_block]
```

---

## Prefix Caching

### Radix Tree 기반 (vLLM / SGLang)

```python
class RadixTreePrefixCache:
    """
    Radix Tree (Trie)로 prefix 자동 탐지 및 캐싱
    vLLM은 해시 기반, SGLang은 Radix Tree 기반
    """

    def __init__(self):
        self.root = {"children": {}, "block_ids": [], "ref_count": 0}

    def match_prefix(self, token_ids: list[int]) -> tuple[int, list]:
        """최장 공통 prefix 찾기"""
        node = self.root
        matched_len = 0
        block_ids = []

        for token in token_ids:
            if token not in node["children"]:
                break
            node = node["children"][token]
            matched_len += 1
            block_ids.extend(node.get("block_ids", []))

        return matched_len, block_ids  # (매칭된 길이, 재사용 가능한 블록들)

    def insert(self, token_ids: list[int], new_block_ids: list) -> None:
        """새 시퀀스의 KV를 트리에 삽입"""
        node = self.root
        for i, token in enumerate(token_ids):
            if token not in node["children"]:
                node["children"][token] = {
                    "children": {},
                    "block_ids": [new_block_ids[i // 16]] if i % 16 == 0 else [],
                    "ref_count": 0
                }
            node = node["children"][token]

# 사용 예시 (RAG 시스템에서)
cache = RadixTreePrefixCache()

# 첫 번째 요청: 긴 시스템 프롬프트 + 질문1
tokens_1 = [101, 200, 201, 202, ..., 999]  # 시스템프롬프트 + 질문1
matched, blocks = cache.match_prefix(tokens_1)  # 처음엔 0 매칭

# 두 번째 요청: 같은 시스템 프롬프트 + 질문2
tokens_2 = [101, 200, 201, 202, ..., 888]  # 시스템프롬프트는 동일!
matched, blocks = cache.match_prefix(tokens_2)  # prefix 매칭! 재계산 불필요
```

### Prefix Caching 효과

```python
# 실제 워크로드에서의 효과
# RAG: 같은 retrieved context + 다른 질문
# Few-shot: 같은 예시들 + 다른 입력
# 챗봇: 같은 system prompt + 다른 user message

# vLLM에서 활성화
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    enable_prefix_caching=True,  # prefix caching 활성화
)

# 같은 system prompt를 여러 요청에서 공유
system_prompt = "You are a helpful assistant..." * 100  # 긴 시스템 프롬프트
questions = ["What is AI?", "Explain RAG", "What is RLHF?"]

from vllm import SamplingParams
sampling_params = SamplingParams(temperature=0.7, max_tokens=200)

outputs = []
for q in questions:
    prompt = f"{system_prompt}\n\nQuestion: {q}\nAnswer:"
    # 첫 번째 요청: prefix 계산 후 캐싱
    # 두 번째 이후: prefix KV cache 재사용
    output = llm.generate([prompt], sampling_params)
    outputs.append(output[0].outputs[0].text)

# 로그에서 prefix cache hit rate 확인
# 이상적: 50~90% hit rate (긴 공통 prefix일수록 높음)
```

---

## Speculative Decoding

### 수학적 원리

```python
import torch
import torch.nn.functional as F

def speculative_decoding(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    gamma: int = 5,          # draft 토큰 수
    max_new_tokens: int = 200
) -> torch.Tensor:
    """
    Speculative Decoding:
    1. draft_model로 gamma개 토큰 빠르게 생성
    2. target_model로 gamma+1개 토큰 한번에 검증
    3. 확률적 수락/거절 (분포 보장)
    """

    generated = input_ids.clone()

    while len(generated[0]) - len(input_ids[0]) < max_new_tokens:
        # 1. Draft: gamma개 토큰 빠르게 생성
        draft_tokens = []
        draft_probs = []
        draft_input = generated.clone()

        for _ in range(gamma):
            with torch.no_grad():
                draft_logits = draft_model(draft_input).logits[:, -1, :]
            draft_prob = F.softmax(draft_logits, dim=-1)
            token = torch.multinomial(draft_prob, 1)
            draft_tokens.append(token)
            draft_probs.append(draft_prob[0, token.item()].item())
            draft_input = torch.cat([draft_input, token], dim=1)

        # 2. Verify: target model로 gamma+1개 한번에 검증 (1 forward pass)
        verify_input = torch.cat([generated] + [t for t in draft_tokens], dim=1)
        with torch.no_grad():
            target_logits = target_model(verify_input).logits
        # target_probs: generated 이후부터 gamma+1개 위치
        target_probs = F.softmax(target_logits[:, len(generated[0])-1:-1, :], dim=-1)

        # 3. Accept/Reject
        accepted = 0
        for i in range(gamma):
            draft_token = draft_tokens[i].item()
            q = draft_probs[i]      # draft 확률
            p = target_probs[0, i, draft_token].item()  # target 확률

            # 수락 확률: min(1, p/q)
            accept_prob = min(1.0, p / (q + 1e-8))
            if torch.rand(1).item() < accept_prob:
                generated = torch.cat([generated, draft_tokens[i]], dim=1)
                accepted += 1
            else:
                # 거절: target 분포에서 보정 샘플링
                # adjusted: target - draft * accept_prob (음수 제거)
                adjusted = torch.clamp(
                    target_probs[0, i] - q * torch.ones_like(target_probs[0, i]),
                    min=0
                )
                adjusted = adjusted / adjusted.sum()
                token = torch.multinomial(adjusted.unsqueeze(0), 1)
                generated = torch.cat([generated, token], dim=1)
                break

        if accepted == gamma:
            # 모든 draft 토큰 수락: bonus token (gamma+1 번째 target 샘플)
            bonus_prob = F.softmax(target_logits[:, -1, :], dim=-1)
            bonus_token = torch.multinomial(bonus_prob, 1)
            generated = torch.cat([generated, bonus_token], dim=1)

    return generated
```

### Speculative Decoding 변형들

```
Medusa (Cai et al., 2024):
  Target 모델에 여러 prediction head 추가
  head_i: i+1번째 미래 토큰 예측
  → 별도 draft 모델 불필요
  → draft head 학습 비용 (약 1% 추가 파라미터)

EAGLE (Li et al., 2024):
  Draft 모델이 target feature를 직접 사용
  feature = target의 마지막 hidden state 공유
  → 더 높은 acceptance rate (>80%)
  → draft 품질 극적 향상

SpecTr (헬퍼 시퀀스):
  여러 draft 모델 → Tree 구조로 후보 생성
  Tree Attention으로 동시 검증

Lookahead Decoding (Fu et al., 2024):
  n-gram 캐시로 draft 생성
  모델 기반 draft 불필요
  단순, 어느 모델에나 적용 가능
```

---

## MLA (Multi-head Latent Attention) — DeepSeek-V2

```python
class MLAAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2)"""

    def __init__(self, d_model: int, num_heads: int, d_kv_lora: int = 512, d_rope: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_kv_lora = d_kv_lora  # KV 압축 차원 (훨씬 작음)
        self.d_rope = d_rope  # RoPE 차원

        # KV 압축 행렬
        self.W_DKV = nn.Linear(d_model, d_kv_lora)  # 압축: d_model → d_kv_lora
        self.W_UK = nn.Linear(d_kv_lora, num_heads * self.d_head)  # 복원
        self.W_UV = nn.Linear(d_kv_lora, num_heads * self.d_head)  # 복원

        # Q 압축 (DeepSeek-V2에서 Q도 압축)
        self.W_DQ = nn.Linear(d_model, 1536)
        self.W_UQ = nn.Linear(1536, num_heads * self.d_head)

        # RoPE 전용 차원 (비압축)
        self.W_KR = nn.Linear(d_model, d_rope)
        self.W_QR = nn.Linear(1536, num_heads * d_rope)

        self.W_O = nn.Linear(num_heads * self.d_head, d_model)

    def forward(self, x: torch.Tensor, rotary_emb, kv_cache=None):
        B, T, _ = x.shape

        # 압축된 KV 계산 (이것만 KV Cache에 저장!)
        c_kv = self.W_DKV(x)  # [B, T, d_kv_lora]

        if kv_cache is not None:
            # Decode: 누적된 c_kv 불러오기
            c_kv_full = torch.cat([kv_cache, c_kv], dim=1)
        else:
            c_kv_full = c_kv

        # KV 복원
        K = self.W_UK(c_kv_full).view(B, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_UV(c_kv_full).view(B, -1, self.num_heads, self.d_head).transpose(1, 2)

        # Q 계산
        c_q = self.W_DQ(x)
        Q = self.W_UQ(c_q).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        # Attention
        scale = self.d_head ** -0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)

        return self.W_O(out.transpose(1, 2).reshape(B, T, -1)), c_kv

# KV Cache 절약:
# 기존 GQA: num_kv_heads × d_head × 2 (K+V) = 8 × 128 × 2 = 2048 per layer
# MLA: d_kv_lora = 512 per layer
# 절약: 4× (실제 DeepSeek-V2: 93.3% 절약)
```

---

## 메모리 최적화 전략 종합

### H2O (Heavy-Hitter Oracle)

```python
class H2OKVCache:
    """중요한 KV만 유지 (Heavy-Hitter Oracle)"""

    def __init__(self, heavy_budget: int, recent_budget: int):
        """
        heavy_budget: attention을 많이 받은 토큰 수
        recent_budget: 최근 토큰 수 (항상 유지)
        """
        self.heavy_budget = heavy_budget
        self.recent_budget = recent_budget
        self.accumulated_attention = None  # 누적 attention score

    def evict(self, attention_weights: torch.Tensor,
              cache: torch.Tensor) -> torch.Tensor:
        """
        attention_weights: [batch, heads, seq_len] 현재 attention scores
        예산 초과 시 중요도 낮은 토큰 제거
        """
        # 누적 attention score 업데이트
        if self.accumulated_attention is None:
            self.accumulated_attention = attention_weights.mean(dim=1)
        else:
            self.accumulated_attention += attention_weights.mean(dim=1)

        total_budget = self.heavy_budget + self.recent_budget
        if cache.shape[2] <= total_budget:
            return cache

        # Heavy Hitter 선택 (처음 ~ heavy_budget)
        # Recent 유지 (마지막 recent_budget개)
        heavy_scores = self.accumulated_attention[:, :-self.recent_budget]
        heavy_topk = heavy_scores.topk(self.heavy_budget, dim=-1).indices.sort().values

        # 선택된 KV만 유지
        recent_indices = torch.arange(cache.shape[2] - self.recent_budget, cache.shape[2])
        keep_indices = torch.cat([heavy_topk[0], recent_indices])
        return cache[:, :, keep_indices, :]
```

### SnapKV

```python
def snap_kv_select(attention_weights: torch.Tensor, kv_cache: torch.Tensor,
                   observation_window: int = 64, budget: int = 256) -> torch.Tensor:
    """
    Prefill에서 관찰 window의 attention으로 중요한 KV 선택
    Decode 시 고정 유지 (오버헤드 없음)
    """
    # Observation window: 마지막 W 토큰의 attention 패턴
    obs_attn = attention_weights[:, :, -observation_window:, :]  # [B, H, W, L]
    importance = obs_attn.mean(dim=(0, 1, 2))  # 각 위치의 평균 중요도

    # 상위 budget개 선택 (recent window 항상 포함)
    non_recent = importance[:-observation_window]
    top_k = non_recent.topk(min(budget, len(non_recent))).indices.sort().values
    recent_indices = torch.arange(len(importance) - observation_window, len(importance))
    keep_indices = torch.cat([top_k, recent_indices])

    # 선택된 KV만 유지
    return kv_cache[:, :, keep_indices, :]
```

---

## 추론 시스템 설계

### Chunked Prefill

```python
class ChunkedPrefillScheduler:
    """Prefill을 청크로 나눠 Decode 요청과 교번"""

    def __init__(self, chunk_size: int = 512):
        self.chunk_size = chunk_size
        self.prefill_queue = []  # (seq_id, remaining_tokens)
        self.decode_queue = []   # (seq_id, current_len)

    def schedule_batch(self) -> dict:
        """한 iteration에 처리할 배치 결정"""
        batch = {"prefill_chunks": [], "decode_seqs": []}

        # Decode 요청 우선 (레이턴시 민감)
        for seq_id, current_len in self.decode_queue[:32]:  # 최대 32개
            batch["decode_seqs"].append(seq_id)

        # 남은 compute budget으로 prefill chunk 처리
        remaining_budget = self.chunk_size
        for i, (seq_id, remaining) in enumerate(self.prefill_queue):
            chunk = min(remaining, remaining_budget)
            batch["prefill_chunks"].append((seq_id, chunk))
            remaining_budget -= chunk
            if remaining_budget <= 0:
                break

        return batch
```

### Disaggregated Serving

```
전통적 서빙: 단일 서버에서 Prefill + Decode 모두 처리
  문제: Prefill (compute-bound)와 Decode (memory-bound)의 최적 조건이 다름
  → 두 단계 모두 최적화 불가

Disaggregated Serving:
  Prefill 서버 (P):
    강한 compute (A100 High-Mem for big batches)
    Prefill 완료 → KV cache를 Decode 서버로 전송 (RDMA)

  Decode 서버 (D):
    고대역폭 메모리 (H200 141GB)
    많은 동시 시퀀스 처리

통신:
  KV cache 전송: RDMA over InfiniBand (단방향)
  크기: 수십MB ~ 수백MB per sequence
  → 병렬 Prefill 가능 (여러 P서버 → 단일 D서버)

이점:
  P서버: Prefill에 최적화된 GPU/설정
  D서버: KV cache 최대화에 최적화
  독립적 스케일: 처리량 변화에 각 서버 독립적 확장

구현: DistServe, Mooncake (월식 ByteDance), PrefillProxy
```

### 추론 클러스터 설계

```
소규모 (7-8B 모델):
  단일 A100 80GB or H100 80GB → 충분
  TP=1 (오버헤드 없음)
  vLLM 단일 서버

중규모 (70B 모델):
  2× H100 (TP=2, NVLink)
  또는 1× H200 141GB (INT4로 ~35GB)

대규모 (405B 모델):
  8× H100 (TP=8, NVLink 내부)
  또는 4× H100 (AWQ INT4로 ~200GB)

API 서비스 (고가용성):
  Load Balancer → 여러 Replica (각 위 구성)
  Auto-scaling (KEDA): 큐 깊이 기반
  Health check: /health 엔드포인트
  모델 캐시 공유: 여러 서버가 같은 모델 파일 사용 (NFS/S3)
```

---

## Flash Decoding (긴 시퀀스 Decode 최적화)

```
문제: Decode 단계에서 KV cache가 너무 커 단일 GPU 처리 불가능
  seq_len=100K, 32 layers: 수십 GB KV cache

Flash Decoding:
  KV cache를 여러 청크로 병렬 처리
  각 청크에 대해 partial softmax 계산
  reduction 단계: partial 결과 통합 (log-sum-exp trick)

  vs Flash Attention (학습): 시퀀스 차원 병렬화
  vs Flash Decoding (추론): KV cache 청크 차원 병렬화

구현: vLLM 내장, FlashInfer 라이브러리

결과:
  긴 컨텍스트 Decode: 8× 속도 향상
  KV cache 크기가 클수록 효과적 (>4K 시퀀스)
  배치 크기 1에서도 GPU 병렬성 활용 가능
```

---

## Further Questions

**Q. PagedAttention이 메모리 효율을 높이는 방법은?**
> 연속 메모리 사전 할당(낭비) → 블록 단위 비연속 할당(필요한 만큼만). 블록 테이블로 논리→물리 매핑. prefix 블록 공유(Copy-on-Write). OS 가상 메모리와 동일 원리. GPU 이용률 20-40% → 90%+. 단편화 거의 0.

**Q. Speculative Decoding이 target 모델 출력 분포를 유지하는 이유?**
> 확률적 수락/거절 메커니즘: draft token의 target 확률 p, draft 확률 q. 수락 확률 min(1, p/q). 거절 시 adjusted = max(0, p-q)/Z에서 샘플링. 수학적 증명: 이 과정이 target 분포를 정확히 유지. 즉 draft 모델 품질 관계없이 target과 동일한 최종 분포.

**Q. Prefill vs Decode의 bottleneck 차이와 최적화 방향은?**
> Prefill: 큰 행렬 곱 → compute-bound. GPU FLOP 활용률이 핵심. 최적화: 큰 배치, Flash Attention, Chunked Prefill. Decode: 매 step마다 전체 KV cache 읽기 → memory-bandwidth-bound. 최적화: GQA/MQA로 KV 줄이기, KV 양자화, Flash Decoding(병렬 처리), 더 적은 레이어.

**Q. KV cache를 줄이는 방법들의 트레이드오프는?**
> GQA/MQA: 품질 영향 거의 없음, 학습 시 적용해야 함. KV INT8: 품질 <1% 저하, 2× 절약, 추론 중 적용 가능. H2O/SnapKV: 품질 저하 가능 (태스크 의존), 학습 불필요, 동적 조정. MLA: 품질 영향 없음, 93% 절약, 아키텍처 변경 필요. 실용적: KV INT8 + Prefix Caching 조합이 최선.

**Q. 100K 토큰 컨텍스트를 효율적으로 처리하는 방법은?**
> 1) Flash Attention + Chunked Prefill: 100K를 512 단위로 점진적 처리, TTFT 분산. 2) Flash Decoding: Decode 단계에서 병렬 처리. 3) KV 압축: H2O/SnapKV로 중요한 것만. 4) Disaggregated: Prefill 전용 서버. 5) GQA/MLA: KV 크기 자체 감소. 조합 적용이 실용적.
