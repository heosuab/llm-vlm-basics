# 심화 개념 질문 & 답변 — 엔지니어 편

## 추론 최적화

**Q1. KV Cache 크기를 어떻게 계산하나?**
```
KV cache = 2 × num_layers × num_kv_heads × head_dim × seq_len × bytes_per_element

LLaMA-3-8B (FP16):
  num_layers = 32
  num_kv_heads = 8 (GQA)
  head_dim = 128
  seq_len = 8192
  bytes = 2

= 2 × 32 × 8 × 128 × 8192 × 2 = 2.15 GB
```

**Q2. PagedAttention의 내부 동작을 설명하라.**
```
기존 문제:
- 각 요청에 max_seq_len만큼 연속 메모리 사전 할당
- 내부/외부 단편화 심각 → GPU 이용률 20-40%

PagedAttention:
1. 메모리를 고정 크기 block(16 tokens)으로 나눔
2. Block Table: 논리 block → 물리 block 매핑
3. 비연속 메모리 가능 (OS 페이징과 동일 원리)
4. 완료된 요청의 block → 즉시 반환 및 재사용
5. Prefix 공유: 같은 prefix → 물리 block 공유 (CoW)

결과: GPU 이용률 90%+, 3-24× 처리량 향상
```

**Q3. Speculative Decoding의 수학적 근거는?**
```
목표: target 분포와 동일한 샘플 생성, 속도만 향상

방법:
1. Draft 모델로 γ개 토큰 생성: x₁,...,xγ ~ p_draft
2. Target 모델이 한 번에 γ+1 위치 계산
3. 각 토큰 수락/거절:

  토큰 xᵢ에서:
  - p_target(xᵢ) ≥ p_draft(xᵢ): 항상 수락
  - p_target(xᵢ) < p_draft(xᵢ): 확률 p_target/p_draft로 수락
  - 거절 시: p_target에서 보정 샘플링

증명: 수락된 토큰은 target 분포에서 샘플링과 동일
따라서 품질 손실 없이 속도 향상 (acceptance rate 높을 때)
```

**Q4. FP8 학습이 BF16보다 어려운 이유와 DeepSeek의 해결 방법은?**
```
어려운 이유:
- FP8 E4M3: 최대값 448, 표현 범위 제한
- Activation 이상치 (outliers) 처리 어려움
- 기울기(gradient) 크기 불안정

DeepSeek-V3 해결:
1. Fine-grained quantization (블록별 scale)
   - 일부 이상치 때문에 전체 scale 손실 방지
2. High-precision accumulation: FP8 곱셈 후 FP32 누적
3. E4M3 for weights, E5M2 for gradients
   - E5M2: 표현 범위 넓어 gradient에 적합
4. Master weights: BF16 유지 → 수렴 보장
```

---

## 서빙 & 배포

**Q5. Continuous Batching vs Static Batching 선택 기준은?**
```
Static Batching:
- 같은 길이의 요청들을 배치
- 배치 내 가장 긴 요청 완료까지 대기
- GPU 낭비 (짧은 요청 완료 후 idle)
- 간단한 구현

Continuous Batching:
- 매 step마다 완료 요청 제거, 새 요청 추가
- GPU 항상 바쁨
- 복잡한 스케줄러 필요 (vLLM, TGI)
- 실제 서비스에서 항상 선택

선택:
- 요청 길이 균일 + 배치 처리: static
- 다양한 길이 + 실시간 서비스: continuous
```

**Q6. Tensor Parallelism과 Pipeline Parallelism의 차이와 선택 기준은?**
```
Tensor Parallelism (TP):
- 행렬 연산을 GPU간 분할 (열/행)
- All-Reduce 통신 필요 (매 레이어)
- 빠르지만 통신 overhead 큼
- 적합: NVLink로 연결된 같은 노드 GPU

Pipeline Parallelism (PP):
- 레이어를 노드에 분할
- P2P 통신 (이전→다음 노드)
- Micro-batching으로 bubble 최소화
- 적합: 여러 노드 간 (느린 네트워크 허용)

실제: TP × PP × DP 3D 조합 사용
예: 8-way TP (노드 내) × 4-way PP (노드 간)
```

**Q7. INT8 vs INT4 양자화 선택 기준은?**
```
INT8 (W8A8):
- 품질 손실 최소 (FP16과 거의 동일)
- GEMM 연산: INT8 native 지원
- Activation도 양자화 → 최대 속도

INT4 (W4A16, GPTQ/AWQ):
- 메모리 4× 감소
- 가중치만 양자화, 계산은 FP16 (역양자화 후 계산)
- 품질 약간 저하 (보통 허용 가능)
- 더 작은 모델 크기 → 큰 배치 가능

선택:
- 메모리 부족: INT4
- 속도 최우선 + 메모리 충분: INT8
- 품질 최우선: FP16
- 로컬 CPU: GGUF Q4_K_M
```

---

## 학습 인프라

**Q8. 대규모 학습에서 loss spike 발생 시 대처법은?**
```
예방:
- Gradient clipping (max_norm=1.0)
- 안정적인 BF16
- Warmup 충분히

발생 시 대처 순서:
1. 현재 배치 로그 확인 (이상한 데이터?)
2. 해당 배치 스킵 후 재시도
3. 이전 체크포인트로 롤백
4. 그래도 반복 시: LR 줄이기, 데이터 점검

자동화:
- loss > threshold이면 gradient skip
- 연속 spike이면 자동 롤백 + alert
```

**Q9. FSDP와 DeepSpeed ZeRO의 차이는?**
```
DeepSpeed ZeRO:
- Stage 1: Optimizer state 샤딩
- Stage 2: + Gradient 샤딩
- Stage 3: + Parameter 샤딩
- CPU offload 지원

PyTorch FSDP (Fully Sharded DP):
- PyTorch 내장 (외부 의존성 없음)
- ZeRO-3와 유사 (parameter 완전 샤딩)
- 더 쉬운 통합
- activation checkpointing 통합

선택:
- 순수 PyTorch: FSDP
- 더 많은 최적화 옵션: DeepSpeed
- CPU offload 필요: DeepSpeed ZeRO-3
```

---

## 코딩 질문

**Q10. Flash Attention을 Triton 커스텀 커널로 구현한다면 핵심은?**
```python
# 핵심 알고리즘 (의사 코드)
def flash_attention_forward(Q, K, V, block_size=64):
    """
    Q, K, V: (batch, heads, seq_len, head_dim)
    SRAM 기반 블록 단위 계산
    """
    n = Q.shape[-2]
    O = zeros_like(Q)
    l = zeros(batch, heads, n)  # softmax normalizer
    m = full((batch, heads, n), -inf)  # running max

    # Q를 블록으로 나눔
    for i in range(0, n, block_size):
        Qi = Q[..., i:i+block_size, :]
        Oi = zeros_like(Qi)
        li = zeros(block_size)
        mi = full(block_size, -inf)

        # K,V를 블록으로 나눔
        for j in range(0, n, block_size):
            Kj = K[..., j:j+block_size, :]
            Vj = V[..., j:j+block_size, :]

            # attention score 계산
            Sij = Qi @ Kj.T / sqrt(head_dim)

            # running softmax (online)
            mij = max(Sij, dim=-1)
            Pij = exp(Sij - mij)
            lij = sum(Pij, dim=-1)

            # m, l 업데이트
            mi_new = max(mi, mij)
            li = exp(mi - mi_new) * li + exp(mij - mi_new) * lij
            Oi = exp(mi - mi_new) * Oi + Pij @ Vj
            mi = mi_new

        O[..., i:i+block_size, :] = Oi / li.unsqueeze(-1)

    return O
```

**Q11. 간단한 KV Cache 구현을 보여줘라.**
```python
class KVCache:
    def __init__(self, max_batch_size, max_seq_len, num_layers,
                 num_heads, head_dim, dtype=torch.float16):
        self.cache_k = torch.zeros(
            (max_batch_size, max_seq_len, num_layers, num_heads, head_dim),
            dtype=dtype, device='cuda'
        )
        self.cache_v = torch.zeros_like(self.cache_k)
        self.seq_len = 0

    def update(self, k, v, layer_idx, start_pos):
        """k, v: (batch, seq, heads, head_dim)"""
        end_pos = start_pos + k.shape[1]
        self.cache_k[:k.shape[0], start_pos:end_pos, layer_idx] = k
        self.cache_v[:v.shape[0], start_pos:end_pos, layer_idx] = v
        self.seq_len = end_pos
        return (
            self.cache_k[:k.shape[0], :end_pos, layer_idx],
            self.cache_v[:v.shape[0], :end_pos, layer_idx]
        )
```

---

## 분산 학습

**Q12. vLLM의 PagedAttention 블록 관리를 직접 구현한다면?**
```python
class BlockManager:
    def __init__(self, total_blocks, block_size=16):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_blocks = list(range(total_blocks))
        # 요청별 할당된 블록 리스트
        self.seq_to_blocks: dict[int, list[int]] = {}

    def allocate(self, seq_id: int, num_tokens: int) -> bool:
        """요청에 필요한 블록 할당"""
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_blocks:
            return False  # OOM
        blocks = [self.free_blocks.pop() for _ in range(num_blocks)]
        self.seq_to_blocks[seq_id] = blocks
        return True

    def extend(self, seq_id: int) -> bool:
        """토큰 추가 시 블록 확장"""
        blocks = self.seq_to_blocks[seq_id]
        current_tokens = (len(blocks) - 1) * self.block_size  # 보수적 추정
        if current_tokens % self.block_size == 0:
            # 새 블록 필요
            if not self.free_blocks:
                return False
            blocks.append(self.free_blocks.pop())
        return True

    def free(self, seq_id: int):
        """요청 완료 시 블록 반환"""
        blocks = self.seq_to_blocks.pop(seq_id, [])
        self.free_blocks.extend(blocks)

    def get_block_table(self, seq_id: int) -> list[int]:
        return self.seq_to_blocks.get(seq_id, [])
```

**Q13. Tensor Parallelism에서 Column/Row Parallel Linear를 구현하라.**
```python
import torch
import torch.distributed as dist

class ColumnParallelLinear(torch.nn.Module):
    """W를 열방향 분할: Y = X @ W → Y_i = X @ W_i (각 GPU)"""
    def __init__(self, in_features, out_features, tp_size, tp_rank):
        super().__init__()
        assert out_features % tp_size == 0
        self.out_features_per_rank = out_features // tp_size
        # 이 GPU는 자신의 열 파티션만 보유
        self.weight = torch.nn.Parameter(
            torch.randn(self.out_features_per_rank, in_features)
        )

    def forward(self, x):
        # x는 모든 GPU에서 동일 (broadcast된 상태)
        return x @ self.weight.T  # partial output

class RowParallelLinear(torch.nn.Module):
    """W를 행방향 분할: Y = X @ W, X도 분할됨"""
    def __init__(self, in_features, out_features, tp_size, tp_rank):
        super().__init__()
        assert in_features % tp_size == 0
        self.in_features_per_rank = in_features // tp_size
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, self.in_features_per_rank)
        )

    def forward(self, x):
        # x는 이미 이 GPU 담당 파티션만 있음
        partial = x @ self.weight.T  # partial sum
        # AllReduce: 모든 GPU의 partial sum 합산
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        return partial

# FFN Tensor Parallel:
# ColumnParallel(d_model → d_ff) → activation → RowParallel(d_ff → d_model)
# 레이어 당 통신: RowParallel의 AllReduce 1번
```

**Q14. Continuous Batching 스케줄러를 설계하라.**
```python
from dataclasses import dataclass
from typing import Optional
import heapq

@dataclass
class Request:
    id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    generated: list[int] = None
    priority: float = 0.0  # 우선순위 (낮을수록 높은 우선순위)

    def __post_init__(self):
        self.generated = []

    def __lt__(self, other):
        return self.priority < other.priority

class ContinuousBatchScheduler:
    def __init__(self, max_tokens_per_batch=4096, max_seqs=64):
        self.max_tokens = max_tokens_per_batch
        self.max_seqs = max_seqs
        self.waiting: list[Request] = []  # priority queue
        self.running: list[Request] = []  # 현재 배치

    def add_request(self, request: Request):
        heapq.heappush(self.waiting, request)

    def schedule(self) -> list[Request]:
        """매 step: 완료된 요청 제거, 새 요청 추가"""
        # 완료된 요청 제거
        completed = [r for r in self.running
                    if len(r.generated) >= r.max_new_tokens
                    or r.generated and r.generated[-1] == EOS_TOKEN]
        for r in completed:
            self.running.remove(r)
            yield r  # 완료 처리

        # 현재 배치 토큰 수 계산
        current_tokens = sum(
            len(r.prompt_tokens) + len(r.generated)
            for r in self.running
        )

        # 새 요청 추가 (토큰 예산 내에서)
        while (self.waiting
               and len(self.running) < self.max_seqs
               and current_tokens + len(self.waiting[0].prompt_tokens) <= self.max_tokens):
            req = heapq.heappop(self.waiting)
            self.running.append(req)
            current_tokens += len(req.prompt_tokens)

        return self.running  # 이번 step에 처리할 요청들
```

**Q15. 모델 추론 서버의 SLA를 정의하고 모니터링하는 방법은?**
```python
import time
from collections import deque
import numpy as np

class InferenceMonitor:
    def __init__(self, window_size=1000):
        self.ttft_samples = deque(maxlen=window_size)  # ms
        self.tpot_samples = deque(maxlen=window_size)  # ms/token
        self.total_requests = 0
        self.failed_requests = 0

    def record_request(self, ttft_ms: float, tpot_ms: float,
                       num_tokens: int, success: bool):
        if success:
            self.ttft_samples.append(ttft_ms)
            self.tpot_samples.append(tpot_ms)
        else:
            self.failed_requests += 1
        self.total_requests += 1

    def get_metrics(self) -> dict:
        ttft = np.array(self.ttft_samples)
        tpot = np.array(self.tpot_samples)
        return {
            "ttft_p50": np.percentile(ttft, 50),
            "ttft_p95": np.percentile(ttft, 95),
            "ttft_p99": np.percentile(ttft, 99),
            "tpot_p50": np.percentile(tpot, 50),
            "tpot_p99": np.percentile(tpot, 99),
            "error_rate": self.failed_requests / max(1, self.total_requests),
            "throughput_rps": len(self.ttft_samples) / (time.time() - self.start_time),
        }

    def check_sla(self, sla_config: dict) -> list[str]:
        """SLA 위반 탐지"""
        violations = []
        metrics = self.get_metrics()
        if metrics["ttft_p99"] > sla_config["max_ttft_p99_ms"]:
            violations.append(f"TTFT P99 위반: {metrics['ttft_p99']:.0f}ms")
        if metrics["error_rate"] > sla_config["max_error_rate"]:
            violations.append(f"Error rate 위반: {metrics['error_rate']:.2%}")
        return violations

# SLA 예시:
# TTFT P99 < 2000ms (스트리밍 서비스)
# TPOT P99 < 100ms (실시간 타이핑 느낌)
# Error rate < 0.1%
# Availability > 99.9%
```

**Q16. LLM 출력에서 JSON 강제 생성(Structured Output)의 구현 원리는?**
```python
import json
from typing import Any

class StructuredDecoder:
    """Grammar-constrained decoding: JSON 스키마에 맞는 토큰만 허용"""

    def __init__(self, schema: dict, tokenizer):
        self.schema = schema
        self.tokenizer = tokenizer

    def get_valid_tokens(self, partial_output: str) -> set[int]:
        """현재까지 생성된 문자열에서 다음 유효 토큰 집합 반환"""
        # JSON 파서로 현재 위치 파악
        state = self._parse_state(partial_output)

        valid_chars = self._get_valid_next_chars(state)
        valid_tokens = set()

        for tok_id, tok_str in self.tokenizer.vocab.items():
            # 이 토큰이 추가되어도 유효한 JSON이 될 수 있는지
            if self._is_valid_prefix(partial_output + tok_str, valid_chars):
                valid_tokens.add(tok_id)

        return valid_tokens

    def constrained_generation(self, prompt: str, model) -> dict:
        """스키마를 만족하는 JSON 생성"""
        generated = ""
        input_ids = self.tokenizer.encode(prompt)

        for _ in range(max_new_tokens):
            logits = model(input_ids)  # (vocab_size,)

            # 유효하지 않은 토큰 마스킹
            valid_tokens = self.get_valid_tokens(generated)
            mask = torch.full_like(logits, float('-inf'))
            mask[list(valid_tokens)] = 0
            logits = logits + mask

            next_token = logits.argmax()
            generated += self.tokenizer.decode([next_token])
            input_ids.append(next_token)

            if generated.endswith("}") and self._is_valid_json(generated):
                break

        return json.loads(generated)

# 실제로는 Outlines, llama.cpp Grammar 등 라이브러리 사용
# 기반 기술: context-free grammar → finite state automaton → token masking
```
