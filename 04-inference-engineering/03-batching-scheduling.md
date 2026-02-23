# Batching & Scheduling 전략

## LLM 서빙의 특성

```
LLM 추론의 두 단계:

Prefill (프롬프트 처리):
  - 전체 입력 토큰 한 번에 처리
  - Compute-bound (행렬 곱셈 집약적)
  - KV cache 생성

Decode (응답 생성):
  - 토큰을 하나씩 auto-regressive 생성
  - Memory-bandwidth-bound (KV cache 읽기)
  - 각 step: 전체 KV cache를 GPU 메모리에서 읽음

도전:
  - 요청마다 입력/출력 길이 다양 (1 ~ 수천 토큰)
  - 생성 시 자동회귀적 → 요청마다 완료 시간 다름
  - Memory bandwidth bottleneck → 배치 크기가 처리량 결정
```

---

## Static Batching (기존 방식)

```
문제:
  Request A: 50 tokens 생성
  Request B: 500 tokens 생성

  → Batch [A, B]를 함께 시작
  → A가 50 step에 완료 → B 완료까지 기다림
  → A 자리에 "bubble" 발생 → GPU 낭비

  GPU 이용률: 요청들의 최소 길이/최대 길이 비율에 비례
  극단적 경우 GPU 이용률 < 10%
```

---

## Continuous Batching (Dynamic Batching)

### Orca [Yu et al., 2022]

```
핵심: 요청이 완료되는 즉시 새 요청 추가

스케줄러 동작:
  - Running queue: 현재 실행 중인 요청들
  - Waiting queue: 대기 중인 요청들

  매 iteration(step)마다:
    1. 완료된 요청(EOS 토큰 생성) 제거
    2. 여유 KV cache 메모리에 새 요청 추가
    3. 남은 모든 요청 함께 1 step 처리

결과:
  - GPU 이용률 대폭 향상
  - 처리량 최대 23× 향상 (논문 주장)
  - 현재 vLLM, TGI, TensorRT-LLM의 표준

구현 예시 (간략화):
```

```python
class ContinuousBatchingScheduler:
    def __init__(self, model, max_batch_size, max_memory):
        self.model = model
        self.running: list[Request] = []  # 실행 중
        self.waiting: deque[Request] = deque()  # 대기
        self.max_batch = max_batch_size

    def step(self):
        # 1. 완료된 요청 제거
        self.running = [
            req for req in self.running
            if not req.is_finished()
        ]

        # 2. 메모리 여유가 있으면 새 요청 추가
        while (self.waiting and
               len(self.running) < self.max_batch and
               self.has_enough_kv_memory()):
            req = self.waiting.popleft()
            self.running.append(req)
            self.allocate_kv_cache(req)

        # 3. 모든 실행 중인 요청 1 step
        if self.running:
            outputs = self.model.decode_step(self.running)
            for req, token in zip(self.running, outputs):
                req.append_token(token)

    def run(self, requests: list):
        self.waiting.extend(requests)
        while self.waiting or self.running:
            self.step()
```

---

## PagedAttention (vLLM)

### 핵심 아이디어

```
문제: 기존 KV Cache 관리
  요청마다 최대 길이(max_len)만큼 연속 메모리 사전 할당
  - 실제 생성 길이 < max_len → 메모리 낭비
  - 긴 요청이 짧은 요청의 메모리를 점유
  - 내부 단편화(fragmentation) 심각

PagedAttention:
  KV Cache를 "Page" 단위로 관리 (OS 가상 메모리 기법 응용)
  각 Page: 고정 크기 블록 (예: 16 토큰)
  Block Table: 각 요청 → Page 인덱스 매핑
  Physical KV: 연속 메모리가 아니어도 됨
```

```python
# PagedAttention 개념 코드
class PagedKVCache:
    def __init__(self, block_size=16, num_blocks=1000, num_heads=32, head_dim=128):
        self.block_size = block_size
        # 물리적 KV 블록 풀
        # [num_blocks, 2, num_heads, block_size, head_dim]
        self.physical_blocks = torch.zeros(
            num_blocks, 2, num_heads, block_size, head_dim
        )
        self.free_blocks = list(range(num_blocks))

    def allocate_block(self) -> int:
        """새 블록 할당"""
        return self.free_blocks.pop()

    def free_block(self, block_id: int):
        """블록 해제"""
        self.free_blocks.append(block_id)

# Block Table: sequence_id → [block_0, block_1, ...]
# 각 sequence의 KV cache가 어느 물리 블록에 있는지

# Prefix Caching (Radix Tree):
# 같은 prefix를 공유하는 요청들이 KV cache 재사용
# prefix = system prompt + common examples
```

### Prefix Caching (Radix Tree)

```python
# 같은 프롬프트 prefix를 공유하는 요청에서 KV cache 재사용
# vLLM의 automatic prefix caching

class RadixTree:
    """
    KV cache prefix를 Radix Tree로 관리
    공통 prefix의 KV cache를 여러 요청이 공유
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token_ids: list, kv_block_ids: list):
        """토큰 시퀀스와 KV 블록 저장"""
        node = self.root
        for token_id, block_id in zip(token_ids, kv_block_ids):
            if token_id not in node.children:
                node.children[token_id] = TrieNode(block_id=block_id)
            node = node.children[token_id]

    def lookup(self, token_ids: list) -> tuple[int, list]:
        """
        prefix match 찾기
        returns: (matched_length, matched_block_ids)
        """
        node = self.root
        matched_block_ids = []

        for i, token_id in enumerate(token_ids):
            if token_id not in node.children:
                return i, matched_block_ids
            node = node.children[token_id]
            matched_block_ids.append(node.block_id)

        return len(token_ids), matched_block_ids

# 실용적 효과:
# system_prompt가 같은 요청들: ~80% prefix hit
# 4096 토큰 system prompt → 4096 토큰 prefill 절약
# TTFT 대폭 감소
```

---

## Chunked Prefill

```
문제:
  긴 프롬프트의 prefill이 GPU를 독점
  → decode 요청들의 TPOT(token latency) 증가

예시:
  Request A: 10,000 token prompt (prefill 중)
  Request B: 200 token prompt, 이미 50 tokens 생성 (decode 중)
  → B가 A의 prefill이 끝날 때까지 기다림

해결: Prefill을 청크로 나눠 decode와 교번
  Chunk size: 512 ~ 2048 토큰
  매 iteration: prefill 청크 1개 + 모든 decode 요청

결과:
  - TTFT: 청크로 나눠 일찍 첫 prefill 완료 가능
  - TPOT: decode가 prefill에 블로킹되지 않아 일정
  - 더 공정한 스케줄링

tradeoff:
  개별 요청의 총 prefill 시간은 증가할 수 있음
  (chunking overhead 때문)
```

---

## Disaggregated Serving (분리형 서빙)

```
Prefill과 Decode를 별도 인스턴스에서 처리

Prefill 서버 (P-server):
  특성: Compute-bound
  최적 하드웨어: 많은 FLOPs (NVIDIA A100, H100)
  처리: 요청 수신 → prefill → KV cache 전송

Decode 서버 (D-server):
  특성: Memory-bandwidth-bound
  최적 하드웨어: 고대역폭 메모리 (HBM 용량 중요)
  처리: KV cache 수신 → autoregressive 생성

KV Cache 전송:
  P-server → D-server로 네트워크 전송 필요
  RDMA (Remote Direct Memory Access) 활용

장점:
  각 단계에 최적화된 하드웨어 사용
  P-server와 D-server 독립적 스케일링
  P-server 더 많이 → 동시 요청 처리량 ↑
  D-server 더 많이 → 생성 처리량 ↑

관련 논문: Splitwise, DistServe, Mooncake (ByteDance)
```

---

## 스케줄링 정책

### FCFS (First Come First Served)

```
장점: 공정함, 단순 구현
단점: 긴 요청이 짧은 요청들을 blocking
      Head-of-Line Blocking 문제
```

### SJF (Shortest Job First)

```
예상 출력 길이를 기준으로 짧은 것 먼저

예상 방법:
  1. 사용자 max_tokens 설정 사용
  2. 히스토리 기반 통계 모델
  3. 입력 길이로 추정 (입력 짧으면 출력도 짧음)

장점: 평균 지연 최소화
단점:
  기아 현상(starvation): 짧은 요청이 계속 오면 긴 요청 대기
  정확한 예측 어려움
  Max-tokens 미설정 요청 처리 곤란
```

### Priority Queue

```
SLA(Service Level Agreement) 기반:
  Tier 1 (Premium): TTFT < 100ms, max priority
  Tier 2 (Standard): TTFT < 500ms, normal priority
  Tier 3 (Batch): best effort, low priority

  batch_jobs: 비용 절약, 지연 허용 (밤에 처리)
  interactive: 낮은 지연 우선

Preemption:
  우선순위 높은 요청 도착 → 낮은 우선순위 요청 중단
  KV cache를 CPU로 swap-out 또는 recompute
  재개 시 swap-in 또는 재계산
```

---

## Speculative Decoding

```
아이디어: 작은 draft 모델이 여러 토큰 예측 → 큰 모델이 검증

알고리즘:
  1. Draft 모델(소형): K개 토큰 예측 [t₁, t₂, ..., tₖ]
  2. Target 모델(대형): K개 토큰을 한 번의 forward pass로 검증
     → Parallel prefill (모든 K개를 동시에)
  3. Accept/Reject:
     각 토큰 독립적으로 수락 여부 결정:
     P_accept(tᵢ) = min(1, P_target(tᵢ) / P_draft(tᵢ))
  4. 거부된 토큰부터 다시 draft

결과:
  동일한 품질 (target 모델의 분포 보장)
  속도: 2-4× 향상 (draft 모델이 충분히 정확할 때)

Speculative Decoding 변형:
  - Medusa: 하나의 모델이 여러 head로 draft
  - EAGLE: fine-tuned draft head
  - Self-speculative: 동일 모델의 early exit을 draft로 사용
```

```python
def speculative_decoding(target_model, draft_model, input_ids, k=5):
    """간략한 speculative decoding 구현"""
    generated = input_ids.clone()

    while not is_finished(generated):
        # 1. Draft: k개 토큰 생성
        draft_tokens = draft_model.generate(generated, max_new_tokens=k)
        candidate = torch.cat([generated, draft_tokens], dim=-1)

        # 2. Target: 모든 후보 한 번에 검증
        with torch.no_grad():
            target_logits = target_model(candidate).logits
            draft_logits = draft_model(candidate).logits

        # 3. Accept/Reject
        accepted = []
        for i in range(k):
            pos = generated.shape[1] + i
            target_prob = F.softmax(target_logits[:, pos-1], dim=-1)
            draft_prob = F.softmax(draft_logits[:, pos-1], dim=-1)
            token = draft_tokens[:, i]

            r = target_prob[0, token] / (draft_prob[0, token] + 1e-10)
            if torch.rand(1) <= r:
                accepted.append(token)
            else:
                # Reject: target 분포에서 수정 샘플링
                adjusted = F.relu(target_prob - draft_prob)
                token = torch.multinomial(adjusted / adjusted.sum(), 1)
                accepted.append(token)
                break

        generated = torch.cat([generated, torch.stack(accepted, dim=1)], dim=1)

    return generated
```

---

## 메모리 관리 전략

### KV Cache Swap

```python
# GPU → CPU KV Cache swap
class KVCacheManager:
    def __init__(self, gpu_memory_gb=80, cpu_memory_gb=512):
        self.gpu_blocks = {}     # seq_id → GPU KV blocks
        self.cpu_blocks = {}     # seq_id → CPU KV blocks (swapped out)

    def preempt_request(self, seq_id: str):
        """
        메모리 부족 시 요청 일시 중단
        KV cache를 CPU로 이동
        """
        kv = self.gpu_blocks.pop(seq_id)
        self.cpu_blocks[seq_id] = kv.to('cpu')

    def resume_request(self, seq_id: str):
        """
        요청 재개
        KV cache를 CPU에서 GPU로 복원
        """
        kv = self.cpu_blocks.pop(seq_id)
        self.gpu_blocks[seq_id] = kv.to('cuda')

    def recompute_request(self, seq_id: str, input_ids, model):
        """
        swap-in 비용이 클 때 KV cache 재계산 선택
        """
        # 기존 KV cache 버리고 다시 prefill
        del self.cpu_blocks[seq_id]
        kv = model.prefill(input_ids)
        self.gpu_blocks[seq_id] = kv
```

---

## 성능 지표 (Serving Metrics)

```
Throughput (처리량):
  - Requests/second (RPS): 초당 완료 요청 수
  - Output Tokens/second (OPS): 실제 사용 지표
  - Tokens/second/GPU: 하드웨어 효율성

Latency (지연):
  - TTFT (Time To First Token): 첫 토큰까지 시간
    → 사용자가 체감하는 응답성 (< 200ms 목표)
    → Prefill 시간과 비례
  - TPOT (Time Per Output Token): 토큰 당 생성 시간
    → 스트리밍 속도 (~30-50ms/token 목표)
  - E2E Latency: 전체 응답 완료 (TTFT + TPOT × output_len)
  - P50/P90/P99: 백분위수 지연 (SLA 기준)

메모리 효율:
  - KV Cache Hit Rate: prefix caching 효과 (높을수록 좋음)
  - GPU Memory Utilization: 메모리 활용률 (~90% 목표)
  - Block Fragmentation: 낭비된 KV 블록 비율

GPU 효율:
  - MFU (Model FLOP Utilization): 이론적 피크 대비 실제
  - GPU Utilization %: GPU 코어 사용률
```

---

## 서빙 프레임워크 비교

| 특성 | vLLM | TGI (HuggingFace) | TensorRT-LLM |
|------|------|---------------------|--------------|
| PagedAttention | ✓ | ✓ (후속 버전) | ✓ |
| Continuous Batching | ✓ | ✓ | ✓ |
| Multi-modal | ✓ | ✓ | ✓ |
| 프로덕션 안정성 | 높음 | 높음 | 높음 |
| 최적화 수준 | 중-높음 | 중간 | 최고 |
| 커스터마이징 | 쉬움 | 쉬움 | 어려움 |
| 오픈소스 | ✓ | ✓ | ✓ |
| 주요 사용처 | 범용 | 범용 | NVIDIA 플랫폼 |

---

## 실제 서빙 설정 예시

```bash
# vLLM OpenAI 호환 서버
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --tensor-parallel-size 2 \              # TP degree
  --max-model-len 8192 \                  # 최대 컨텍스트
  --max-num-seqs 256 \                    # 동시 처리 요청 수
  --gpu-memory-utilization 0.90 \         # GPU 메모리 사용률
  --enable-prefix-caching \              # Radix Tree prefix caching
  --enable-chunked-prefill \             # Chunked prefill
  --max-num-batched-tokens 8192 \         # 배치당 최대 토큰
  --quantization awq \                   # AWQ 양자화
  --speculative-model draft-model/ \     # Speculative decoding draft
  --num-speculative-tokens 5 \
  --host 0.0.0.0 --port 8000
```

```python
# vLLM Python API
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=2,
    enable_prefix_caching=True,
    max_model_len=8192,
    gpu_memory_utilization=0.90
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    stop=["</s>", "<|eot_id|>"]
)

prompts = ["Tell me about AI.", "What is Python?"]
outputs = llm.generate(prompts, sampling_params)
# 내부적으로 continuous batching 처리됨
```

---

## Further Questions

**Q. Continuous batching과 static batching의 차이는?**
> Static: 배치 내 모든 요청 완료 후 다음 배치 → 짧은 요청 완료 후 GPU bubble 발생. Continuous: 매 iteration마다 완료된 요청 제거, 새 요청 추가 → 항상 GPU 꽉 채워서 실행. Orca 논문: 23× 처리량 향상 (이론적 최대). 현재 모든 프로덕션 서빙의 표준.

**Q. Chunked prefill이 왜 TTFT와 TPOT 모두 개선하나?**
> TTFT 개선: 긴 prefill을 여러 청크로 나눠서 → 첫 청크 완료 시점이 빨라짐 (decode 시작 가능). TPOT 개선: prefill 청크가 decode step들과 교번 실행 → decode가 긴 prefill에 블로킹되지 않음. 단, 개별 요청의 총 prefill 완료 시간은 증가 (청크 오버헤드). 전체적으로 지연 분산 효과.

**Q. PagedAttention이 KV Cache 관리를 어떻게 개선하나?**
> 기존: 각 요청에 max_len 연속 메모리 예약 → 내부 단편화 (실제 생성 < max_len). PagedAttention: 고정 크기 블록(16 토큰) 단위로 필요할 때마다 할당 → 단편화 최소화. 물리적 연속성 불필요 → Block Table로 논리-물리 매핑. Copy-on-Write: Beam search에서 prefix 블록 공유.

**Q. Speculative Decoding이 왜 품질을 유지하면서 빠른가?**
> Draft 모델이 틀려도 거부 후 Target 모델의 올바른 분포에서 샘플링. 수락 확률: min(1, P_target/P_draft). 거부된 위치부터 다시 시작. 이론적으로 Target 모델과 동일한 분포 보장 (rejection sampling 원리). 가속: Draft 모델이 충분히 정확하면 K개 토큰을 1번 forward로 처리 → 최대 K× 속도. 실제: 2-3× 속도 향상 (draft 정확도 ~70-80%일 때).
