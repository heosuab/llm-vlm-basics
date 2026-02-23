# 분산 학습 (Distributed Training)

## 왜 분산 학습인가?

```
모델 크기와 GPU 메모리:
  LLaMA-3-8B  : ~16 GB (BF16) — A100 1장 가능, optimizer state 포함 시 불가
  LLaMA-3-70B : ~140 GB (BF16) — A100 80GB 2장 필요 (weights만)
  LLaMA-3-405B: ~810 GB (BF16) — A100 10장 필요 (weights만)

학습 시 메모리 = weights + gradients + optimizer states + activations
  BF16 weight + BF16 grad + FP32 master weight + FP32 Adam m + FP32 Adam v
  = 2 + 2 + 4 + 4 + 4 = 16 bytes/param

LLaMA-3-8B 학습 메모리:
  8B × 16 bytes = 128 GB (activation 제외)
  → A100 80GB 2장 이상 필수
```

---

## 1. Data Parallelism (DP)

### 기본 개념
```
같은 모델을 N개 GPU에 복사
미니배치를 N분할 → 각 GPU가 독립적으로 forward/backward
gradient를 AllReduce로 동기화 → 모든 GPU 동일 gradient로 업데이트

장점: 구현 단순, 통신 단순 (한 번의 AllReduce)
단점: 각 GPU에 전체 모델 필요 → 큰 모델 불가
```

### DDP (DistributedDataParallel)
```python
# Ring-AllReduce 기반
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 초기화
dist.init_process_group(backend='nccl')  # NCCL: GPU 간 최적화
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

model = MyModel().cuda()
model = DDP(model, device_ids=[local_rank])

# 학습 루프
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()      # backward 중에 gradient AllReduce 자동 수행
    optimizer.step()     # 모든 GPU 동일한 optimizer step
```

### DDP 최적화 기법
```
Gradient Bucketing:
  작은 gradient를 bucket에 모아 한 번에 AllReduce
  통신-계산 오버랩 가능 (backward 계산과 통신 동시)
  bucket_cap_mb=25 (기본값), 크면 통신 효율, 작으면 오버랩 증가

find_unused_parameters=False:
  Dynamic computation graph 없으면 False로 성능 향상

gradient_as_bucket_view=True:
  gradient를 bucket 메모리에 직접 저장 → 복사 오버헤드 제거
```

---

## 2. ZeRO Optimization (Zero Redundancy Optimizer)

### 메모리 분석
```
혼합 정밀도 학습 시 파라미터당 메모리:
  BF16 weight:      2 bytes
  BF16 gradient:    2 bytes (선택적으로 FP32)
  FP32 master weight: 4 bytes
  FP32 Adam m (1st moment): 4 bytes
  FP32 Adam v (2nd moment): 4 bytes
  합계: 16 bytes/param

순수 optimizer state: 4+4+4 = 12 bytes/param (FP32)
→ 7B 모델: 7×10⁹ × 12 ≈ 84 GB (optimizer state만)
→ 전체: 7×10⁹ × 16 ≈ 112 GB
```

### ZeRO Stage 1: Optimizer State 샤딩
```
N개 GPU에 optimizer state 균등 분할
각 GPU: 전체 weights + gradients, optimizer state의 1/N

절약: optimizer state 메모리 / N
  12 bytes → 12/N bytes (Adam m, v, master weight)

업데이트 과정:
  1. AllReduce로 gradient 수집 (기존 DDP와 동일)
  2. 각 GPU가 자신 담당 파라미터만 optimizer step
  3. AllGather로 updated weight 전파 (자신 담당 제외 필요 시)

효과: 7B 모델, 8 GPU → optimizer state 84GB → 10.5GB/GPU
```

### ZeRO Stage 2: Gradient + Optimizer State 샤딩
```
Gradient도 1/N 유지
통신 패턴:
  - Backward: ReduceScatter (각 GPU가 자신 담당 gradient sum 수집)
  - Update: 각 GPU가 자신 담당만 optimizer step
  - 다음 forward 전: AllGather (모든 GPU가 최신 weight 획득)

AllReduce = ReduceScatter + AllGather
→ ZeRO-2: ReduceScatter만 사용 (통신량 동일하지만 피크 메모리 절반)
```

### ZeRO Stage 3: Parameter + Gradient + Optimizer State 샤딩
```
Model weight까지 1/N 분할
  각 GPU: 1/N weights, 1/N gradients, 1/N optimizer states

Forward pass:
  1. AllGather: 필요한 레이어의 weight 수집
  2. Forward 계산
  3. 사용 후 weight 즉시 해제

Backward pass:
  1. AllGather: 해당 레이어 weight
  2. Backward 계산
  3. ReduceScatter: gradient 분산
  4. Weight 해제

통신량: 3× 증가 (AllGather forward + AllGather backward + ReduceScatter)
메모리: 16/N bytes/param
```

### ZeRO-Infinity (Stage 3 + CPU/NVMe Offload)
```
CPU RAM offload:
  Optimizer state, gradient를 CPU RAM에 저장
  H100 80GB + 512GB CPU RAM → 매우 큰 모델 가능

NVMe offload:
  Optimizer state를 NVMe SSD에도 저장
  가장 큰 모델 학습 가능 (속도 희생)

실제 속도 트레이드오프:
  GPU ↔ CPU: PCIe 3.0 × 16 ≈ 32 GB/s (양방향)
  GPU ↔ NVMe: PCIe SSD ≈ 3-7 GB/s
  → offload 비율 최소화해야 함

사용 예: 단일 DGX (8×A100) + CPU offload로 100B+ 모델 파인튜닝
```

### DeepSpeed ZeRO 설정
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "offload_param": {"device": "cpu", "pin_memory": true},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9
  }
}
```

---

## 3. Tensor Parallelism (TP)

### 핵심 아이디어 (Megatron-LM)
```
행렬 연산을 여러 GPU에 분할
Y = XW → W를 열방향 또는 행방향으로 분할

Column Parallel (f conjugate):
  W = [W₁ | W₂] (열방향 분할)
  Y = [XW₁ | XW₂] = [Y₁ | Y₂]
  → GPU0: Y₁, GPU1: Y₂
  → 입력 X를 broadcast (또는 Identity 연산)

Row Parallel (g conjugate):
  W = [W₁]  (행방향 분할)
      [W₂]
  X = [X₁ | X₂]  (입력도 분할)
  Y = X₁W₁ + X₂W₂
  → AllReduce로 합산
```

### FFN Tensor Parallelism
```
FFN: Y = GELU(XW₁)W₂

GPU0: Y₁ = GELU(XW₁_part1) → partial output
GPU1: Y₁ = GELU(XW₁_part2) → partial output

Column parallel W₁ → Row parallel W₂
→ 레이어 내 AllReduce 1번 (Y₁ + Y₂)

통신 없이 내부 계산, 끝에 1번 AllReduce
```

### Attention Tensor Parallelism
```
Multi-Head Attention (8 heads, TP-4):
  GPU0: heads 0,1
  GPU1: heads 2,3
  GPU2: heads 4,5
  GPU3: heads 6,7

Q, K, V projection: Column parallel
Output projection: Row parallel
→ 역시 레이어당 AllReduce 2번 (attention + FFN)
```

### TP 통신 비용
```
통신 횟수: 각 Transformer 레이어당 AllReduce 2번 (FFN + Attention)
통신량/step: 2 × num_layers × AllReduce_size
  LLaMA-3-8B, TP=4: 2 × 32 × (8192×4096 × 2bytes) ≈ huge

→ NVLink (900 GB/s) 필수, PCIe (64 GB/s)는 느림
→ 같은 노드 내 GPU끼리만 효율적
→ TP degree 제한: num_heads로 나눠 떨어져야 함
```

### 구현 (Megatron-LM 스타일)
```python
from megatron.core import tensor_parallel

# Column parallel linear (입력 X → 각 GPU가 일부 output 계산)
class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tp_size):
        self.weight = nn.Parameter(
            torch.randn(output_size // tp_size, input_size)
        )

    def forward(self, x):
        # x: 모든 GPU에 동일 (broadcast)
        out = x @ self.weight.T  # 각 GPU가 partial output
        return out  # AllGather 나중에

# Row parallel linear (입력도 분할 → 합산 필요)
class RowParallelLinear(nn.Module):
    def forward(self, x):
        out = x @ self.weight.T  # partial sum
        dist.all_reduce(out)     # AllReduce로 합산
        return out
```

---

## 4. Pipeline Parallelism (PP)

### 기본 아이디어
```
Transformer 레이어를 여러 GPU(노드)에 순차 배치

GPU0: Layer 0~7   (Embedding + 8 layers)
GPU1: Layer 8~15
GPU2: Layer 16~23
GPU3: Layer 24~31 + LM head

통신: 레이어 경계에서 activation tensor 전달 (P2P)
  Forward: GPU0 → GPU1 → GPU2 → GPU3
  Backward: GPU3 → GPU2 → GPU1 → GPU0
```

### Naive PP의 문제: Bubble
```
Timeline (4 GPU, 4 microbatch):
  GPU0: [F0][F1][F2][F3]         [B3][B2][B1][B0]
  GPU1:     [F0][F1][F2][F3][B3][B2][B1][B0]
  ...

Bubble (유휴 시간) = (p-1) / (m + p-1)
  p = pipeline stages, m = microbatch 수

  p=4, m=1: bubble = 3/4 = 75%  → 매우 비효율
  p=4, m=8: bubble = 3/11 ≈ 27%
  p=4, m=16: bubble = 3/19 ≈ 16%

→ microbatch 수를 늘려 bubble 최소화
```

### 1F1B Schedule (One Forward, One Backward)
```
PipeDream 스타일의 스케줄링

Steady state:
  각 GPU가 1 forward → 1 backward 교대
  Inflight microbatch = p (pipeline stages 수)

Timeline (p=4, m=4):
  GPU0: F0 F1 F2 F3 B0 B1 B2 B3
  GPU1:    F0 F1 F2 B0 F3 B1 B2 B3
  GPU2:       F0 F1 B0 F2 B1 F3 B2 B3
  GPU3:          F0 B0 F1 B1 F2 B2 F3 B3

메모리: p개의 inflight microbatch activation 유지
Bubble: (p-1)/(m+p-1) — Naive와 동일하지만 메모리 효율 다름
```

### Interleaved 1F1B (Megatron-LM v2)
```
각 GPU가 연속이 아닌 레이어 청크 담당

예 (p=4, v=2 chunks):
  GPU0: Layer 0-3, 16-19
  GPU1: Layer 4-7, 20-23
  GPU2: Layer 8-11, 24-27
  GPU3: Layer 12-15, 28-31

Bubble: (p-1)/(m×v + p-1)  ← v배 감소
  p=4, m=8, v=2: 3/(16+3) = 3/19 ≈ 16%  vs  3/11 ≈ 27%

단점: 통신 횟수 v배 증가 (chunk 경계마다)
```

### PP 메모리 vs 통신 균형
```
메모리 균등 분배:
  각 스테이지가 비슷한 메모리 사용해야 효율적
  Embedding layer: vocab_size × d_model 크기 주의
  → Embedding을 마지막 스테이지와 합치거나 따로 처리

통신 텐서 크기:
  batch × seq_len × d_model × 2 bytes (BF16)
  예: batch=1, seq=4096, d_model=8192
  → 1 × 4096 × 8192 × 2 = 64 MB per boundary

재시작 시 체크포인트:
  각 PP rank가 자신의 layer checkpoint 저장
  재시작 시 rank별로 load
```

---

## 5. Sequence Parallelism (SP)

### 동기
```
TP로 처리 못하는 연산들이 존재:
  LayerNorm, Dropout: 각 token 위치에서 독립적
  → 시퀀스 길이 차원으로 분할 가능

Megatron-LM SP:
  TP와 상보적으로 사용
  TP region: Attention, FFN (column/row parallel)
  SP region: LayerNorm, Dropout (sequence split)
```

### Ring Attention (장문 시퀀스)
```
문제: 시퀀스 길이 N의 attention = O(N²) 메모리
  N=128K: 매우 큰 attention matrix

해결: 시퀀스를 P개 GPU에 분할
  각 GPU: N/P tokens의 Q
  전체 K, V는 ring 방식으로 순환

알고리즘:
  1. 각 GPU가 Q[i], K[i], V[i] 보유
  2. Round 0: Q[i] × K[i] 계산
  3. Round 1: K[1→0], V[1→0] 전달, Q[i] × K[(i-1)%P] 계산
  4. ... P rounds
  5. Online softmax로 partial attention 누적 (Flash Attention 방식)

통신: P-1 번의 P2P (K, V 텐서 전달)
메모리: O(N/P) per GPU
```

### DeepSpeed Ulysses
```
All2All 통신 기반 시퀀스 병렬화

  입력: (N/P, d) → All2All → (N, d/P)
  → 각 GPU가 전체 시퀀스 보지만 head 일부만 처리
  Attention 계산
  → All2All → (N/P, d) 복원

vs Ring Attention:
  Ulysses: 통신 횟수 적음, 구현 단순
  Ring: head 수에 제한 없음, 극도로 긴 시퀀스 적합
```

---

## 6. Expert Parallelism (EP) — MoE용

```
각 Expert를 서로 다른 GPU에 배치
입력 token → Router → 어떤 Expert로 갈지 결정

통신 패턴:
  1. AlltoAll: 토큰을 담당 Expert GPU로 전송
  2. 각 GPU에서 Expert 계산
  3. AlltoAll: 결과를 원래 GPU로 반환

Expert capacity:
  각 Expert가 처리할 수 있는 최대 token 수
  capacity_factor × (tokens/experts)
  초과 시: token drop (load imbalance 발생)

EP vs TP:
  EP: Expert 단위 분할, AlltoAll 통신
  TP: 행렬 내부 분할, AllReduce 통신
  → 대형 MoE: EP + TP 조합 (DeepSeek-V3)
```

---

## 7. 3D/4D Parallelism

### 최적 병렬화 전략
```
변수:
  - 모델 크기 (parameter 수)
  - 배치 크기, 시퀀스 길이
  - GPU 수, 노드 수
  - 노드 내/간 대역폭

일반 원칙:
  TP: 노드 내 GPU (NVLink 필요), degree = num_heads의 약수
  PP: 노드 간 (InfiniBand 허용), degree = num_stages
  DP: 나머지 → effective_batch_size / local_batch
  SP: TP와 함께, 장문 시퀀스 시

수식:
  total_gpus = TP × PP × DP (× SP × EP)
```

### 실제 설정 예시
```
LLaMA-3-8B (8× A100 80GB 노드):
  단일 노드: ZeRO-2 + DP=8, 또는 TP=4 + DP=2
  권장: FSDP (ZeRO-3 equivalent) DP=8

LLaMA-3-70B (4노드 × 8× A100):
  TP=8 (노드 내) × DP=4 (노드 간)
  또는 TP=4 × PP=2 × DP=4

LLaMA-3-405B (16노드 × 8× H100):
  TP=8 × PP=8 × DP=16
  Megatron-LM 권장

DeepSeek-V3 (2048× H800):
  TP=1 (MLA 덕분에 TP 불필요)
  PP=16 × EP=64 × DP=2048/(16×64)
```

### 메모리 절약 조합
```
ZeRO-3 + Activation Checkpointing + Offload:
  ZeRO-3: parameter 샤딩 (16/N bytes)
  Activation Checkpointing: activation 재계산 (메모리 ↓, 계산 ↑30%)
  CPU Offload: optimizer state CPU로 (속도 ↓)

세 기법 모두 적용: 모델 크기 대비 GPU 메모리 사용 최소화
```

---

## 8. 통신 프리미티브 (Communication Primitives)

### 기본 연산
```
N개 GPU, 각 GPU가 x_i 보유

AllReduce:
  모든 GPU가 sum(x_i) 또는 mean(x_i) 획득
  = ReduceScatter + AllGather
  사용: DDP gradient 동기화

AllGather:
  모든 GPU가 [x_0, x_1, ..., x_{N-1}] 획득
  사용: ZeRO-3 parameter 복원

ReduceScatter:
  각 GPU i가 x_i의 i번째 chunk의 sum 획득
  사용: ZeRO gradient 동기화

P2P (Send/Recv):
  GPU i → GPU j 직접 전송
  사용: Pipeline Parallelism activation 전달

AlltoAll:
  GPU i의 j번째 chunk → GPU j로 전달
  사용: MoE Expert Parallelism, Ulysses SP

Broadcast:
  하나의 GPU → 나머지 모두
  사용: 모델 weight 초기 배포
```

### Ring-AllReduce 알고리즘
```
N개 GPU가 ring 구조로 연결
2(N-1) 단계로 수행:

1단계 ReduceScatter: N-1 번
  각 GPU가 자신의 데이터 chunk를 시계 방향 전달+합산

2단계 AllGather: N-1 번
  합산된 chunk를 반시계 방향으로 배포

총 통신량: 2 × (N-1)/N × data_size
  → AllReduce는 N이 커도 통신량 거의 일정 (bandwidth optimal)
```

### 통신-계산 오버랩
```
DDP:
  backward 시작과 동시에 gradient AllReduce
  마지막 레이어 backward → 즉시 통신 시작
  backward 중 앞 레이어 gradient 통신

ZeRO:
  forward: AllGather 통신과 이전 레이어 계산 오버랩
  backward: ReduceScatter 통신과 이전 레이어 backward 오버랩

PP:
  micro-batch forward를 겹쳐서 P2P 통신 숨김
```

---

## 9. 네트워크 토폴로지

### 노드 내 (Intra-node)
```
NVLink:
  A100: NVLink 3.0 — 600 GB/s 양방향 (12 links × 25 GB/s)
  H100: NVLink 4.0 — 900 GB/s 양방향 (18 links × 25 GB/s)
  H200: NVLink 4.0 — 900 GB/s

NVSwitch:
  8-way 또는 4-way full-mesh 연결
  모든 GPU 쌍이 최대 대역폭으로 통신

PCIe (NVLink 없는 경우):
  PCIe 4.0 × 16: 32 GB/s (양방향 64 GB/s)
  TP에 부적합 (NVLink 대비 10-15× 느림)
```

### 노드 간 (Inter-node)
```
InfiniBand:
  HDR (200 Gb/s = 25 GB/s)
  NDR (400 Gb/s = 50 GB/s)
  실제: 여러 rail → 100-400 GB/s 집계 대역폭

RoCE (RDMA over Converged Ethernet):
  표준 이더넷 기반 RDMA
  DeepSeek 클러스터: RoCE 사용

Topology:
  Fat-tree: 균형 잡힌 bisection bandwidth
  Rail-optimized: GPU pair별 전용 링크
```

### 대역폭이 왜 중요한가
```
AllReduce 시간 (Ring) = 2 × (N-1)/N × size / bandwidth

TP=8, H100 cluster, gradient 10GB:
  NVLink: 10GB / 900GB/s × 2 ≈ 22ms
  PCIe: 10GB / 64GB/s × 2 ≈ 313ms
  → TP는 반드시 NVLink 노드 내 사용

PP inter-node 통신:
  activation 64MB / 50GB/s = 1.28ms (NDR)
  계산 시간이 훨씬 크면 문제없음
```

---

## 10. 학습 프레임워크

### Megatron-LM (NVIDIA)
```
기능:
  TP, PP, SP, DP, VPP (Interleaved PP) 지원
  Flash Attention 통합
  RoPE, 최신 Transformer 구조 지원
  distributed checkpoint

사용 모델:
  GPT, BERT, T5, LLaMA 등 거의 모든 LLM

실행:
  torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --num-layers 32 ...
```

### DeepSpeed (Microsoft)
```
기능:
  ZeRO (Stage 1/2/3/Infinity)
  CPU/NVMe offload
  Pipeline engine
  커스텀 CUDA kernels (FusedAdam 등)

장점: ZeRO가 가장 잘 구현됨, 다양한 CPU offload

통합: transformers, LightningAI, accelerate

설정 파일 기반으로 간단 적용:
  model = deepspeed.initialize(model, config="ds_config.json")
```

### PyTorch FSDP (Fully Sharded Data Parallel)
```
PyTorch 내장 (외부 의존 없음)
ZeRO-3와 동일한 개념

장점:
  PyTorch 생태계 완벽 통합
  Activation checkpointing 통합
  더 단순한 API

사용:
  from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
  model = FSDP(model,
               sharding_strategy=ShardingStrategy.FULL_SHARD,
               cpu_offload=CPUOffload(offload_params=False),
               auto_wrap_policy=transformer_auto_wrap_policy)
```

### Accelerate + Trainer (HuggingFace)
```
DeepSpeed/FSDP 위에 추상화 레이어
  from accelerate import Accelerator
  accelerator = Accelerator()
  model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

TRL (Transformer Reinforcement Learning):
  SFT, DPO, PPO, GRPO 학습에 특화
  FSDP + DeepSpeed 지원

Torchtune (Meta):
  LLaMA 학습에 최적화된 순수 PyTorch
```

---

## 11. 실전: GPU 메모리 계산 및 설정

### 메모리 예산 계산 예시 (7B 모델 학습)
```
Model parameters: 7B × 2 bytes (BF16) = 14 GB
Gradients: 7B × 2 bytes = 14 GB
Optimizer states: 7B × 12 bytes = 84 GB (FP32 Adam)
Activations (seq=4096, batch=4): 수십 GB (레이어당)

합계 (activations 제외): 14+14+84 = 112 GB

단일 A100 80GB → 불가
2× A100 160GB → 가능 (빠듯, activation checkpointing 필요)
4× A100 320GB → 여유로움

ZeRO-3로 4 GPU:
  (14+14+84)/4 = 28 GB per GPU → A100 여유
```

### 배치 크기 & 처리량
```
Tokens/sec ∝ GPU수 × GPU FLOPs / (모델 FLOP/token)

1 forward token당 FLOPs ≈ 2 × N_params (N = non-embedding)
LLaMA-3-8B: 2 × 8B = 16 GFLOPs/token

A100 (312 TFLOPS BF16), MFU = 50%:
  312 × 0.5 × 10¹² / 16 × 10⁹ ≈ 9,750 tokens/s per GPU
  8× A100: ~78K tokens/s
  → 15T token 학습: 15×10¹² / 78000 ≈ 192,000 seconds ≈ 55시간 (이상적)
  실제: 효율 고려, 며칠~몇 주
```

---

## 12. 디버깅 & 모니터링

### 자주 발생하는 문제
```
Rank 간 Loss 불일치:
  원인: 데이터 분배 오류, RNG state 불일치
  해결: 각 rank의 loss 로깅, DataLoader seed 고정

NCCL Timeout:
  원인: 네트워크 장애, 일부 GPU hang, 불균형 통신
  해결: NCCL_DEBUG=INFO로 원인 파악

OOM (Out of Memory):
  원인: activation 메모리 과다, 배치 너무 큼
  해결: gradient checkpointing, 배치 감소, ZeRO stage 높이기

Loss Spike:
  원인: 나쁜 데이터 배치, LR 너무 큼
  해결: gradient clipping, 배치 스킵, 롤백

Dead GPU:
  원인: 하드웨어 결함, ECC 오류
  해결: 자동 노드 교체 시스템, health check 스크립트
```

### 모니터링 지표
```python
# 분산 학습 모니터링
import torch.distributed as dist

def log_dist_metrics(model, loss, step):
    # Gradient norm 체크 (rank 간 일치해야 함)
    grad_norm = sum(p.grad.norm()**2 for p in model.parameters()
                   if p.grad is not None) ** 0.5

    # 각 rank의 loss 수집 (디버깅용)
    loss_tensor = torch.tensor([loss]).cuda()
    dist.all_gather_into_tensor(loss_all, loss_tensor)

    if dist.get_rank() == 0:
        wandb.log({
            "loss": loss,
            "grad_norm": grad_norm,
            "gpu_memory": torch.cuda.memory_allocated() / 1e9,
            "step": step
        })

# 주요 지표
metrics_to_track = [
    "loss (global & per-rank)",
    "grad_norm",
    "learning_rate",
    "tokens_per_second",
    "gpu_memory (peak & current)",
    "mfu (model flops utilization)",
    "forward/backward time",
    "data loading time"
]
```

---

## Further Questions

**Q1. TP, PP, DP의 차이와 언제 무엇을 쓰나?**
```
DP: 모델 복사, 배치 분할. 모델이 GPU에 들어갈 때 사용. 구현 단순.
TP: 행렬 연산 분할, AllReduce 필요. NVLink 필수. 노드 내.
PP: 레이어 분할, P2P 통신. 노드 간. Bubble overhead.
실제: 3D 조합 — TP(노드 내) × PP(노드 간) × DP(나머지)
```

**Q2. ZeRO-3가 TP와 다른 점은?**
```
ZeRO-3: DP 계열. 파라미터를 DP rank에 샤딩.
  AllGather로 사용 시 복원, forward 후 즉시 해제.
  통신: DP 통신과 동일한 시점에 발생.

TP: 행렬 자체를 분할. 레이어 내에서 분산 계산.
  통신: 각 레이어 forward/backward마다 발생.

ZeRO-3: 단순 DP 환경에서 메모리만 줄임
TP: 실제 연산을 분산 (계산 속도 향상 가능)
```

**Q3. Pipeline bubble을 어떻게 최소화하나?**
```
1. Microbatch 수 증가 (m ↑): bubble = (p-1)/(m+p-1) → m 크면 작아짐
2. Interleaved 1F1B (v chunks): bubble/v 감소 (통신 v배 증가)
3. ZeroBubble PP: backward의 W(weight grad)와 B(input grad) 분리
   → bubble을 거의 0으로 만들 수 있음 (NVIDIA 2024 논문)
4. 스테이지 간 compute time 균등화: 메모리/계산 균형
```

**Q4. AllReduce = ReduceScatter + AllGather인 이유는?**
```
ReduceScatter: 각 GPU가 전체 합의 1/N 담당
  GPU0: sum(x[0] 청크), GPU1: sum(x[1] 청크), ...

AllGather: 모든 GPU가 모든 청크 수집
  → 모든 GPU가 sum(전체) 획득 = AllReduce와 동일

이점:
  ZeRO-2: backward에서 ReduceScatter만 → 각 GPU 1/N gradient만 보유
  → AllReduce보다 피크 메모리 절반
  통신량은 동일
```

**Q5. 학습 도중 GPU가 하나 죽으면?**
```
현재 표준: Checkpoint 기반 복구
  - 주기적으로 모든 rank가 checkpoint 저장
  - 장애 발생 시 마지막 체크포인트부터 재시작

Elastic Training (Torch Elastic):
  - rank 수 변경 가능
  - 죽은 GPU 제외하고 계속 학습
  - batch size 자동 조정

체크포인트 전략:
  - 너무 자주: I/O overhead
  - 너무 드물게: 많은 재계산
  - 일반적: 1000~5000 steps마다
```

**Q6. 메가트론-LM vs DeepSpeed 선택 기준은?**
```
Megatron-LM:
  - TP, PP, SP 필요 (다중 노드 대규모 학습)
  - LLM pretraining 특화
  - NVIDIA GPU 최적화

DeepSpeed:
  - ZeRO (특히 ZeRO-3, CPU offload)
  - 다양한 GPU 환경
  - 파인튜닝에 주로 사용
  - CPU 메모리 활용한 대형 모델 학습

FSDP:
  - 순수 PyTorch, 외부 의존 없음
  - 중소규모 분산 학습
  - HuggingFace Trainer와 통합 쉬움

실제: 대규모 pretraining은 Megatron-DeepSpeed (조합)
      파인튜닝은 FSDP 또는 DeepSpeed ZeRO-3
```
