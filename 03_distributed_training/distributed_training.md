# Section 3: Distributed Training & Systems

> 여러 GPU를 사용하여 대형 모델을 학습시키는 방법들.

---

## Multi-GPU vs Multi-Node

대형 LLM/VLM 학습에서는 단일 GPU로는 메모리와 연산량이 부족하므로 여러 GPU를 사용. 이때 분산 학습은 크게 Single-Node Multi-GPU와 Multi-Node Multi-GPU로 구분.

### Single-Node Multi-GPU

하나의 서버(node) 안에 여러 GPU를 사용하는 방식.
```
[Node 0]
 ├─ GPU 0
 ├─ GPU 1
 ├─ GPU 2
 └─ GPU 3
```
- 통신: NVLink / PCIe (고속, 저지연)
- 구현 난이도: 비교적 낮음
- 주 사용: 연구 실험, 중형 모델 학습
- 제한: GPU 수가 서버 내부로 제한됨 (보통 8~16개)

Tensor Parallel(TP)는 주로 same-node GPU 간(NVLink) 에서 가장 효율적.

### Multi-Node Multi-GPU

여러 서버(node)를 네트워크로 연결하여 학습.

```
[Node 0]          [Node 1]          [Node 2]
 ├─ GPU 0          ├─ GPU 0          ├─ GPU 0
 ├─ GPU 1          ├─ GPU 1          ├─ GPU 1
 ├─ GPU 2          ├─ GPU 2          ├─ GPU 2
 └─ GPU 3          └─ GPU 3          └─ GPU 3
```
- 통신: InfiniBand / Ethernet (노드 간 네트워크)
- 확장성: 수백~수천 GPU까지 확장 가능
- 필수: LLaMA-3 405B, GPT-4급 모델 pretraining
- 문제: 네트워크 병목이 주요 성능 제한 요인

일반적으로는 다음과 같이 설계:
```
Intra-node  → Tensor Parallel (고속 NVLink 활용)
Inter-node  → Data Parallel (gradient 동기화)
```

### Hybrid 구조 (현실적인 LLM 학습)

대규모 모델 학습에서는 대부분 `Multi-Node × Multi-GPU` 구조이며, `TP × PP × DP = 전체 GPU 수`와 같은 3D parallelism으로 구성.

```
총 256 GPU
  8 GPU  per node
  32 nodes

TP = 4   (node 내부)
PP = 8   (node 간 stage 분할)
DP = 8   (data replica)
→ 4 × 8 × 8 = 256
```

---

## Data Parallel (DP)

### 개념

가장 단순한 형태의 분산 학습. 모델 전체를 **각 GPU에 복제**하고, 데이터를 나눠 처리함.

```
GPU 0: 모델 사본 + mini-batch [0:32]  → gradient_0
GPU 1: 모델 사본 + mini-batch [32:64] → gradient_1
GPU 2: 모델 사본 + mini-batch [64:96] → gradient_2
GPU 3: 모델 사본 + mini-batch [96:128]→ gradient_3

→ gradient들을 평균 → 모든 GPU의 파라미터 업데이트
```

**한계**: 각 GPU에 전체 모델이 올라가야 하므로, 모델이 단일 GPU 메모리에 맞아야 함.

---

## Distributed Data Parallel (DDP)

DP의 개선 버전으로, PyTorch의 기본 분산 학습 방법.

### Ring-AllReduce

각 GPU의 gradient를 모아 평균내는 방법. 중앙 서버 없이 ring topology로 효율적으로 통신함:

```
GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0 (reduce-scatter)
GPU 0 ← GPU 1 ← GPU 2 ← GPU 3 ← GPU 0 (all-gather)
```

총 통신량: 2 × (N-1)/N × model_size (N: GPU 수)

### Communication-Computation Overlap

Backward pass 도중, 이미 gradient 계산이 끝난 레이어는 **forward 중에 통신을 시작**함 (overlap):

```
Layer 100 backward → gradient 계산 완료 → 즉시 AllReduce 시작
Layer 99 backward  →                       동시 진행
Layer 98 backward  →                       동시 진행
...
```

GPU utilization이 크게 높아집니다.

---

## Tensor Parallel (TP)

하나의 **weight matrix 자체를 여러 GPU에 분할**함.

### Column Parallel (MLP 예시)

```
원래: Y = GELU(XW)  W ∈ ℝ^(d × 4d)

TP: W를 컬럼 방향으로 2개 GPU에 분할
  GPU 0: W₁ ∈ ℝ^(d × 2d) → Y₁ = GELU(XW₁)
  GPU 1: W₂ ∈ ℝ^(d × 2d) → Y₂ = GELU(XW₂)
  Y = concat(Y₁, Y₂)  ← AllGather 필요
```

### Row Parallel (다음 레이어)

```
원래: Z = YW₂  W₂ ∈ ℝ^(4d × d)

TP: W₂를 행 방향으로 분할
  GPU 0: W₂₁ ∈ ℝ^(2d × d), 입력 Y₁ → partial Z₁
  GPU 1: W₂₂ ∈ ℝ^(2d × d), 입력 Y₂ → partial Z₂
  Z = Z₁ + Z₂  ← AllReduce 필요
```

**Column + Row parallel**: 하나의 FFN 블록에서 1번의 AllReduce만 필요함.

**Megatron-LM**: TP의 표준 구현. Attention도 head를 GPU에 분산함.

**한계**: 레이어 내 통신이 발생하므로 same-node GPU 간(NVLink)에서만 효율적.

---

## Pipeline Parallel (PP)

**모델의 레이어들을 여러 GPU에 분산**함.

```
GPU 0: Layer 1-8
GPU 1: Layer 9-16
GPU 2: Layer 17-24
GPU 3: Layer 25-32
```

### Pipeline Bubble 문제

단순한 구현에서는 한 micro-batch가 전 GPU를 순차적으로 지나가야 하므로 대기 시간(bubble)이 발생함:

```
GPU 0: [F1][F2][F3][F4][B4][B3][B2][B1]
GPU 1:     [F1][F2][F3][F4][B4][B3][B2][B1]
GPU 2:         [F1][F2][F3][F4][B4][B3][B2][B1]
GPU 3:             [F1][F2][F3][F4][B4][B3][B2][B1]
          ↑ bubble                            ↑ bubble
```

Bubble 비율 = (p-1) / (m + p - 1)  (p: pipeline stages, m: micro-batches)
→ m을 크게 할수록 bubble 비율 감소

### 1F1B (One Forward, One Backward)

각 GPU가 forward와 backward를 번갈아 실행하여 메모리를 절약함.

---

## Sequence Parallel (SP)

Tensor Parallel과 함께 사용함. LayerNorm과 Dropout처럼 TP로 분산하기 어려운 연산을 **시퀀스 차원으로 분산**함:

```
TP region:   [Attention (head parallel)] [FFN (column/row parallel)]
SP region:   [LayerNorm (seq parallel)]  [Dropout (seq parallel)]
```

시퀀스를 chunk로 나눠 각 GPU가 다른 portion을 처리함.

---

## Context Parallel

매우 긴 시퀀스(128K+)를 여러 GPU에 분산함.

**Ring Attention**: GPU들이 ring topology로 연결되어, 각 GPU는 local window를 처리하면서 K, V를 순차적으로 다음 GPU로 전달함. Flash Attention과 결합하여 사용함.

---

## FSDP (Fully Sharded Data Parallel)

PyTorch에서 제공하는 메모리 효율적인 분산 학습.

```
ZeRO-3와 유사:
  - Model parameters: 각 GPU에 1/N씩 분산
  - Gradient: 각 GPU에 1/N씩 분산
  - Optimizer state: 각 GPU에 1/N씩 분산

Forward/Backward 시:
  필요한 파라미터를 그 때 AllGather로 모아서 연산
  연산 후 즉시 해제
```

**메모리**: DDP 대비 최대 N배 절약 (N: GPU 수)

---

## ZeRO (Zero Redundancy Optimizer)

DeepSpeed의 메모리 최적화 기법. 단계적으로 메모리를 절약함.

### 메모리 분석 (mixed precision 기준)

```
16B 파라미터 모델의 메모리:
  Parameters (BF16):    32 GB
  Gradients (FP32):     64 GB
  Optimizer states (Adam, FP32): 128 GB
  TOTAL:               ~224 GB
```

### ZeRO Stage 1: Optimizer State Sharding

Optimizer states만 분산함. 파라미터와 gradient는 각 GPU에 복사본 유지.
메모리: 224 GB → ~60 GB (N=8 기준)

### ZeRO Stage 2: Gradient Sharding

Optimizer states + Gradient를 분산함.
메모리: 224 GB → ~30 GB

### ZeRO Stage 3: Parameter Sharding

모든 것(파라미터 + gradient + optimizer state)을 분산함.
메모리: 224 GB → ~8 GB (N=8 기준)

단, forward/backward 시 AllGather/ReduceScatter 통신이 더 많이 발생함.

### ZeRO-Infinity

CPU RAM, NVMe SSD도 메모리로 활용함. GPU 메모리가 부족할 때 파라미터와 optimizer state를 CPU/NVMe로 offload함. 학습 속도는 느려지지만 매우 큰 모델도 학습 가능함.

---

## DeepSpeed

Microsoft에서 개발한 LLM 학습 라이브러리.

**주요 기능**:
- ZeRO Stage 1/2/3 + ZeRO-Infinity
- Mixed precision (BF16, FP8)
- Activation checkpointing
- Pipeline parallel

```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config={"zero_optimization": {"stage": 3}, ...}
)
```

---

## Megatron-LM

NVIDIA에서 개발한 LLM 학습 프레임워크로, **3D Parallelism**을 지원함.

```
3D Parallelism = TP × PP × DP

예시: 1024 GPU
  TP = 8   (같은 node의 NVLink GPU들)
  PP = 16  (서로 다른 node 간)
  DP = 8   (data replicas)
  → 8 × 16 × 8 = 1024 GPUs
```

LLaMA-3 405B, GPT-4 같은 초대형 모델 학습에 사용됨.

---

## AllReduce / AllGather 통신

### AllReduce

모든 GPU의 tensor를 **합산(또는 평균)**하여 모든 GPU에 동일한 결과를 전달함.

```
GPU 0: [1, 2, 3]
GPU 1: [4, 5, 6]  → AllReduce(sum) → GPU 0: [5, 7, 9]
GPU 2: [0, 0, 0]                    → GPU 1: [5, 7, 9]
                                    → GPU 2: [5, 7, 9]
```

DDP에서 gradient 평균, TP에서 row parallel의 결과 합산에 사용함.

### AllGather

각 GPU가 가진 tensor 조각을 **모아서** 모든 GPU에 전달함.

```
GPU 0: [a, b]
GPU 1: [c, d]  → AllGather → GPU 0: [a, b, c, d]
GPU 2: [e, f]              → GPU 1: [a, b, c, d]
                           → GPU 2: [a, b, c, d]
```

ZeRO Stage 3에서 분산된 파라미터를 모을 때 사용함.

### ReduceScatter

AllReduce = ReduceScatter + AllGather

```
ReduceScatter: 합산 후 각 GPU에 조각을 분산
AllGather: 각 GPU의 조각을 모아서 전달
```

---

## Checkpoint Sharding

대형 모델의 체크포인트를 여러 파일로 분산 저장함.

```
llama-3-405b/
  model-00001-of-00191.safetensors
  model-00002-of-00191.safetensors
  ...
  model.safetensors.index.json  ← 어떤 파라미터가 어떤 파일에 있는지 매핑
```

**huggingface safetensors** 포맷이 안전하고 빠른 표준.

---

## Fault Tolerance & Elastic Training

### Fault Tolerance

수백~수천 GPU 학습에서 노드 장애는 불가피함.

**전략**:
1. 주기적 checkpoint (보통 수백~수천 step마다)
2. 장애 감지 시 마지막 checkpoint에서 재시작
3. 실패한 노드를 대체 노드로 교체

**AsyncCheckpoint**: 체크포인트를 비동기로 저장하여 학습 interruption 최소화

### Elastic Training

학습 중 GPU 수를 동적으로 추가하거나 줄임.

**PyTorch Elastic**: 노드 추가/제거 시 DDP 그룹을 재구성함.
주로 클라우드 환경에서 spot instance 활용 시 유용함.
