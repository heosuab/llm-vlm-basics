# GPU 아키텍처 & 하드웨어 최적화

## GPU 아키텍처 기초

### 계층 구조
```
GPU
├── GPC (Graphics Processing Cluster)
│   ├── TPC (Texture Processing Cluster)
│   │   └── SM (Streaming Multiprocessor) × 2
│   └── ...
└── ...

SM (H100 기준):
  ├── 128 CUDA Cores (FP32)
  ├── 4 Tensor Core 그룹 (4th gen)
  ├── 256KB L1 Cache / Shared Memory (SRAM)
  ├── Register File (65536 × 32-bit)
  └── Warp Schedulers × 4

H100 SXM5:
  132 SMs × (128 FP32 cores + Tensor Cores)
  CUDA Cores: 16,896
  Tensor Cores: 528 (4th gen, FP8/BF16/FP16/INT8)
```

### 메모리 계층
```
                속도       크기       레이턴시
Register File:  매우 빠름    64KB/SM    ~1 cycle
L1/Shared(SRAM): 매우 빠름  256KB/SM   ~20-30 cycles
L2 Cache:       빠름        50MB(H100)  ~200 cycles
HBM (DRAM):     느림        80GB(H100)  ~400-600 cycles

대역폭:
  HBM3 (H100): 3.35 TB/s
  L2:          ~10 TB/s (effective)
  L1/Shared:   ~100 TB/s (per SM, 이론)

핵심 insight:
  LLM 추론은 대부분 HBM 대역폭이 병목
  → 계산보다 메모리 접근이 느림 (Memory Bound)
```

---

## 주요 GPU 스펙 비교

```
┌─────────────────┬──────────┬──────────┬──────────┬──────────┐
│ GPU             │ A100 80G │ H100 80G │ H200 80G │ MI300X   │
├─────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Memory          │ 80GB HBM2│ 80GB HBM3│141GB HBM3│192GB HBM3│
│ Mem BW          │ 2 TB/s   │ 3.35 TB/s│4.8 TB/s  │ 5.3 TB/s │
│ BF16 TFLOPs     │ 312      │ 989      │ 989      │ 1307     │
│ FP8 TFLOPs      │ -        │ 1979     │ 1979     │ 2614     │
│ NVLink BW       │ 600 GB/s │ 900 GB/s │ 900 GB/s │ -        │
│ TDP             │ 400W     │ 700W     │ 700W     │ 750W     │
├─────────────────┼──────────┼──────────┼──────────┼──────────┤
│ 주용도          │학습 표준  │현재 표준  │긴 컨텍스트│추론 특화  │
└─────────────────┴──────────┴──────────┴──────────┴──────────┘

A10G (추론용):
  24GB GDDR6, 600 GB/s, 125 TFLOPS BF16
  AWS g5 인스턴스, 비용 효율적

RTX 4090 (소비자용):
  24GB GDDR6X, 1 TB/s, 165 TFLOPS BF16
  로컬 추론/개발, SXM 형태 아님
```

---

## Roofline 모델

### 개념
```
하드웨어 성능 한계를 분석하는 프레임워크

산술 강도 (Arithmetic Intensity) = FLOPs / 메모리 접근 bytes

Roofline:
  Performance [FLOPs/s] = min(
    compute_peak [FLOPs/s],
    bandwidth [bytes/s] × arithmetic_intensity [FLOPs/byte]
  )

Ridge Point (분기점):
  arithmetic_intensity = compute_peak / bandwidth

H100 Ridge Point:
  989 × 10¹² / (3.35 × 10¹²) ≈ 295 FLOPs/byte

  AI < 295: Memory Bound (대역폭이 병목)
  AI > 295: Compute Bound (계산이 병목)
```

### LLM 연산 분석
```
GEMM (행렬 곱):
  M×K × K×N 행렬 곱
  FLOPs: 2MKN
  메모리: (MK + KN + MN) bytes
  AI = 2MKN / ((MK + KN + MN) × 2)
  → M, N, K 크면 AI 높음 (Compute Bound)

Prefill (배치 처리):
  seq_len = 1024, d_model = 4096, BF16
  GEMM (1024×4096) × (4096×4096):
    FLOPs: 2 × 1024 × 4096 × 4096 ≈ 34 GFLOPs
    Memory: (1024×4096 + 4096×4096 + 1024×4096) × 2 ≈ 50 MB
    AI ≈ 34×10⁹ / 50×10⁶ ≈ 680 FLOPs/byte → Compute Bound ✓

Decode (토큰 1개씩):
  seq_len = 1, d_model = 4096
  GEMV (1×4096) × (4096×4096):
    FLOPs: 2 × 1 × 4096 × 4096 ≈ 33.5 MFLOPs
    Memory: 4096×4096 × 2 ≈ 32 MB (weight load)
    AI ≈ 33.5×10⁶ / 32×10⁶ ≈ 1 FLOPs/byte → Memory Bound ✗

결론:
  Prefill: Compute Bound → GPU utilization 높음
  Decode: Memory Bound → 대역폭이 병목, GPU 낭비
```

### Decode 병목 해결 방법
```
문제: Decode 시 매 step마다 모든 weight 로드 (GEMV)
  8B 모델 BF16: 16GB weight / 3.35 TB/s ≈ 4.8 ms/step 최소
  → 최대 208 tokens/sec (단일 GPU, batch=1)

해결책:
1. Batching (여러 요청 묶기):
   batch=64: GEMV → GEMM으로 변환 → Compute Bound
   → throughput 64× 향상 (latency도 약간 증가)

2. 양자화 (Quantization):
   BF16 → INT4: weight 크기 4× 감소 → bandwidth 4× 감소
   → decode 속도 ~2-3× 향상 (overhead 있음)

3. Speculative Decoding:
   작은 모델로 여러 토큰 예측 → 큰 모델이 한 번에 검증
   → 같은 메모리 bandwidth로 더 많은 토큰

4. Flash Decoding:
   Attention 계산을 KV cache 병렬로 처리
   → decode attention의 병렬성 향상
```

---

## 연산 최적화

### Tensor Core 활용
```
Tensor Core (H100, 4th gen):
  BF16: 4×8 × 8×8 행렬 곱 → 4×8 결과 (1 cycle)
  FP8: 8×16 × 16×8 행렬 곱 (2× 처리량)
  INT8: 2× BF16 처리량
  FP16/INT4: 지원

필요 조건:
  행렬 차원: 8의 배수 (BF16/FP16), 16의 배수 (INT8/INT4)
  메모리 정렬: 128 bytes 정렬 권장

cuBLAS: GEMM에 자동 Tensor Core 사용
→ 커스텀 코드에서 반드시 차원 체크 필요
```

### CUDA 최적화 기초
```
Memory Coalescing (메모리 합치기):
  한 warp (32 threads)가 연속 주소 접근 → 1 memory transaction
  비연속 접근 → 여러 transaction → 느림

  예: 2D 배열에서 행 vs 열 접근
  행 접근: coalesced (빠름)
  열 접근: strided (느림, N× 더 많은 transaction)

Thread Divergence:
  한 warp 내 if-else 분기 → 순차 실행 (병렬성 손실)
  → 분기 최소화, 조건문 warp 경계 정렬

Shared Memory 사용:
  자주 접근하는 데이터를 HBM에서 SRAM으로
  Bank conflict 주의 (32 banks, stride 회피)

Occupancy:
  active warps / max warps per SM
  높을수록 레이턴시 숨기기 좋음
  Register/Shared memory 사용량이 occupancy 결정
```

### Kernel Fusion
```
개념: 여러 작은 커널을 하나로 합치기
  → HBM round-trip 최소화

예: LayerNorm + Linear + GELU
  전통: 각각 별도 커널
    HBM read(x) → compute LN → HBM write
    HBM read(x_ln) → compute Linear → HBM write
    HBM read(x_lin) → compute GELU → HBM write

  Fused: 한 커널에서 처리
    HBM read(x) → LN → Linear → GELU → HBM write(x_out)
    → HBM 접근 3× → 1×

Flash Attention = Attention의 kernel fusion
  기존: QK^T → softmax → AV (3 커널, 각각 HBM write/read)
  Flash: SRAM 내에서 전부 처리 → HBM 접근 O(N²) → O(N)
```

---

## 프로파일링 도구

### nsys (Nsight Systems) — 타임라인 분석
```bash
# 실행
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=profile_report \
  python train.py

# 분석 포인트:
# - GPU idle time (흰 구간) → 병목 지점
# - NCCL 통신 vs compute 겹침
# - CPU/GPU 동기화 지점
# - Data loading bottleneck
```

### ncu (Nsight Compute) — 커널 분석
```bash
# 특정 커널 분석
ncu --metrics l1tex__t_bytes.sum,sm__throughput.avg.pct \
    python train.py

# 주요 지표:
# l1tex__t_bytes: L1 cache 접근량 (coalescing 확인)
# sm__warps_active: occupancy
# gpu__time_duration: 커널 실행 시간
# memory_l2_theoretical_sectors: L2 이상적 접근
# smsp__sass_thread_inst_executed_op_fadd.sum: FP32 FLOP 수
```

### torch.profiler
```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("model_forward"):
        output = model(input)

# 결과 출력
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=20))

# Chrome trace 내보내기
prof.export_chrome_trace("trace.json")

# TensorBoard
prof.export_stacks("profiler_stacks.txt", "self_cuda_time_total")
```

### 메모리 프로파일링
```python
# 메모리 통계
print(torch.cuda.memory_summary(device=None, abbreviated=False))

# 커스텀 모니터링
def log_memory(step):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    print(f"Step {step}: {allocated:.2f}GB alloc, "
          f"{reserved:.2f}GB reserved, {max_allocated:.2f}GB peak")

# Memory snapshot (PyTorch 2.0+)
torch.cuda.memory._record_memory_history()
# ... training ...
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
# → memory_viz.py로 시각화
```

---

## Model FLOPs Utilization (MFU)

### 개념 및 계산
```
MFU = 실제 처리량 / 이론적 최대 처리량

실제 처리량 = tokens/sec (측정값)
이론 최대 = GPU_FLOPS / (FLOPs per token)

FLOPs per token (forward pass):
  Attention: 4 × n × d (self-attention QKVO projection)
           + 4 × n² (attention matrix, short seq에선 작음)
  FFN:      2 × n × d_model × 4 × d_model ≈ 8nd²
  합계: ≈ 2 × (12 × d² × L) ≈ 24d²L per token
  (L=레이어 수, 근사치)

더 정확하게는:
  ≈ 2 × N_params (비임베딩 파라미터)
  이유: 각 param이 forward에서 2 FLOPs (mul+add)

예: LLaMA-3-8B
  N_params_no_embed ≈ 7.5B
  FLOPs/token ≈ 15 GFLOPs

  실제 측도: 5000 tokens/sec (8× A100)
  이론 최대: 312×10¹² × 8 / 15×10⁹ = 166,400 tokens/sec
  MFU = 5000 / 166,400 ≈ 3% (매우 낮음)

실제 달성 MFU:
  Flash Attention + ZeRO-2 + 최적화: 30-50%
  최고 수준 (Megatron-LM): 50-60%

왜 낮나?:
  - 통신 오버헤드
  - Data loading bottleneck
  - Memory Bound 연산
  - 불균형 계산
```

---

## CUDA 커널 작성 (Triton)

### Triton 소개
```python
# Triton: Python 기반 GPU 커널 작성 (NVIDIA CUDA 대체)
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

# 호출
grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
```

### Flash Attention Triton 핵심 패턴
```python
@triton.jit
def flash_attention_forward(
    Q, K, V, Out,
    stride_qm, stride_qk,  # Q tensor strides
    N_CTX: tl.constexpr,   # sequence length
    BLOCK_M: tl.constexpr, # block size for queries
    BLOCK_N: tl.constexpr, # block size for keys
):
    # program_id: (batch × head, block_m_idx)
    start_m = tl.program_id(0)

    # Q 블록 로드 (SRAM으로)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q = tl.load(Q + offs_m[:, None] * stride_qm + ...)

    # 누적 변수 초기화 (online softmax)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # K, V 블록 순회
    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K + ...)
        v = tl.load(V + ...)

        # Attention score
        qk = tl.dot(q, tl.trans(k))  # SRAM 내에서 계산

        # Online softmax 업데이트
        m_ij = tl.max(qk, axis=1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i = alpha * l_i + beta * l_ij
        acc = alpha[:, None] * acc + tl.dot(p.to(tl.float16), v)
        m_i = m_i_new

    # 최종 정규화 후 HBM에 저장
    out = acc / l_i[:, None]
    tl.store(Out + ..., out.to(tl.float16))
```

---

## 실전: 추론 최적화 파이프라인

### 처리량 vs 지연시간 트레이드오프
```
Latency-oriented (단일 요청 빠르게):
  - 배치 크기 1
  - 작은 모델 또는 INT4 양자화
  - Speculative decoding (작은 draft 모델)

Throughput-oriented (많은 요청 처리):
  - 큰 배치 (continuous batching)
  - Tensor Parallelism으로 처리량 증가
  - 양자화로 더 큰 배치 가능

GPU 선택:
  H100: 학습 + 대규모 추론
  A100: 안정적 추론 워크로드
  H200: 긴 컨텍스트 추론 (메모리 141GB)
  MI300X: 192GB 메모리, 추론 비용 효율 (AMD)
  A10G: 소규모 추론, 비용 효율
```

### 모델 별 최적 하드웨어
```
7B 모델 (BF16):
  단일 A100 80GB → 가능 (여유 있음)
  INT4: 단일 A10G 24GB도 가능

70B 모델 (BF16):
  2× A100 80GB → 가능 (TP=2)
  INT4: 단일 A100 80GB 가능 (~35GB)

405B 모델 (BF16):
  8× A100 → 불가 (640GB < 810GB)
  → 8× H200 (1128GB) 가능
  INT4: 4× A100 (320GB > 202GB) 가능
```

---

## Further Questions

**Q1. LLM decode가 왜 Memory Bound인가?**
```
Decode: batch=1, seq에서 1토큰 생성
  → 모든 weight matrix에 대해 GEMV (행렬-벡터 곱)
  → Weight 크기 (예: 16GB for 8B BF16) 전부 HBM에서 로드
  → 연산: 2N FLOPs (매우 적음)
  → AI = 2N / (2N bytes) = 1 FLOPs/byte << ridge point (295)
  → Memory Bound 확정

해결: 배치 크기 증가 (배치가 크면 GEMM으로 전환 → Compute Bound)
```

**Q2. Tensor Core를 쓰려면 무엇을 지켜야 하나?**
```
1. 행렬 차원: BF16/FP16 → 8의 배수, INT8 → 16의 배수
2. 메모리 정렬: 128 bytes 정렬
3. 데이터 타입: FP32 accumulation (cuBLAS 기본값)
4. cuBLAS/cuDNN 사용: 자동 Tensor Core 활성화
   수동 CUDA: wmma API 또는 Triton tl.dot

모델 설계 시: d_model, d_ffn을 64 또는 128의 배수로
예: d_model=5000 → 비효율, d_model=4096 또는 5120 권장
```

**Q3. MFU가 낮을 때 병목 찾는 방법은?**
```
1. GPU utilization: nvidia-smi → GPU 사용률 낮으면 CPU 병목
2. nsys profile: 타임라인에서 GPU idle 구간 확인
   → NCCL 통신이 많으면: TP/PP degree 조정
   → Data loading이면: num_workers 증가, prefetch
3. ncu: 커널별 메모리 vs compute 분석
   → memory bound 커널: 양자화 또는 fusion 고려
4. 통신 오버헤드:
   profile NCCL ops 시간 비율
   → TP 줄이고 DP 늘리기 고려
```

**Q4. Flash Attention이 빠른 이유를 하드웨어 관점에서 설명하라.**
```
기존 Attention 병목:
  N×N attention matrix = N² 요소
  N=4096: 16M × 2bytes = 32MB HBM 쓰기+읽기
  각 연산(QK^T, softmax, AV)마다 HBM 왕복

Flash Attention:
  SRAM(256KB/SM)에서 블록 단위 처리
  QK^T → softmax → AV를 한 번에 처리 (SRAM 내)
  HBM: input Q, K, V 읽기 + output 쓰기만 (O(N))

속도 향상:
  HBM bandwidth: 3.35 TB/s
  SRAM throughput: 훨씬 높음
  → HBM 접근 횟수 감소 = 직접적 속도 향상

실제: N=4096에서 2-4× 빠름, N=16384에서 더 큰 향상
```

**Q5. BF16 vs FP16 학습 차이는?**
```
FP16: 1 sign + 5 exponent + 10 mantissa (max ≈ 65504)
BF16: 1 sign + 8 exponent + 7 mantissa (max ≈ 3.4×10³⁸)

BF16 장점:
  - 표현 범위가 FP32와 동일 (같은 exponent bits)
  - Loss scaling 불필요 (FP16은 overflow/underflow 발생)
  - Gradient가 큰 경우 BF16이 안정적

FP16 장점:
  - 더 높은 mantissa 정밀도 (10 vs 7 bits)
  - 일부 오래된 하드웨어 (V100)는 BF16 미지원

현대 LLM: BF16 표준 (A100+, H100)
  mixed precision: forward/backward BF16, master weights FP32
```
