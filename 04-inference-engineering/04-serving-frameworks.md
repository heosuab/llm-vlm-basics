# 추론 서빙 프레임워크

## 프레임워크 비교 개요

| 프레임워크 | 개발사 | 특징 | 최적 사용 |
|-----------|--------|------|---------|
| vLLM | UC Berkeley | 범용, PagedAttention, 모든 GPU | 범용 프로덕션 |
| TGI | HuggingFace | HF 생태계, Docker 친화 | HF 모델 빠른 배포 |
| TensorRT-LLM | NVIDIA | NVIDIA 최적화, 최고 성능 | NVIDIA 엔터프라이즈 |
| SGLang | Stanford | 복잡한 프로그래밍, RadixAttention | 다단계 추론 |
| DeepSpeed-FastGen | Microsoft | ZeRO 기반, 대형 모델 | 수백B 모델 |
| Ollama | Ollama | 로컬, 사용 편의, 원클릭 | 개발/테스트 |
| llama.cpp | ggerganov | CPU, GGUF, 엣지 | 엣지/오프라인 |

---

## vLLM

### 핵심 기능

```
PagedAttention:
  - KV Cache를 비연속 메모리 블록으로 관리
  - OS의 가상 메모리 페이징과 유사
  - 메모리 단편화 거의 0
  - 동적 메모리 할당 → 배치 크기 유연

Continuous Batching:
  - 요청이 완료되는 즉시 새 요청 삽입
  - GPU 활용률 극대화

Prefix Caching:
  - 동일 prefix의 KV Cache 재사용
  - Radix Tree 기반 자동 탐지

Chunked Prefill:
  - Prefill을 청크로 분할
  - Prefill 중에도 Decode 요청 처리 가능

Speculative Decoding:
  - Draft model로 여러 토큰 후보 생성
  - 대형 모델로 한 번에 검증

Multi-GPU 지원:
  - Tensor Parallelism (intra-node)
  - Pipeline Parallelism (inter-node)
```

### vLLM 사용법

```python
from vllm import LLM, SamplingParams

# 기본 초기화
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=2,          # TP degree
    gpu_memory_utilization=0.9,      # GPU 메모리 사용률
    max_model_len=8192,              # 최대 시퀀스 길이
    enable_prefix_caching=True,      # prefix 캐싱
    enable_chunked_prefill=True,     # chunked prefill
    quantization="awq",              # AWQ 양자화
    dtype="bfloat16",
    # speculative_model="facebook/opt-125m",  # 스펙큘레이티브 디코딩
    # num_speculative_tokens=5,
)

# 샘플링 파라미터
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_tokens=512,
    stop=["<|eot_id|>", "\n\nHuman:"],  # 정지 토큰
    repetition_penalty=1.1,
)

# 배치 추론 (최적 효율)
prompts = [
    "What is machine learning?",
    "Explain transformer architecture.",
    "What is RLHF?",
]
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

### vLLM OpenAI 호환 서버

```bash
# 서버 실행
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key your-secret-key

# OpenAI SDK로 호출
from openai import OpenAI

client = OpenAI(
    api_key="your-secret-key",
    base_url="http://localhost:8000/v1"
)

# 스트리밍
stream = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Tell me about LLMs"}],
    stream=True,
    temperature=0.7,
    max_tokens=512,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### vLLM Docker 배포

```dockerfile
FROM vllm/vllm-openai:latest

ENV MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
ENV TENSOR_PARALLEL_SIZE=2

CMD ["python", "-m", "vllm.entrypoints.openai.api_server",
     "--model", "$MODEL_NAME",
     "--tensor-parallel-size", "$TENSOR_PARALLEL_SIZE",
     "--gpu-memory-utilization", "0.9"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  vllm:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
          - driver: nvidia
            count: 2
            capabilities: [gpu]
    volumes:
    - ~/.cache/huggingface:/root/.cache/huggingface
    ports:
    - "8000:8000"
    command: >
      python -m vllm.entrypoints.openai.api_server
      --model meta-llama/Meta-Llama-3-8B-Instruct
      --tensor-parallel-size 2
```

---

## TensorRT-LLM

### 특징 및 아키텍처

```
특징:
  - NVIDIA GPU 전용 최고 성능
  - 커스텀 CUDA/Triton 커널
  - INT8/FP8/INT4 양자화 native
  - In-flight batching
  - Tensor Parallelism, Pipeline Parallelism
  - AWQ, GPTQ, SmoothQuant 지원

성능 (A100 80GB 기준):
  Llama-3-8B: vLLM 대비 ~30-50% 처리량 향상
  단점: 빌드 시간 길고 (수 시간), NVIDIA 전용
```

### TensorRT-LLM 빌드 프로세스

```bash
# 1. 모델 가중치 변환 (HuggingFace → TensorRT-LLM 체크포인트)
python convert_checkpoint.py \
    --model_dir /path/to/llama-3-8b \
    --output_dir /tmp/trtllm_ckpt \
    --dtype bfloat16 \
    --tp_size 2  # Tensor Parallelism

# 2. TensorRT 엔진 빌드 (시간 오래 걸림)
trtllm-build \
    --checkpoint_dir /tmp/trtllm_ckpt \
    --output_dir /tmp/trtllm_engines \
    --gemm_plugin bfloat16 \
    --gpt_attention_plugin bfloat16 \
    --max_batch_size 32 \
    --max_input_len 2048 \
    --max_output_len 512 \
    --max_seq_len 2560

# 3. 엔진 실행 (직접)
python run.py \
    --engine_dir /tmp/trtllm_engines \
    --max_output_len 100 \
    --input_text "Hello, what is AI?"

# 4. Triton Inference Server 배포
tritonserver \
    --model-repository /path/to/triton_models \
    --log-verbose 1
```

### TensorRT-LLM Python API

```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner
import numpy as np

runner = ModelRunner.from_dir(
    engine_dir="/tmp/trtllm_engines",
    rank=tensorrt_llm.mpi_rank(),
)

# 배치 추론
batch_input_ids = [
    [1, 1234, 5678],  # 토큰화된 입력
    [1, 9999, 1111],
]

outputs = runner.generate(
    batch_input_ids=batch_input_ids,
    max_new_tokens=100,
    temperature=0.7,
    end_id=2,  # EOS 토큰 ID
)
```

---

## SGLang (Stanford, 2024)

### RadixAttention 원리

```
vLLM Prefix Caching:
  - 완전히 동일한 token sequence만 공유
  - 해시 기반: 정확히 일치해야 캐시 히트

SGLang RadixAttention:
  - Radix Tree (Trie 구조) 로 prefix 관리
  - 부분 공유도 가능 (공통 prefix 자동 탐지)
  - LRU 방식으로 오래된 캐시 제거

예시:
  요청 A: "System: You are helpful. User: What is AI?"
  요청 B: "System: You are helpful. User: What is ML?"

  공통 prefix: "System: You are helpful. User: What is "
  → 이 부분의 KV Cache 재사용 (Radix Tree로 자동 탐지)

  vLLM은 완전히 동일한 prefix만 캐싱 가능
  SGLang은 공통 부분 자동 추출 → 더 높은 캐시 히트율
```

### SGLang 사용법

```python
import sglang as sgl

# 서버 실행
# python -m sglang.launch_server --model meta-llama/Llama-3-8B-Instruct --port 30000

# 기본 생성
@sgl.function
def simple_qa(s, question: str):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=256, temperature=0.7))

with sgl.Session() as s:
    simple_qa.run(s, question="What is RAG?")
    print(s["answer"])

# 복잡한 다단계 생성 (SGLang 강점)
@sgl.function
def multi_step_analysis(s, text: str):
    # 1단계: 핵심 포인트 추출
    s += sgl.user(f"Extract key points from: {text}")
    s += sgl.assistant(sgl.gen("key_points", max_tokens=100))

    # 2단계: 각 포인트 상세 분석
    s += sgl.user(f"Analyze each point in detail: {s['key_points']}")
    s += sgl.assistant(sgl.gen("analysis", max_tokens=300))

    # 3단계: 요약
    s += sgl.user("Summarize the analysis in one paragraph.")
    s += sgl.assistant(sgl.gen("summary", max_tokens=100))

# 병렬 생성 (fork)
@sgl.function
def parallel_generation(s, prompt: str):
    s += sgl.user(prompt)
    # 다른 시각으로 동시 생성
    forks = s.fork(3)  # 3개 병렬 생성
    for fork in forks:
        fork += sgl.assistant(sgl.gen("response", max_tokens=100, temperature=0.9))

    # 가장 좋은 응답 선택 (또는 투표)
    s += sgl.user("Select the best response: " + str([f["response"] for f in forks]))
    s += sgl.assistant(sgl.gen("best", max_tokens=50))
```

---

## Text Generation Inference (TGI)

```bash
# Docker로 빠른 배포
docker run --gpus all \
  -v ~/.cache/huggingface:/data \
  -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:2.0 \
  --model-id meta-llama/Meta-Llama-3-8B-Instruct \
  --num-shard 2 \
  --max-input-length 4096 \
  --max-total-tokens 8192 \
  --max-batch-prefill-tokens 16384

# Python 클라이언트
from huggingface_hub import InferenceClient

client = InferenceClient("http://localhost:8080")

# 스트리밍
for token in client.text_generation(
    "What is machine learning?",
    max_new_tokens=200,
    stream=True,
    temperature=0.7,
):
    print(token, end="", flush=True)

# 구조화 출력 (JSON Schema)
import json
schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
response = client.text_generation(
    prompt,
    grammar={"type": "json", "value": schema}
)
```

**TGI 특징:**
```
Speculative Decoding: Medusa (다중 헤드)
Flash Attention: 자동 적용
Rope Scaling: 자동 지원
Quantization: GPTQ, AWQ, EETQ, bitsandbytes
HF Hub 통합: 모델 자동 다운로드
```

---

## llama.cpp

### 특징 및 최적화

```
C++ 구현 (의존성 최소):
  - CPU: BLAS (OpenBLAS, MKL), AVX2, AVX-512
  - GPU: CUDA, Metal (Apple Silicon), OpenCL
  - 부분 GPU 오프로딩: 일부 레이어만 GPU, 나머지 CPU

GGUF 포맷:
  - Q4_K_M: 4-bit, K-means 양자화, 중간 품질
  - Q5_K_M: 5-bit, 더 높은 품질
  - Q8_0: 8-bit, 거의 원본 품질
  - IQ3_XXS: 혁신적 3-bit (imatrix 방식)

Apple Silicon 최적화:
  - Metal 백엔드
  - Unified Memory (CPU+GPU 공유)
  - M3 Max: 96GB 통합 메모리 → 70B 모델 구동 가능
```

```python
# llama-cpp-python 사용
from llama_cpp import Llama

llm = Llama(
    model_path="./llama-3-8b-instruct.Q4_K_M.gguf",
    n_ctx=8192,           # 컨텍스트 길이
    n_gpu_layers=32,      # GPU 오프로드 레이어 수 (-1: 전체)
    n_batch=512,          # 배치 크기
    n_threads=8,          # CPU 스레드 수
    verbose=False,
)

# Chat 형식 사용
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is AI?"}
    ],
    temperature=0.7,
    max_tokens=200,
    stream=True,
)
for chunk in response:
    delta = chunk["choices"][0]["delta"]
    if "content" in delta:
        print(delta["content"], end="", flush=True)

# OpenAI 호환 서버 실행
# python -m llama_cpp.server --model model.gguf --n_gpu_layers 32
```

---

## Triton Inference Server (NVIDIA)

```
엔터프라이즈급 멀티모델 서빙:
  - 다양한 백엔드: TensorRT-LLM, vLLM, PyTorch, ONNX, TensorFlow
  - 동시 다중 모델 배포 및 버전 관리
  - A/B 테스팅 (앙상블)
  - 동적 배치 (시간/크기 기반)
  - Prometheus 메트릭 내장
  - gRPC + HTTP REST API
  - Kubernetes 친화적
```

```python
# Triton Python 클라이언트
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient("localhost:8000")

# 모델 상태 확인
model_metadata = client.get_model_metadata("llm_model")
print(model_metadata)

# 추론 요청
inputs = [
    httpclient.InferInput("input_ids", [1, 512], "INT32"),
]
inputs[0].set_data_from_numpy(np.array([[1, 2, 3, ...]], dtype=np.int32))

outputs = [
    httpclient.InferRequestedOutput("logits"),
]

response = client.infer("llm_model", inputs=inputs, outputs=outputs)
logits = response.as_numpy("logits")
```

```
Triton 모델 레포지토리 구조:
model_repository/
├── llm_ensemble/        # 앙상블 (전처리 + 모델 + 후처리)
│   ├── config.pbtxt
│   └── 1/
├── preprocessing/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py    # 토크나이저
├── llm_trtllm/
│   ├── config.pbtxt
│   └── 1/
│       └── llm/        # TensorRT-LLM 엔진
└── postprocessing/
    ├── config.pbtxt
    └── 1/
        └── model.py    # 디토크나이저
```

---

## 분산 추론

### Tensor Parallelism

```python
# vLLM에서 TP 설정
llm = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
    tensor_parallel_size=4,  # 4개 GPU에 걸쳐 TP
)

# TensorRT-LLM에서 TP
# convert_checkpoint.py --tp_size 4
# trtllm-build --tp_size 4

# 내부 동작:
# Q, K, V 행렬: 컬럼 방향 분할 (각 GPU가 일부 헤드 담당)
# FFN: Up/Gate 컬럼 분할, Down 행 분할
# All-Reduce: 각 레이어 끝에서 GPU간 합산
# 통신 오버헤드: NVLink > NVSwitch > PCIe
```

### Pipeline Parallelism

```python
# 레이어를 여러 노드에 분산
# vLLM에서 PP 설정
llm = LLM(
    model="meta-llama/Llama-3-405B-Instruct",
    tensor_parallel_size=8,    # 노드 내 TP
    pipeline_parallel_size=4,  # 노드 간 PP (vLLM 지원)
    # 총 32 GPU = 8 TP × 4 PP
)

# 내부 동작:
# Node 0: Layer 0-25   (micro-batch 1 처리 중 micro-batch 2 수신)
# Node 1: Layer 26-51
# Node 2: Layer 52-77
# Node 3: Layer 78-103
# Pipeline Bubble: 첫 micro-batch만 낭비
# 해결: 많은 micro-batch로 bubble 비율 감소
```

---

## 추론 최적화 기법

### Flash Attention 커널

```python
# Flash Attention 2 (Tri Dao)
from flash_attn import flash_attn_func

# 메모리 효율적 attention
# O(n) HBM reads (vs 표준 O(n²))
# 타일링 + 온라인 소프트맥스
output = flash_attn_func(
    q, k, v,
    dropout_p=0.0,
    causal=True,    # 인과적 마스킹
    softmax_scale=1.0 / math.sqrt(head_dim),
)

# Flash Attention 3 (H100 전용)
# FP8 지원, 2배 추가 속도향상
```

### CUDA 커널 최적화 (Triton)

```python
import triton
import triton.language as tl

@triton.jit
def fused_rms_norm_kernel(
    x_ptr, w_ptr, y_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm + 가중치 곱셈 융합 커널"""
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row_idx * n_cols + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)

    # RMS 계산
    x_squared = x * x
    rms = tl.sqrt(tl.sum(x_squared, axis=0) / n_cols + eps)

    # 정규화 + 가중치
    y = (x / rms) * w
    tl.store(y_ptr + row_idx * n_cols + cols, y, mask=mask)

# 여러 연산을 하나의 커널로 → HBM 읽기/쓰기 대폭 감소
```

### 양자화별 성능 비교

```
Llama-3-8B 기준 (A100 80GB):

| 방법 | VRAM | 처리량 | 품질 손실 |
|------|------|--------|---------|
| FP16 | 16GB | 1× | 0% |
| BF16 | 16GB | 1× | ~0% |
| INT8 (SmoothQuant) | 8GB | 1.5× | <1% |
| INT4 (GPTQ) | 5GB | 2× | 2-5% |
| INT4 (AWQ) | 5GB | 2× | 1-3% |
| FP8 (H100) | 8GB | 2× | ~0% |
| Q4_K_M (GGUF) | 5GB | - | 2-4% |
```

---

## 벤치마크 및 성능 측정

```python
import asyncio
import aiohttp
import time
import statistics

async def benchmark_throughput(
    api_url: str,
    prompts: list[str],
    concurrency: int = 32,
    model: str = "llama-3-8b"
) -> dict:
    """처리량 벤치마크"""

    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def send_request(prompt: str) -> dict:
        async with semaphore:
            start = time.perf_counter()
            first_token_time = None
            token_count = 0

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{api_url}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 200,
                        "stream": True,
                    },
                    headers={"Authorization": f"Bearer {API_KEY}"}
                ) as resp:
                    async for line in resp.content:
                        if line and line != b"data: [DONE]\n":
                            if first_token_time is None:
                                first_token_time = time.perf_counter() - start
                            token_count += 1

            total_time = time.perf_counter() - start
            return {
                "ttft": first_token_time,
                "total_time": total_time,
                "tokens": token_count,
                "tps": token_count / total_time if total_time > 0 else 0
            }

    tasks = [send_request(p) for p in prompts]
    results = await asyncio.gather(*tasks)

    ttfts = [r["ttft"] for r in results if r["ttft"]]
    return {
        "total_requests": len(prompts),
        "concurrency": concurrency,
        "ttft_p50": statistics.median(ttfts),
        "ttft_p99": sorted(ttfts)[int(len(ttfts)*0.99)],
        "mean_tps": statistics.mean([r["tps"] for r in results]),
        "total_throughput": sum([r["tokens"] for r in results]) / (
            max([r["total_time"] for r in results])
        )
    }

# 실행
results = asyncio.run(benchmark_throughput(
    api_url="http://localhost:8000",
    prompts=test_prompts,
    concurrency=32
))
print(f"TTFT P99: {results['ttft_p99']*1000:.1f}ms")
print(f"Throughput: {results['total_throughput']:.1f} tokens/sec")
```

---

## 프레임워크 상세 비교

```
vLLM vs TensorRT-LLM:
  - vLLM: 설치 쉬움, 다양한 모델/GPU 지원, 커뮤니티 활발
  - TRT-LLM: NVIDIA H100에서 20-40% 더 빠름, 빌드 복잡

vLLM vs SGLang:
  - vLLM: 범용, 더 많은 모델 지원
  - SGLang: 복잡한 프로그램 구조 (다단계, fork), 더 높은 캐시 히트율

vLLM vs TGI:
  - vLLM: 더 높은 처리량, 더 다양한 최적화
  - TGI: HuggingFace 통합, Docker 배포 간편

선택 가이드:
  범용 프로덕션: vLLM (대부분의 경우 최선)
  NVIDIA 최적화: TensorRT-LLM + Triton
  HF 모델 빠른 배포: TGI
  복잡한 다단계 추론/RAG: SGLang
  로컬 개발/테스트: Ollama
  엣지/오프라인: llama.cpp
  엔터프라이즈 멀티모델: Triton Inference Server
```

---

## Kubernetes 배포

```yaml
# vLLM Kubernetes Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm-server
  template:
    metadata:
      labels:
        app: vllm-server
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: "2"
          requests:
            nvidia.com/gpu: "2"
            memory: "64Gi"
        args:
        - "--model"
        - "meta-llama/Meta-Llama-3-8B-Instruct"
        - "--tensor-parallel-size"
        - "2"
        - "--gpu-memory-utilization"
        - "0.9"
        - "--enable-prefix-caching"
        ports:
        - containerPort: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 10
        env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token
              key: token
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm-server
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## Further Questions

**Q. vLLM과 TensorRT-LLM을 언제 선택하나?**
> vLLM: 범용, 빠른 개발, 다양한 모델/GPU 지원, 오픈소스 커뮤니티. TensorRT-LLM: NVIDIA H100에서 20-40% 더 높은 처리량 필요 시, 엔터프라이즈 NVIDIA 환경. 프로덕션 빠른 시작은 vLLM, 성능이 최우선이고 NVIDIA 환경이면 TRT-LLM. 빌드 복잡도(수 시간)와 성능 이득을 비교해 결정.

**Q. SGLang의 RadixAttention이 vLLM의 Prefix Caching과 다른 점?**
> vLLM Prefix Caching: 완전히 동일한 token sequence만 공유, 해시 기반. RadixAttention: Radix Tree(Trie) 구조로 부분 공유 가능, 자동 최장공통전위(LCP) 탐지. RAG나 few-shot 처럼 공통 시스템 프롬프트는 같고 나머지가 다른 패턴에서 SGLang이 더 높은 캐시 히트율.

**Q. 추론 서빙에서 KV Cache 메모리를 최적화하는 방법은?**
> 1) PagedAttention (vLLM): 비연속 블록 → 단편화 최소화. 2) Prefix Caching: 공통 prefix KV 재사용. 3) Quantized KV Cache: FP8/INT8로 KV 저장. 4) MQA/GQA: KV 헤드 수 줄임 (LLaMA-3에서 채택). 5) Sliding Window Attention: 긴 컨텍스트에서 Window만 유지. 6) StreamingLLM Attention Sink: 처음 몇 토큰 + 최근 Window만 유지.

**Q. 처리량 최대화를 위한 배치 크기 최적화는?**
> Continuous Batching이 기본 전제. GPU 메모리 = 모델 가중치 + KV Cache(배치 크기 × seq_len × 레이어 × KV 차원 × 2bytes). GPU 활용률 70-85% 목표. 너무 큰 배치: OOM. 너무 작은 배치: GPU 활용 낮음. vLLM의 --max-num-seqs 파라미터로 동시 처리 요청 수 제한. 실제 부하 테스트로 최적값 탐색.
