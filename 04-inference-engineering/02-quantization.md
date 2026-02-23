# 양자화 (Quantization)

## 기본 개념

```
왜 양자화인가?
  LLaMA-3-70B (BF16): 140 GB → A100 80GB 2장 필요
  LLaMA-3-70B (INT4):  35 GB → A100 80GB 1장으로!

FP32 → BF16 → INT8 → INT4 → INT2
  메모리: 절반, 절반, 절반으로 감소
  속도: 메모리 대역폭 bound 연산에서 2~4× 향상
  품질: 방법에 따라 거의 무손실 ~ 상당한 저하

용어:
  WxAy: Weight x-bit, Activation y-bit
  PTQ: Post-Training Quantization (학습 후)
  QAT: Quantization-Aware Training (학습 중)
```

---

## 양자화 수식

```
Uniform Quantization:
  양자화: X_q = round((X - zero_point) / scale)
  역양자화: X ≈ X_q × scale + zero_point

  scale = (X_max - X_min) / (2^bits - 1)

Symmetric (zero_point = 0):
  X_q = round(X / scale)
  scale = max(|X|) / (2^(bits-1) - 1)
  → INT8: [-127, 127] 범위

Asymmetric (zero_point ≠ 0):
  더 정확 (비대칭 분포에서)
  → INT8: [0, 255] 범위 활용 가능

Granularity (세분성):
  Per-tensor: 전체 텐서에 하나의 (scale, zp)
    → 이상치 하나가 전체 정밀도 저하
  Per-token: 각 행(토큰)마다 다른 (scale, zp)
    → Activation 양자화에 적합
  Per-channel: 각 열(채널)마다 다른 (scale, zp)
    → Weight 양자화에 적합, 더 정확
  Per-group: 연속 N개 요소마다 다른 (scale, zp)
    → GPTQ, GGUF에서 group_size=128
```

```python
import torch

def quantize_per_channel(weight: torch.Tensor, bits: int = 8) -> tuple:
    """Per-channel symmetric quantization"""
    # weight: (out_features, in_features)
    max_val = weight.abs().max(dim=1, keepdim=True).values  # (out, 1)
    q_max = 2 ** (bits - 1) - 1  # 127 for INT8

    scale = max_val / q_max  # (out, 1)
    weight_q = torch.clamp(torch.round(weight / scale), -q_max, q_max).to(torch.int8)

    return weight_q, scale

def dequantize_per_channel(weight_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """역양자화"""
    return weight_q.float() * scale

def quantize_per_group(weight: torch.Tensor, bits: int = 4, group_size: int = 128) -> tuple:
    """Per-group quantization (GPTQ/AWQ 방식)"""
    n_rows, n_cols = weight.shape
    assert n_cols % group_size == 0
    n_groups = n_cols // group_size

    # Reshape to groups: (n_rows * n_groups, group_size)
    w_grouped = weight.view(n_rows * n_groups, group_size)

    q_max = 2 ** (bits - 1) - 1
    max_val = w_grouped.abs().max(dim=1, keepdim=True).values
    scale = max_val / q_max  # (n_rows * n_groups, 1)

    w_q = torch.clamp(torch.round(w_grouped / scale), -q_max, q_max)

    return w_q.view(n_rows, n_cols), scale.view(n_rows, n_groups)
```

---

## LLM에서의 양자화 도전

```
LLM 특유의 문제:
  1. Activation 이상치 (Outlier):
     특정 차원에서 매우 큰 값 등장
     → Per-tensor INT8로 양자화 시 대부분 값이 0으로 압축
     → 큰 성능 저하

  2. 레이어별 다른 민감도:
     입력/출력 레이어 특히 민감
     Attention: weight 중요도 불균등

  3. 메모리 vs 속도 트레이드오프:
     INT4 weight: 메모리 ↓↓ but compute는 INT8/FP16으로 dequantize 필요
     INT8 weight+activation: 실제 INT8 GEMM으로 속도 ↑

  LLM.int8() (Dettmers et al., 2022):
    이상치 해결 방법:
      이상치(outlier) 차원 → FP16으로 처리
      나머지 차원 → INT8로 양자화
      → Mixed-precision 행렬 곱
```

---

## Post-Training Quantization (PTQ)

### GPTQ [Frantar et al., 2022]

```
목표: 각 레이어의 출력을 최대한 유지하는 INT4 찾기

핵심 알고리즘 (OBQ 기반):
  W_q = argmin ||WX - W_q X||²_F  (Frobenius norm)

  레이어별로:
    1. 보정 데이터 C개 샘플로 X 수집
    2. Hessian H = XX^T 계산 (입력 상관 행렬)
    3. 컬럼 순서대로 양자화:
       - 컬럼 i 양자화
       - 에러를 나머지 컬럼에 보상 (Cholesky 역행렬 활용)
       - Choleksy 업데이트로 효율화

수식:
  δW = -Q(w_i) × H^{-1}_{ii} × H_{i,:}  (에러 전파)

특성:
  - 보정 데이터 128 samples 필요 (C4, WikiText-2 등)
  - INT4 양자화 시 FP16 대비 ~1% 성능 저하
  - GPU에서 빠른 양자화 가능 (7B: ~30분)
  - Group quantization (group_size=128): 정밀도 향상

사용:
  AutoGPTQ, GPTQModel 라이브러리
  Hugging Face Transformers 통합
```

```python
# GPTQ 핵심 로직 (단순화 버전)
import torch

def gptq_quantize_layer(W: torch.Tensor, H: torch.Tensor,
                         bits: int = 4, group_size: int = 128) -> torch.Tensor:
    """
    GPTQ: 레이어 가중치 W를 Hessian H를 이용해 최적 양자화
    W: (out, in) 가중치 행렬
    H: (in, in) Hessian = X @ X.T
    """
    W = W.clone().float()
    n_rows, n_cols = W.shape

    # Hessian 역행렬 (Cholesky 분해)
    H = H.clone().float()
    H.diagonal().add_(1e-5)  # 수치 안정성
    L = torch.linalg.cholesky(H)  # H = L @ L.T
    H_inv = torch.cholesky_inverse(L)  # H^{-1}

    W_q = torch.zeros_like(W)
    q_max = 2 ** (bits - 1) - 1

    for i in range(n_cols):
        # 현재 컬럼의 scale 계산 (group-wise)
        group_idx = i // group_size
        group_start = group_idx * group_size
        group_end = min(group_start + group_size, n_cols)

        col = W[:, i]
        scale = col.abs().max() / q_max

        # 양자화
        col_q = torch.clamp(torch.round(col / scale), -q_max, q_max)
        W_q[:, i] = col_q

        # 역양자화 에러
        quant_error = col - col_q * scale  # (out,)

        # 에러를 나머지 컬럼에 전파: W[:, i+1:] -= error.outer(H_inv[i, i+1:] / H_inv[i, i])
        if i + 1 < n_cols:
            compensation = quant_error.unsqueeze(1) * (H_inv[i, i+1:] / H_inv[i, i]).unsqueeze(0)
            W[:, i+1:] -= compensation

    return W_q
```

### AWQ (Activation-aware Weight Quantization) [Lin et al., 2023]

```
핵심 관찰:
  가중치의 1%가 성능의 대부분 결정
  → 이 중요한 가중치는 "activation 크기가 큰 채널"에 위치

방법:
  1. 각 입력 채널 j의 중요도 s_j = mean(|X_j|)^α 측정
     (α: 튜닝 파라미터, 보통 0.5)
  2. Scale 변환:
     W̃ = W · diag(s)     (가중치 채널 스케일 ↑)
     X̃ = X / diag(s)^T  (activation 스케일 ↓)
  3. W̃를 균일 INT4로 양자화
  → 중요한 가중치 채널의 정밀도 보존

수학:
  Y = XW = X·(diag(s)⁻¹)·(diag(s)·W) = X̃·W̃
  (양자화 시 X̃는 low magnitude → quantization 쉬움)

특성:
  - GPTQ보다 빠른 양자화 (Hessian 계산 불필요)
  - 다양한 하드웨어 지원 (GPU, CPU, 모바일)
  - GPTQ와 비슷한 품질

라이브러리: autoawq
```

```python
import torch

def awq_scale_search(W: torch.Tensor, X: torch.Tensor,
                     alpha: float = 0.5, bits: int = 4) -> torch.Tensor:
    """
    AWQ: Activation-aware scale 탐색
    W: (out, in) 가중치
    X: (n_samples, in) 보정 데이터 activation
    """
    # 입력 채널별 activation 크기
    x_scale = X.abs().mean(dim=0)  # (in,)

    # 최적 alpha 그리드 탐색
    best_error = float('inf')
    best_s = torch.ones(W.shape[1])

    for a in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        s = x_scale ** a  # 채널 중요도 스케일

        # 스케일 변환
        W_scaled = W * s.unsqueeze(0)  # (out, in) * (in,) → (out, in)
        X_scaled = X / s.unsqueeze(0)  # (n_samples, in) / (in,)

        # INT4 양자화 (per-group)
        q_max = 2 ** (bits - 1) - 1
        scale_q = W_scaled.abs().max(dim=1, keepdim=True).values / q_max
        W_q = torch.clamp(torch.round(W_scaled / scale_q), -q_max, q_max)
        W_dq = W_q.float() * scale_q

        # 원래 스케일로 역변환
        W_restored = W_dq / s.unsqueeze(0)

        # 에러 측정 (보정 데이터 기준)
        err = ((W_restored @ X.T) - (W @ X.T)).pow(2).mean().item()

        if err < best_error:
            best_error = err
            best_s = s.clone()

    return best_s

def awq_quantize(W: torch.Tensor, s: torch.Tensor, bits: int = 4) -> tuple:
    """AWQ로 최적 scale 찾은 후 양자화"""
    W_scaled = W * s.unsqueeze(0)
    q_max = 2 ** (bits - 1) - 1
    scale_q = W_scaled.abs().max(dim=1, keepdim=True).values / q_max
    W_q = torch.clamp(torch.round(W_scaled / scale_q), -q_max, q_max)
    return W_q, scale_q, s  # 복원 시 / s 필요
```

### SmoothQuant [Xiao et al., 2022]

```
목표: W8A8 (가중치 + 활성화 모두 INT8)

문제: Activation이 Weight보다 양자화 어려움
  LLM activation: 특정 채널에서 매우 큰 이상치 (100× 이상)
  → Per-tensor INT8로 대부분 정보 손실

해결 (Activation Migration):
  Y = (X · diag(s)⁻¹) · (diag(s) · W)
    = X̃ · W̃

  X̃ = X / diag(s):   activation 이상치 줄임 → INT8 양자화 쉬움
  W̃ = diag(s) · W: weight가 이상치 흡수 → 여전히 양자화 가능

  s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
  α: migration strength (0.5 권장)

효과:
  Activation의 이상치를 Weight로 이동 (migration)
  → 두 텐서 모두 양자화 쉬워짐
  → W8A8 달성 (실제 INT8 GEMM → 속도 향상)

사용: vLLM FP8 quantization, TensorRT-LLM
```

---

## GGUF / llama.cpp

```
로컬 CPU/GPU 실행을 위한 경량 양자화 포맷

정밀도 옵션:
  Q8_0:  8비트 (per-block)  → 품질 최고, 속도 느림
  Q6_K:  6비트 (k-quant)    → 매우 좋음
  Q5_K_M: 5비트             → 좋음
  Q4_K_M: 4비트             → 추천 (품질/속도 균형)
  Q4_K_S: 4비트 small       → Q4_K_M보다 약간 작음
  Q3_K_M: 3비트             → 괜찮음
  Q2_K:  2비트              → 손실 큼

K 양자화:
  K-quants: block-wise 양자화 with scale 정보
  M/S/L: Medium/Small/Large variant (다른 레이어에 다른 정밀도)

실용적 선택:
  7B 모델 + 8GB VRAM → Q4_K_M (4.1GB)
  13B 모델 + 12GB VRAM → Q4_K_M (7.9GB)
  70B 모델 + 24GB VRAM → Q3_K_M (29GB) or Q4_K_M (40GB)

도구: Ollama, LM Studio, Jan
```

```bash
# GGUF 변환 및 양자화
pip install llama-cpp-python

# HuggingFace 모델 → GGUF 변환
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
python convert_hf_to_gguf.py meta-llama/Meta-Llama-3-8B \
    --outfile llama3-8b-f16.gguf --outtype f16

# 양자화
./llama-quantize llama3-8b-f16.gguf llama3-8b-q4_k_m.gguf Q4_K_M

# Python에서 사용
from llama_cpp import Llama

llm = Llama(
    model_path="llama3-8b-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=35,  # GPU로 올릴 레이어 수
    n_threads=8
)

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Explain quantization"}],
    max_tokens=512,
    temperature=0.7
)
print(response["choices"][0]["message"]["content"])
```

---

## QAT (Quantization-Aware Training)

```
아이디어:
  학습 시부터 양자화 효과 시뮬레이션
  → 양자화 에러에 강건한 가중치 학습

Straight-Through Estimator (STE):
  Forward: 실제 양자화 적용 (불연속)
    q = round(w / scale)
  Backward: 양자화 무시, gradient 직접 통과
    ∂L/∂w = ∂L/∂q  (STE)

  직관: round()는 미분 불가 → STE로 근사

효과:
  PTQ보다 품질 높음
  학습 비용 필요 (일반적으로 ~10% 추가 연산)
```

```python
import torch
import torch.nn as nn

class FakeQuantize(nn.Module):
    """QAT를 위한 Fake Quantization (STE)"""

    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits = bits
        self.q_min = -(2 ** (bits - 1))
        self.q_max = 2 ** (bits - 1) - 1

        # 학습 가능한 scale과 zero_point
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.zero_point = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward: 실제 양자화 (불연속)
        x_scaled = x / self.scale + self.zero_point
        x_clamped = torch.clamp(x_scaled, self.q_min, self.q_max)
        x_rounded = torch.round(x_clamped)  # 불연속

        # STE: backward는 round를 무시하고 gradient 직접 통과
        # x_rounded ≈ x_clamped (for gradient purposes)
        x_dequant = (x_rounded - self.zero_point) * self.scale

        # STE trick: stop gradient at quantization boundary
        return x + (x_dequant - x).detach()

class QuantizedLinear(nn.Module):
    """QAT용 Linear 레이어"""

    def __init__(self, in_features: int, out_features: int, bits: int = 4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.weight_quant = FakeQuantize(bits)
        self.act_quant = FakeQuantize(8)  # Activation은 보통 더 높은 비트

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weight
        w_q = self.weight_quant(self.linear.weight)
        # Quantize activation input
        x_q = self.act_quant(x)
        # Forward with quantized values
        return nn.functional.linear(x_q, w_q, self.linear.bias)
```

### BitNet [Wang et al., 2023]
```
1.58-bit LLM: 가중치를 {-1, 0, 1}로
  모든 Attention + FFN 1-bit/1.58-bit
  에너지 효율 10× 이상 향상
  하지만 BF16 모델 대비 성능 차이 아직 존재

BitNet b1.58 (Ma et al., 2024):
  ternary weights {-1, 0, 1}
  absmean quantization: scale = mean(|W|)
  W_q = sign(W) if |W| > threshold else 0

  장점:
    행렬 곱 → 덧셈/뺄셈만으로 (곱셈 없음)
    에너지 효율 획기적 향상
    Edge AI에서 주목
  단점:
    아직 FP16 대비 성능 차이
    사전 학습부터 필요 (PTQ 불가)
```

---

## FP8 (DeepSeek-V3, 2024 표준화)

```
FP8 타입:
  FP8 E4M3: 지수 4비트, 가수 3비트 → 정밀도 높음 (forward pass)
  FP8 E5M2: 지수 5비트, 가수 2비트 → 범위 넓음 (backward gradient)

H100 하드웨어 지원:
  FP8 Tensor Cores → 4× FP16 처리량
  H100: 989 TFLOPS (FP8), 495 TFLOPS (FP16)

DeepSeek-V3의 FP8 학습:
  Forward: FP8 GEMM (빠름)
  Backward: BF16 gradient (정확)
  Optimizer: BF16 master weights

  세부:
    FP8 quantization per-block (128 channels)
    FP8 E4M3: forward pass (활성화, 가중치)
    FP8 E5M2: backward pass (gradient)

결과:
  학습 속도 2× 향상 (FP16 대비)
  메모리 40% 절감
  품질 손실 거의 없음 (BF16 수준)

vLLM FP8 추론:
  vllm serve model --dtype fp8
  FP8 KV cache도 지원 (--kv-cache-dtype fp8)
```

---

## KV Cache 양자화

```
KV Cache 크기:
  LLaMA-3-8B, seq_len=2048:
  = 2 × 32 layers × 8 kv_heads × 128 d_head × 2048 × 2 bytes (BF16)
  = 268 MB

  긴 시퀀스 (32K tokens): 4.2 GB per sequence
  → 배치 처리 시 메모리 병목

KV 양자화 기법:
  KVQuant: adaptive (key 분포 기반)
  KIVI: INT2/INT4 KV cache (key: per-channel, value: per-token)
  Gear: group quantization

실제 구현:
  vLLM: --kv-cache-dtype fp8 (E5M2)
  TensorRT-LLM: FP8/INT8 KV cache
  llama.cpp: INT8 KV cache

권장:
  FP8 KV cache: 품질 유지하며 2× 메모리 절약
  INT4 KV cache: 4× 절약, 약간 품질 저하
```

---

## Mixed-Precision Quantization 전략

```
레이어별 민감도 분석:
  모든 레이어에 같은 비트 수 적용 비효율
  → 민감한 레이어: 더 높은 비트 (INT8)
  → 덜 민감한 레이어: 낮은 비트 (INT4 or INT2)

민감도 측정:
  각 레이어를 순차적으로 양자화 → Perplexity 변화 측정
  변화 큰 레이어 = 민감 = 더 높은 비트 배정

자동 탐색:
  NAS 방식으로 각 레이어 비트 수 탐색
  목표: 전체 모델 크기 제약 + 최대 성능

실용적 규칙:
  Attention Q/K/V: 민감함 (INT8)
  FFN down_proj: 민감함 (INT8)
  FFN up_proj, gate_proj: 덜 민감 (INT4)
  LM head: 매우 민감 (FP16)
```

```python
from transformers import BitsAndBytesConfig
import torch

# bitsandbytes 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # NormalFloat4 (정규분포 가중치에 최적)
    bnb_4bit_compute_dtype=torch.bfloat16,  # 계산은 BF16
    bnb_4bit_use_double_quant=True,  # 이중 양자화 (scale도 양자화)
)

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

# GPTQ 4-bit
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,  # activation 순서 고려 여부
)

model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantize_config=quantize_config,
)

# 보정 데이터로 양자화
examples = [{"input_ids": tokenizer("text", return_tensors="pt").input_ids}]
model.quantize(examples)
model.save_quantized("llama3-8b-gptq-4bit")
```

---

## 실전 양자화 가이드

### 상황별 선택
```
서버 배포 (GPU):
  최고 품질: FP16/BF16 (기본)
  메모리 절약: AWQ INT4 또는 GPTQ INT4
  속도 최우선: FP8 (H100)
  배치 처리: W8A8 (SmoothQuant)

로컬/엣지:
  CPU: GGUF Q4_K_M (균형)
  저사양 GPU: GGUF Q4_K_M
  고사양 로컬: AWQ INT4 (vLLM) 또는 GGUF Q5_K_M

품질 등급 (70B 기준):
  BF16 > AWQ/GPTQ INT4 ≈ GGUF Q8 > GGUF Q5 > GGUF Q4 > GGUF Q3
  실제: INT4와 BF16의 차이는 많은 태스크에서 1-2%
```

### Perplexity 기반 품질 평가
```python
# 양자화 모델 품질 측정
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_perplexity(model, tokenizer, texts):
    model.eval()
    total_loss = 0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
        total_tokens += inputs["input_ids"].size(1)

    ppl = torch.exp(torch.tensor(total_loss / total_tokens))
    return ppl.item()

# WikiText-103로 perplexity 비교
# FP16: 5.2, INT4: 5.5 (차이 작음)
# INT2: 8.3 (차이 큼)
```

### 양자화 품질 모니터링
```python
def compare_quantization_methods(model_name: str, test_texts: list):
    """여러 양자화 방법 비교"""
    results = {}

    # FP16 기준
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results['FP16'] = compute_perplexity(model_fp16, tokenizer, test_texts)
    del model_fp16

    # INT8 bitsandbytes
    model_int8 = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
    results['INT8_bnb'] = compute_perplexity(model_int8, tokenizer, test_texts)
    del model_int8

    # INT4 NF4
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    model_int4 = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    results['INT4_NF4'] = compute_perplexity(model_int4, tokenizer, test_texts)
    del model_int4

    # 결과 출력
    baseline = results['FP16']
    for method, ppl in results.items():
        degradation = (ppl - baseline) / baseline * 100
        print(f"{method}: PPL={ppl:.2f} (Degradation: +{degradation:.1f}%)")

    return results
```

---

## 양자화 비교

| 방법 | Bits | 메모리 | 속도 | 품질 | 용도 |
|------|------|--------|------|------|------|
| FP32 | 32 | 기준 | 기준 | 최고 | 훈련 |
| BF16 | 16 | 0.5× | 1.5× | 최고 | 서버 |
| FP8 | 8 | 0.25× | 2× | 매우 좋음 | H100 서버 |
| INT8 (LLM.int8) | 8 | 0.5× | 1.5~2× | 거의 동일 | 서버 |
| GPTQ INT4 | 4 | 0.25× | 2~3× | 약간 저하 | 서버/로컬 |
| AWQ INT4 | 4 | 0.25× | 2~3× | 약간 저하 | 서버/로컬 |
| GGUF Q4_K_M | 4 | 0.25× | 2~3× | 약간 저하 | CPU/로컬 |
| BitNet 1.58b | ~2 | 0.1× | 10×+ | 저하 (개선 중) | 미래 |

---

## Further Questions

**Q. GPTQ와 AWQ의 차이는?**
> GPTQ: Hessian 기반 레이어별 최적 양자화, 컬럼 순서대로 에러 보상 (정확하지만 느림). AWQ: Activation 중요도 기반 채널 스케일 조정 후 INT4 (빠르고 하드웨어 친화적). 품질은 비슷, AWQ가 다양한 하드웨어에 더 쉽게 배포 가능.

**Q. 양자화 에러가 레이어마다 다른 이유는?**
> 각 레이어의 activation/weight 분포 다름. 초기 레이어와 마지막 LM head 레이어 특히 민감. Attention layer vs FFN: 다른 이상치 패턴. Mixed-precision (특히 민감한 레이어는 INT8, 나머지 INT4)으로 품질-메모리 균형.

**Q. FP8 학습이 BF16과 품질 차이가 없는 이유는?**
> Gradient는 BF16 유지 (정밀도 필요). 행렬 곱만 FP8 (forward pass). Block-wise quantization (128 channels per block) → 이상치 영향 최소화. 학습 완료 후 모델은 BF16으로 저장 → 추론은 이전과 동일.

**Q. GGUF Q4_K_M이 일반적으로 추천되는 이유?**
> 품질: BF16 대비 약 3-5% PPL 상승 (대부분 태스크에서 거의 차이 없음). 메모리: BF16의 25% → 8GB VRAM으로 7B 모델 실행 가능. K-quant: block-wise scale로 per-channel보다 정밀. M variant: 중요 레이어(attention) INT8, 나머지 INT4 → 최적 균형.

**Q. W4A16 vs W8A8의 차이와 언제 각각 사용하나?**
> W4A16: 가중치만 INT4, 활성화는 FP16. Dequantize 후 FP16 GEMM → 메모리 효율 ↑, 속도는 제한적. W8A8: 가중치+활성화 모두 INT8. 실제 INT8 GEMM 가능 → 속도 ↑. H100/A100에서 W8A8 권장 (INT8 Tensor Core). 메모리 제한 환경에서는 W4A16.

**Q. Quantization-aware Training이 PTQ보다 나은 상황은?**
> 작은 모델 (7B 이하)에서 PTQ 품질 저하 클 때. 특수 도메인 (의료, 법률)에서 정확도 중요 시. INT2~INT4처럼 극단적 압축 필요 시. 비용: QAT는 기존 학습의 10-20% 추가 컴퓨트. 대형 모델(70B+)에서는 PTQ(GPTQ/AWQ)도 충분히 좋음.
