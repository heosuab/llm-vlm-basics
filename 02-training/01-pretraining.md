# Pretraining (사전학습)

## 학습 목표 (Objective Functions)

### 1. Causal Language Modeling (CLM) — Decoder-only

```
목표: 이전 토큰들로 다음 토큰 예측

Loss = -1/T · Σₜ₌₁ᵀ log P(xₜ | x₁,...,xₜ₋₁)

특징:
  - 모든 토큰 위치에서 동시에 학습 (효율적)
  - Causal mask: 미래 토큰 attend 불가
  - Autoregressive 생성과 학습 목표 일치
  - 사용: GPT 계열, LLaMA, Mistral, Qwen 등 모든 현대 LLM
```

### 2. Masked Language Modeling (MLM) — BERT 스타일

```
목표: 마스킹된 토큰 예측

마스킹 전략 (BERT):
  - 15% 토큰 선택
    - 80%: [MASK]로 교체
    - 10%: 랜덤 토큰으로 교체
    - 10%: 그대로 유지 (예측하면 보너스)

Loss = -Σᵢ∈masked log P(xᵢ | x_{모든 마스킹 안 된 토큰})

특징:
  - 양방향 컨텍스트 활용 (더 풍부한 표현)
  - 생성에 직접 사용 불가 (양방향이라)
  - 분류, 임베딩, NER 태스크에 강함
```

### 3. Fill-in-the-Middle (FIM) / Infilling

```
코드 자동완성에 특화:
  [PREFIX][MIDDLE][SUFFIX] 형태 학습

PSM 형식 (Prefix-Suffix-Middle):
  <fim_prefix> def add(a, b): <fim_suffix>
      return result <fim_middle> result = a + b <eot>

SFP 형식 (Suffix-First):
  <fim_suffix> ... <fim_prefix> ... <fim_middle> ...

학습 방법:
  - 50%: 일반 CLM (시퀀스 그대로)
  - 50%: FIM 변환 후 학습

사용: Starcoder, CodeLlama, DeepSeek-Coder
```

---

## Pretraining Data 구성

### LLaMA-3 데이터 구성 예시

```
총 토큰: 15T+ (LLaMA-3)

도메인별 비율:
  Web Text (CC 기반): ~45%
  Code: ~17%
  Wikipedia + Books: ~20%
  Academic/Scientific: ~8%
  기타 (대화, 뉴스 등): ~10%

LLaMA-3 특이점:
  - Synthetic data (합성 데이터) 포함
  - 코드 비율 대폭 증가 (LLaMA-2 대비)
  - 다국어 데이터 증가
  - 고품질 필터링 강화
```

### 데이터 토큰 수 확인

```python
# 토큰 수 계산 (대략적)
def estimate_tokens(text_size_gb: float, avg_chars_per_token: float = 4.0) -> float:
    """
    텍스트 크기에서 대략적인 토큰 수 추정
    영어 텍스트: 평균 ~4 chars/token
    코드: ~3.5 chars/token
    한국어: ~2 chars/token (UTF-8 바이트로는 더 많음)
    """
    chars = text_size_gb * 1e9
    tokens = chars / avg_chars_per_token
    return tokens

# 예시:
# C4 (305GB) → 약 176B tokens
# The Pile (825GB) → 약 300B tokens
# Common Crawl monthly (40TB raw, ~5TB cleaned) → 약 1T tokens
```

---

## 학습 인프라

### 병렬화 전략 3D Parallelism

```
대규모 LLM 학습의 표준: DP + TP + PP 조합

1. Data Parallelism (DP):
   같은 모델을 여러 GPU에 복사
   미니배치를 GPU별로 분할 처리
   각 GPU에서 gradient 계산 → All-Reduce로 평균

   DDP (Distributed Data Parallel):
     표준 방식
     gradient All-Reduce: O(2P/N) 통신
     N개 GPU에서 N배 배치 처리량

   FSDP (Fully Sharded Data Parallel, PyTorch):
     ZeRO-3와 유사
     모델 파라미터도 GPU들에 분산
     통신: gather(parameter) → compute → scatter

2. Tensor Parallelism (TP, Megatron-LM):
   행렬 연산 자체를 GPU들에 분할

   예: MLP에서
     W₁ [d → 4d]:
       GPU0: W₁_0 [d → 2d]  GPU1: W₁_1 [d → 2d]
       → 각 GPU에서 독립 계산
     GELU 적용
     W₂ [4d → d]:
       GPU0: W₂_0 [2d → d]  GPU1: W₂_1 [2d → d]
       → All-Reduce로 합산 [d]

   Attention에서:
     Multi-head를 GPU별로 나눔
     각 GPU: num_heads/N 개 head 처리

   통신: forward에서 All-Reduce 1회, backward에서 1회
   → 빠른 interconnect 필요 (NVLink 권장)

3. Pipeline Parallelism (PP):
   레이어를 GPU들에 순서대로 배치
   GPU0: Layer 1-8 (앞부분)
   GPU1: Layer 9-16 (중간)
   GPU2: Layer 17-24 (뒷부분)
   GPU3: Layer 25-32 (마지막)

   Micro-batching으로 bubble 최소화:
     배치를 여러 micro-batch로 분할
     GPU들이 파이프라인으로 처리
     bubble ratio = (p-1)/(m+p-1)  (p=stage 수, m=microbatch 수)

4. Context Parallelism (Sequence Parallelism):
   매우 긴 시퀀스 처리
   시퀀스를 GPU들에 분할
   Ring Attention: GPU들이 순환하며 Q×K 계산
```

### 3D 병렬화 설정 예시

```python
# Megatron-DeepSpeed 스타일 설정
parallelism_config = {
    "tensor_model_parallel_size": 8,    # TP=8
    "pipeline_model_parallel_size": 4,  # PP=4
    "data_parallel_size": "auto",       # 나머지 GPU

    # LLaMA-3 405B 학습 예시:
    # TP=8 × PP=8 × DP=... = 16,000 GPU 이상
}

# DeepSpeed ZeRO Stage 3
zero_config = {
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": True,
        "reduce_scatter": True,
        "overlap_comm": True,  # 통신-계산 overlap
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
        "stage3_gather_16bit_weights_on_model_save": True
    }
}
```

---

## Flash Attention 학습에서의 활용

```python
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

# Variable-length (packed sequences) 지원
from flash_attn import flash_attn_varlen_qkvpacked_func

def attention_forward_with_packing(
    qkv,          # [total_tokens, 3, nheads, headdim]
    cu_seqlens,   # [batch+1] cumulative sequence lengths
    max_seqlen,   # 최대 시퀀스 길이
):
    """
    여러 문서를 패킹한 경우 문서 경계에서 attention 차단
    cu_seqlens: [0, len_doc1, len_doc1+len_doc2, ...]
    """
    return flash_attn_varlen_qkvpacked_func(
        qkv,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        dropout_p=0.0,
        causal=True  # 인과적 마스킹
    )

# 장점:
# - 패딩 없이 여러 문서 처리
# - 문서 간 attention 차단 (cross-contamination 방지)
# - GPU 활용률 최대화
```

---

## Learning Rate Schedule

```python
import math

def get_cosine_lr(step, warmup_steps, max_steps, max_lr, min_lr=None):
    """Warmup + Cosine Decay (표준)"""
    if min_lr is None:
        min_lr = max_lr / 10

    if step < warmup_steps:
        return max_lr * (step / warmup_steps)
    if step > max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
    return lr

# 일반적 값:
# max_lr = 3e-4 (7B), 1e-4 (13B), 3e-5 (70B+)
# min_lr = max_lr / 10
# warmup_steps = 2,000 ~ 4,000 (전체의 1-2%)
```

### WSD (Warmup-Stable-Decay)

```
1. Warmup: 0 → max_lr (전체의 1~2%)
2. Stable: max_lr 유지 (전체의 70~80%)
3. Decay: 급격한 감소 (전체의 10~20%)

장점:
  Stable 구간 체크포인트에서 계속 학습 가능
  → 다른 task로 continual pretraining 분기
  → 다른 decay schedule 실험

MiniCPM 실험:
  Cosine보다 WSD의 최종 성능이 더 좋음
  특히 데이터 추가 학습 시나리오에서 유리
```

---

## 배치 크기 & 토큰

```python
# 배치 크기 계산
def compute_effective_batch(per_device, num_gpus, grad_accum, seq_len):
    """
    실제 학습 시 1 step당 처리 토큰 수
    """
    total_sequences = per_device * num_gpus * grad_accum
    total_tokens = total_sequences * seq_len
    return total_sequences, total_tokens

# 예시:
# per_device=4, num_gpus=512, grad_accum=1, seq_len=8192
# → sequences = 2,048 per step
# → tokens = 16,777,216 ≈ 16M tokens per step

# 현대 LLM 표준:
# LLaMA-3: 16M tokens/step
# GPT-3: 3.2M tokens/step (배치 크기 스케일링)
# 일반: 4M ~ 16M tokens/step

# 배치 크기 스케일링:
# 큰 배치 → 안정적 gradient, 학습 빠름 (더 많은 GPU 필요)
# 작은 배치 → 더 많은 gradient noise → 일반화 better? (논란)
```

---

## 안정적 학습 기법

### Mixed Precision

```python
from torch.cuda.amp import autocast
from contextlib import nullcontext

def training_step(model, batch, optimizer, use_bf16=True):
    # BF16 AMP (A100 이상 권장)
    ctx = autocast(dtype=torch.bfloat16) if use_bf16 else nullcontext()

    with ctx:
        outputs = model(**batch)
        loss = outputs.loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()
```

### Loss Spike 처리

```python
import copy

class StableLLMTrainer:
    def __init__(self, model, checkpoint_freq=100):
        self.model = model
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint = None
        self.prev_loss = float('inf')
        self.spike_threshold = 3.0  # 이전 loss의 3배 이상이면 spike

    def save_checkpoint(self):
        self.checkpoint = copy.deepcopy(self.model.state_dict())

    def train_step(self, batch, optimizer, step):
        # 주기적 체크포인트 저장
        if step % self.checkpoint_freq == 0:
            self.save_checkpoint()

        loss = self.compute_loss(batch)

        # Loss spike 탐지
        if loss > self.prev_loss * self.spike_threshold and step > 100:
            print(f"Loss spike at step {step}: {loss:.4f} vs {self.prev_loss:.4f}")
            # 방법 1: gradient skip
            optimizer.zero_grad()
            return self.prev_loss  # 이번 step 건너뜀

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )

        # Extreme gradient
        if grad_norm > 10.0:
            print(f"Large gradient norm: {grad_norm:.2f}, skipping step")
            optimizer.zero_grad()
            return loss.item()

        optimizer.step()
        optimizer.zero_grad()
        self.prev_loss = loss.item()
        return loss.item()

    def rollback(self):
        """이전 체크포인트로 복원"""
        if self.checkpoint:
            self.model.load_state_dict(self.checkpoint)
            print("Rolled back to previous checkpoint")
```

---

## 체크포인트 & 재현성

```python
import torch
import random
import numpy as np

def set_seed(seed: int = 42):
    """재현 가능한 학습을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_training_checkpoint(
    model, optimizer, scheduler, step, loss, path
):
    """완전한 학습 상태 저장"""
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'rng_state': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all(),
            'numpy': np.random.get_state(),
        }
    }, path)

def load_training_checkpoint(path, model, optimizer, scheduler=None):
    """체크포인트에서 학습 재개"""
    checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # RNG 상태 복원
    torch.set_rng_state(checkpoint['rng_state']['torch'])
    torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])

    return checkpoint['step'], checkpoint['loss']
```

---

## Compute-Optimal Scaling (Chinchilla)

```
Chinchilla (Hoffmann et al., 2022):
  "이 compute 예산으로 최적의 모델을 어떻게 학습할까?"

핵심 발견:
  C = compute budget (FLOP)
  N = model parameters
  D = training tokens

  최적 비율: D/N ≈ 20 (데이터 20배)
  → 모델 파라미터와 학습 토큰을 균형있게 증가

예시:
  GPT-3 (175B, 300B tokens): undertrained
  Chinchilla (70B, 1.4T tokens): 더 효율적
  → Chinchilla가 GPT-3보다 성능 좋음 (더 많은 데이터)

현대 LLM의 추세:
  LLaMA (7B, 1T tokens): D/N = 143 (Chinchilla 20 초과)
  LLaMA-3 (8B, 15T tokens): D/N = 1875 (매우 overtrained)
  → "Inference optimal": 추론 시 더 효율적 (inference 비용 절감)
  → 작은 모델을 더 많이 학습 → 배포 시 적은 메모리/계산

Inference optimal scaling:
  고정 inference 예산 → 어떤 모델 크기가 최적?
  결론: 더 작고 더 많이 학습된 모델이 종종 더 나음
```

---

## 연속 학습 (Continual Pretraining)

```
기존 학습된 모델에 새 도메인 데이터 추가 학습

도메인 적응 (Domain Adaptive Pretraining):
  일반 LLM → 의학 LLM: 의학 논문/교과서로 추가 학습
  일반 LLM → 코드 LLM: 코드 데이터로 추가 학습

도전: Catastrophic Forgetting
  새 도메인 학습 시 이전 능력 감소

  해결:
    1. 기존 데이터를 소량(10-20%) 혼합
    2. 낮은 학습률 사용 (기존 학습의 1/10)
    3. EWC (Elastic Weight Consolidation)
    4. WSD 스케줄 사용 (stable 체크포인트에서 분기)

중국어/수학 특화 학습 예시:
  기반: LLaMA-3-8B
  추가 데이터: 중국어 1T tokens + 수학 500B tokens
  기존 데이터 리플레이: 200B tokens
  학습률: 1e-4 (기존 학습의 1/3)
```

---

## Further Questions

**Q. ZeRO Stage 1/2/3의 차이는?**
> Stage 1: optimizer state만 GPU들에 분산. Stage 2: + gradient. Stage 3: + model parameter. 단계가 높을수록 메모리 절약 크지만 통신 비용 증가. Stage 3: 각 forward/backward마다 parameter를 all-gather하므로 통신 많음. 실용적 선택: 7B는 ZeRO-2 또는 FSDP, 70B+ 단일 노드 학습은 ZeRO-3.

**Q. Gradient Accumulation이 필요한 이유는?**
> GPU 메모리 한계로 큰 배치 불가 시, 여러 small batch 계산 후 gradient 누적. Effective batch size 증가 효과 (정확한 큰 배치와 동일). 단, 각 micro-batch 계산에서 dropout/normalization 동작이 다를 수 있음. BN(Batch Normalization) 사용하는 경우 주의 필요 (LLM은 LayerNorm 사용으로 문제 없음).

**Q. Chinchilla 이후 LLaMA-3는 왜 더 많이 학습하나?**
> Chinchilla: "주어진 compute로 학습"에 최적화. LLaMA-3: "주어진 inference 비용으로 최대 성능" 목표. Inference 시 모델 크기가 속도/메모리 결정. 8B 모델을 15T tokens 학습 = compute 많이 사용했지만, 배포 시 8B 크기의 추론 효율. Overtrained 작은 모델 > Undertrained 큰 모델 (inference 비용 동일 시).

**Q. 대규모 학습에서 실패를 복구하는 방법은?**
> 1) 정기적 체크포인트 저장 (매 1,000-5,000 step). 2) Loss spike 탐지: 이전 대비 3배+ → gradient skip. 3) 자동 rollback: 이전 안정 체크포인트로 복원. 4) 배치 블랙리스트: 문제 배치 식별 후 데이터에서 제거. 5) 체크포인트 중복 저장 (여러 곳에 분산 저장). LLaMA-3 (15T tokens 학습): 수 회 spike 복구 경험.
