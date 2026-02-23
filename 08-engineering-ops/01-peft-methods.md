# PEFT 심화 & 실용 가이드

## PEFT 방법 전체 정리

### Adapter 계열

```
Adapter Layers [Houlsby et al., 2019]:
  각 Transformer 레이어 내 소형 bottleneck 삽입

  구조:
    LayerNorm
      ↓
    Transformer Sub-layer (Attention / FFN)
      ↓
    Adapter:
      Linear(d, r)  ← down projection (d→r)
      Activation (GELU / ReLU)
      Linear(r, d)  ← up projection (r→d)
      + residual
    LayerNorm

  파라미터 수: 2rd (r=64, d=4096 → ~0.5M per adapter)
  레이어당 2개 (attention 후 + FFN 후)

  단점: 직렬 구조 → 추론 지연 발생

Parallel Adapter [He et al., 2022]:
  Sequential 대신 병렬로 Adapter 연결
  Adapter(x) + SubLayer(x) 형태
  → 추론 시 병렬 계산으로 지연 감소
```

### LoRA 계열

```python
# LoRA 수학적 이해
# 가중치 업데이트 ΔW를 저랭크 분해

# 기존 파인튜닝:
# W_fine = W_pretrained + ΔW    (ΔW: d×k)

# LoRA:
# W_fine = W_pretrained + BA    (B: d×r, A: r×k, r << min(d,k))

# 저랭크 가정:
# ΔW가 저랭크임을 가정 (언어 모델에서 task-specific 변화는 낮은 intrinsic rank)
# 실험적 검증: rank 4-16으로도 충분

import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """LoRA를 직접 구현한 Linear 레이어"""
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # LoRA scaling factor

        # 원래 가중치 (frozen)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )

        # LoRA 가중치 (학습 가능)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)

        # 초기화
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # B를 0으로 초기화 → 초기 ΔW = 0

    def forward(self, x):
        base_output = F.linear(x, self.weight)
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_output + self.scaling * lora_output

    def merge_weights(self):
        """추론 최적화: W + BA를 합쳐서 원래 선형 레이어로 만들기"""
        self.weight.data += self.scaling * (self.lora_B @ self.lora_A)
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()
```

---

## PEFT 라이브러리 활용

### LoRA with PEFT

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# LoRA 설정
config = LoraConfig(
    r=16,                          # rank (4-128, 기본 8-16)
    lora_alpha=32,                  # scaling = alpha/r = 2
    target_modules=[               # 적용할 레이어
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",                   # "none", "all", "lora_only"
    task_type=TaskType.CAUSAL_LM,
    use_dora=False,                # DoRA 사용 여부
    use_rslora=False,              # rsLoRA (rank stabilized LoRA)
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 8,030,261,248 || trainable%: 0.5224
```

### QLoRA 설정

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
import torch

# 4비트 양자화
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4 양자화 타입
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # 이중 양자화
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)

# 4비트 학습 준비
model = prepare_model_for_kbit_training(model)
# - Embedding, Norm 레이어를 FP32로
# - Gradient checkpointing 활성화

# LoRA 적용
model = get_peft_model(model, LoraConfig(...))
```

---

## 최신 PEFT 기법

### DoRA (Decomposed LoRA) [Liu et al., 2024]

```python
# 가중치를 크기(magnitude)와 방향(direction)으로 분해
# W = m · V / ||V||_c
# m: 크기 벡터, V: 방향 행렬

class DoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        # 크기 (magnitude) - 학습 가능
        self.magnitude = nn.Parameter(
            self.weight.norm(p=2, dim=1).view(-1, 1)
        )
        # LoRA 성분 (방향 업데이트)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)

    def forward(self, x):
        # 방향: (W + ΔW) / ||W + ΔW||
        adapted = self.weight + self.scaling * self.lora_B @ self.lora_A
        col_norms = adapted.norm(p=2, dim=1, keepdim=True)
        direction = adapted / col_norms.clamp(min=1e-6)

        # 크기 × 방향
        normalized_weight = self.magnitude * direction
        return F.linear(x, normalized_weight)

# 효과: LoRA보다 Full FT의 학습 동역학에 더 가까움
# 성능: LoRA 대비 1-3% 향상
```

### rsLoRA (Rank Stabilized LoRA) [Kalajdzievski, 2023]

```
기존 LoRA scaling: α/r
문제: rank 증가 시 gradient magnitude가 달라짐

rsLoRA scaling: α/√r
  → rank에 관계없이 gradient 크기 안정
  → 높은 rank에서 학습 안정성 향상

PEFT 라이브러리: use_rslora=True
```

### LoRA+ [Hayou et al., 2024]

```
A와 B의 학습률을 다르게 설정:
  lr_A: 기본 lr
  lr_B: 기본 lr × λ (λ = 2 ~ 16 권장)

이유: 초기에 B=0이므로 A를 더 빠르게 학습시켜야
     실질적인 업데이트가 발생하도록 함

효과: 동일 파라미터로 LoRA 대비 2× 학습 효율
```

### GaLore (Gradient Low-Rank Projection) [Zhao et al., 2024]

```
Full fine-tuning이지만 gradient를 저랭크로 투영:

  gradient G ∈ ℝ^{m×n}
  SVD: G ≈ U_r Σ_r V_r^T  (상위 r 성분만)
  optimizer state를 저랭크 공간에서 유지

특징:
  - 실제 Full FT (LoRA와 달리 모든 파라미터 업데이트)
  - Optimizer state의 메모리 절약 (rank r 유지)
  - 배치마다 투영 업데이트 (T step마다 SVD 재계산)

memory 절약:
  Adam: 2P (모멘트)
  GaLore: 2P × r/min(m,n) (저랭크 공간에서만)

단점:
  SVD 계산 비용
  LoRA보다 구현 복잡
```

### LISA (Layerwise Importance Sampled AdamW) [Pan et al., 2024]

```
무작위로 레이어를 선택하여 학습:
  Full FT처럼 행동하지만 메모리 LoRA 수준

알고리즘:
  매 K step마다:
    2-4개 레이어 무작위 선택
    선택된 레이어만 gradient 계산
    나머지 레이어 frozen

효과:
  메모리: LoRA 수준 (~2× base model)
  성능: LoRA보다 좋은 경우 많음
  직관: 랜덤 레이어 선택이 오히려 정규화 효과

구현:
  from lisa import LISATrainer
```

---

## Prefix Tuning / P-Tuning

```python
from peft import PrefixTuningConfig, PromptTuningConfig

# Prefix Tuning
prefix_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,    # 가상 prefix 토큰 수
    prefix_projection=True,   # MLP로 prefix 생성 (안정적)
    encoder_hidden_size=768,  # MLP hidden 크기
)

# → 각 레이어의 K, V에 학습 가능한 prefix 추가
# → 원래 토큰 sequence 앞에 붙음

# P-Tuning v2 [Liu et al., 2022]:
# 모든 레이어에 prefix 적용 (v1은 입력에만)
# Classification, NER에 효과적

# Prompt Tuning [Lester et al., 2021]
prompt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=8,
    prompt_tuning_init="TEXT",           # 텍스트로 초기화
    prompt_tuning_init_text="Classify the text: ",
)

# → 입력에 soft token만 추가
# → 매우 경량 (virtual token 수 × d)
# → 큰 모델(10B+)에서 Full FT에 필적
```

---

## PEFT 방법 비교

| 방법 | 학습 파라미터 | 추론 지연 | 성능 | 구현 난이도 |
|------|------------|---------|------|----------|
| Full FT | 100% | 없음 | 최고 | 쉬움 |
| LoRA (r=16) | ~0.5% | 없음 (merge 후) | 높음 | 쉬움 |
| QLoRA | ~0.5% | 없음 | 중-높음 | 쉬움 |
| DoRA | ~0.5% | 없음 | LoRA+ | 중간 |
| Adapter | ~1-3% | 있음 (직렬) | 중간 | 중간 |
| Prefix Tuning | ~0.1% | 있음 (토큰 추가) | 중간 | 중간 |
| Prompt Tuning | ~0.01% | 있음 (토큰 추가) | 낮-중간 | 쉬움 |
| GaLore | 100% | 없음 | Full FT 수준 | 어려움 |
| LISA | 동적 | 없음 | Full FT 수준 | 중간 |

---

## 학습 프레임워크

### TRL (Transformer Reinforcement Learning)

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    save_steps=200,
    max_seq_length=2048,
    packing=True,
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
)
trainer.train()

# DPO
from trl import DPOTrainer, DPOConfig
dpo_config = DPOConfig(
    beta=0.1,              # KL penalty
    loss_type="sigmoid",   # "sigmoid", "ipo", "hinge" 등
    learning_rate=5e-7,
    bf16=True,
)
dpo_trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=preference_dataset,
    processing_class=tokenizer,
)

# GRPO (Reasoning RL)
from trl import GRPOTrainer, GRPOConfig
grpo_config = GRPOConfig(
    num_generations=8,     # group size G
    beta=0.001,            # KL penalty
    learning_rate=1e-6,
)
```

### LLaMA-Factory

```yaml
# 설정 파일 (yaml)
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
stage: sft               # sft, dpo, grpo, ppo, kto
do_train: true

dataset: alpaca_gpt4_en
template: llama3
finetuning_type: lora    # full, lora, qlora, freeze

lora_rank: 16
lora_alpha: 32
lora_target: all         # q, k, v, o, gate, up, down

per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
num_train_epochs: 3
bf16: true
flash_attn: fa2
```

```bash
# 학습 실행
llamafactory-cli train config/llama3_lora_sft.yaml

# 추론
llamafactory-cli chat \
  --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
  --adapter_name_or_path saves/llama3-8b/lora/sft \
  --template llama3 \
  --finetuning_type lora
```

### Axolotl

```yaml
# axolotl config.yaml
base_model: meta-llama/Meta-Llama-3-8B-Instruct
model_type: LlamaForCausalLM

load_in_4bit: true        # QLoRA
adapter: lora
lora_r: 16
lora_alpha: 32
lora_target_modules:
  - q_proj
  - v_proj

datasets:
  - path: tatsu-lab/alpaca
    type: alpaca

sequence_len: 2048
max_packed_sequence_len: 2048
sample_packing: true

num_epochs: 3
learning_rate: 2e-4
```

---

## 체크포인트 관리

```python
# LoRA 가중치 저장/로드
model.save_pretrained("lora_weights/")
tokenizer.save_pretrained("lora_weights/")

# 추론 시 로드
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "lora_weights/")

# 방법 1: 동적 (LoRA 유지)
# 장점: 여러 LoRA 어댑터를 base 모델에 적용/제거 가능
# 단점: 추론 시 약간 느림

# 방법 2: 가중치 병합 (추론 최적화)
merged = model.merge_and_unload()
merged.save_pretrained("merged_model/")
# → 원래 Linear 레이어와 동일 속도

# 여러 LoRA 병합 (Model Merging)
from peft import PeftModel

# 다른 태스크의 LoRA들 병합
base = AutoModelForCausalLM.from_pretrained("base_model")
model = PeftModel.from_pretrained(base, "lora_task1")
model.load_adapter("lora_task2", adapter_name="task2")

# PEFT의 LoRA 합산 (단순 합산)
merged = model.merge_adapter()
```

---

## LoRA 어댑터 허브 활용

```python
# HuggingFace Hub에서 LoRA 어댑터 다운로드/업로드
from peft import PeftModel

# 커뮤니티 LoRA 적용
base = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = PeftModel.from_pretrained(
    base,
    "username/llama3-8b-lora-coding"  # Hub에서 자동 다운로드
)

# 업로드
model.push_to_hub("username/my-lora-adapter")
tokenizer.push_to_hub("username/my-lora-adapter")
```

---

## 실전 팁

```
LoRA Rank 선택:
  간단한 스타일 전환: r=4~8
  일반 instruction tuning: r=8~16 (기본값)
  복잡한 도메인 적응: r=32~64
  최대 표현력 (Full FT에 가깝게): r=128+

target_modules 선택:
  최소 (빠른 실험): q_proj, v_proj
  표준 (균형): q, k, v, o 프로젝션
  풀 (최고 성능): 모든 linear (q,k,v,o,gate,up,down)

학습률:
  LoRA: 1e-4 ~ 3e-4
  QLoRA: 2e-4 (NF4에서 안정)
  Full FT: 1e-5 ~ 5e-5
  DoRA: LoRA와 동일

배치 크기:
  effective batch = per_device × num_gpu × grad_accum
  권장: 32~256 sequences

과적합 방지:
  에폭: 1-3 (데이터 많으면 1 에폭도 충분)
  Dropout: 0.05~0.1 (작은 데이터셋에서 중요)
  Weight decay: 0.01 (LoRA에서 선택적)

성능 기준:
  LoRA(r=16, full modules): Full FT의 95%+ 달성 가능
  QLoRA(r=16): Full FT의 90-95%
  Adapter: Full FT의 85-95%
```

---

## Further Questions

**Q. LoRA의 rank r을 크게/작게 하면?**
> r 크게 (64+): 표현력 증가, Full FT에 수렴, 파라미터 증가, 학습 시간 증가. r 작게 (4-8): 경량, 빠름, 규제 효과, 간단한 태스크에 적합. 실용적으로 r=8~16으로 시작 후 성능 부족 시 증가. rsLoRA 사용 시 높은 rank에서도 안정적.

**Q. LoRA와 Full FT의 차이는?**
> Full FT: 전체 파라미터 업데이트, 최고 성능, 메모리 많이 필요. LoRA: 저랭크 업데이트, 90-98% 성능 유지, 메모리 절약. 차이: Full FT는 rank가 무한대인 LoRA와 같음. 실용적으로 대부분의 태스크에서 LoRA로 충분. 미묘한 스타일 변화, 완전한 도메인 전환에는 Full FT 필요.

**Q. 언제 각 PEFT 방법을 선택하나?**
> 제한된 GPU (24GB 이하): QLoRA. 충분한 GPU (40GB+): LoRA 또는 Full FT. 빠른 실험: Prompt Tuning (파라미터 극소). 최고 성능 필요: DoRA 또는 GaLore. 여러 task 동시 지원: 각 task별 LoRA 어댑터 (base 모델 공유). Inference 속도 중요: merge_and_unload()로 병합.

**Q. GaLore vs LoRA 언제 선택하나?**
> LoRA: 특정 모듈에만 적용, 구현 쉬움, 추론 시 merge 가능. GaLore: 실제 Full FT (모든 파라미터 업데이트), LoRA의 rank 한계 없음, optimizer state만 절약. 성능: GaLore ≥ LoRA (전체 파라미터 업데이트로). 단점: SVD 계산 오버헤드, 구현 복잡. 높은 품질이 필요하고 LoRA의 rank 한계를 느낄 때 GaLore 고려.
