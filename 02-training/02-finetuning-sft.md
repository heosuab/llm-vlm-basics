# Finetuning & SFT (Supervised Fine-Tuning)

## SFT란?

Pretrained LLM을 지시 따르기(instruction following)에 맞게 학습.
- 입력: (instruction, input) 쌍
- 출력: 원하는 응답
- Loss: Assistant 응답 부분에만 CLM loss 적용

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

def create_labels_with_masking(input_ids, response_start_pos):
    """
    User/System 파트 마스킹하고 Assistant 파트만 학습
    """
    labels = input_ids.clone()
    labels[:response_start_pos] = -100  # user/system 파트 무시
    return labels

class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        example = self.data[idx]
        messages = example["messages"]

        # Chat template 적용
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # System + User 부분 (응답 이전)
        partial_text = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )

        full_ids = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            return_tensors="pt"
        ).input_ids[0]

        prompt_len = len(self.tokenizer(partial_text).input_ids)

        labels = full_ids.clone()
        labels[:prompt_len] = -100  # prompt 마스킹

        return {"input_ids": full_ids, "labels": labels}

def sft_training_step(model, batch, optimizer):
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss  # -100인 위치 자동으로 무시됨

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()
```

---

## SFT 데이터

### 데이터 품질 > 데이터 양

```
LIMA (2023): 단 1,000개의 고품질 데이터로 강력한 SFT 가능
"Less is More for Alignment" (Zhou et al.)

핵심 발견:
  - 품질 높은 1K 샘플 > 품질 낮은 52K 샘플
  - 다양성: 하나의 답만 있는 것보다 여러 형태의 응답
  - 사실 정확성과 형식이 핵심

Alpaca-GPT4 (Peng et al., 2023):
  GPT-3.5로 생성한 52K → GPT-4로 재생성
  → GPT-3.5 버전 대비 큰 성능 향상
  → 수량보다 품질
```

### 주요 SFT 데이터셋

| 데이터셋 | 크기 | 특징 |
|---------|------|------|
| Alpaca | 52K | GPT-3.5 생성 (Self-Instruct) |
| ShareGPT | 90K | 실제 사용자 ChatGPT 대화 |
| OpenHermes-2.5 | 1M | 고품질 혼합 데이터 |
| Orca-2 | 500K | GPT-4 단계별 설명 포함 |
| UltraChat | 1.5M | 다양한 태스크 |
| WizardLM-Evol-Instruct | 200K | 복잡도 진화된 지시 |
| MagPie | 3M+ | 자동 생성 (LLaMA-3) |
| LIMA | 1K | 극도 고품질, 선별 |
| OpenAssistant (OASST) | 160K | 실제 인간 대화 |
| Dolly-15K | 15K | Databricks 직원 생성 |
| Tulu-3-SFT | 939K | 혼합 고품질 |

### 데이터 포맷

```json
// ChatML 형식 (OpenAI/Mistral 스타일)
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."},
    {"role": "user", "content": "Can you give an example?"},
    {"role": "assistant", "content": "Sure! For example..."}
  ]
}
```

---

## PEFT (Parameter-Efficient Fine-Tuning)

전체 파라미터 학습 없이 일부만 학습.

### LoRA (Low-Rank Adaptation) [Hu et al., 2022]

```
핵심 아이디어:
가중치 업데이트 행렬 ΔW를 저랭크 분해:
  ΔW = BA  (B: d×r, A: r×k, r << d, k)
  W' = W₀ + ΔW = W₀ + BA

초기화:
  A: Kaiming uniform (가우시안 노이즈)
  B: 0으로 초기화 (초기 ΔW = 0 보장)

추론 시: W₀ + BA를 merge → 추가 지연 없음
학습 파라미터: 2×r×d (r=8이면 ~6%)

스케일링: α/r (alpha/rank)
  실제 적용: ΔW = (α/r) × BA
  α 큰 값 → 업데이트 크게 반영
  보통 α = 2r (rank의 2배)

하이퍼파라미터:
  r (rank): 4 ~ 128 (보통 8~32)
  α (scaling): 16~64 (보통 r 또는 2r)
  target_modules: q, k, v, o, gate, up, down
  lora_dropout: 0.05~0.1
```

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    use_dora=False,  # DoRA 사용 여부
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable: 41,943,040 / 8,030,261,248 (0.52%)
```

### QLoRA (Quantized LoRA) [Dettmers et al., 2023]

```
NF4 (4-bit NormalFloat) + Double Quantization + LoRA 조합

NF4 양자화:
  가중치가 정규분포 N(0, σ²)를 따른다는 가정
  4비트에 최적화된 quantile 할당 (등비율이 아님)
  → 정규분포에서 최적 표현

Double Quantization:
  Quantization constants도 8비트로 추가 양자화
  메모리 추가 절약 (약 0.37 bits/param)

Paged Optimizer (CPU Offload):
  optimizer state를 CPU 메모리로 오프로드
  GPU OOM 방지

결과:
  65B 모델을 단일 48GB GPU에서 파인튜닝 가능
  7B 모델: 6-10GB GPU에서 학습 가능

품질 비교:
  QLoRA ≈ LoRA (약간 손실, 실용적으로 무시 가능)
  QLoRA < Full FT (더 큰 품질 차이)
  하지만 비용 대비 성능 뛰어남
```

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)

from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)
# gradient checkpointing 활성화됨
```

### DoRA (Decomposed LoRA) [Liu et al., 2024]

```
핵심: 가중치를 크기(magnitude)와 방향(direction)으로 분해

W = m · V/||V||

  m: 크기 (벡터), 학습 가능
  V/||V||: 방향 (정규화된 행렬)

LoRA를 방향 성분에 적용:
  V' = W₀ + ΔW = W₀ + BA
  W' = m · V'/||V'||

효과: LoRA보다 더 안정적이고 좋은 성능
   Full fine-tuning과 더 유사한 학습 동역학
```

### 기타 PEFT 방법들

```
Adapter Layers [Houlsby et al., 2019]:
  각 레이어 내 소형 bottleneck 삽입
  down-proj (d→r) + 비선형 + up-proj (r→d)
  r << d (r=64 → ~1% 파라미터)
  단점: 추론 지연 (직렬 추가)

Prefix Tuning [Li & Liang, 2021]:
  K, V에 학습 가능한 "virtual tokens" 추가
  모든 레이어에 동일한 수의 prefix
  단점: 실제 컨텍스트 길이 감소

Prompt Tuning [Lester et al., 2021]:
  입력에 soft continuous prompt 추가
  모델 파라미터 고정, prompt만 학습
  큰 모델에서는 Full FT와 유사한 성능

IA³ (Infused Adapter by Inhibiting and Amplifying):
  Attention 및 FFN에 element-wise 스케일 인자 추가
  파라미터 수 매우 적음 (~0.01%)
  빠른 적응에 유용

AdaLoRA [Zhang et al., 2023]:
  중요도에 따라 rank를 동적으로 배분
  중요한 가중치: 높은 rank
  덜 중요한 가중치: 낮은 rank (또는 0)
  전체 LoRA와 같은 파라미터로 더 나은 성능

LoftQ [Liu et al., 2023]:
  양자화된 모델에서 LoRA 초기화 최적화
  QLoRA의 품질 갭 감소
```

---

## 전체 파인튜닝 (Full Fine-Tuning)

```python
# DeepSpeed ZeRO-3 + BF16 표준 설정
# ds_config.json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "bf16": {"enabled": true},
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8
}
```

```python
# FSDP (PyTorch 표준)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# LLaMA 레이어 단위로 샤딩
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy(
        transformer_layer_cls={LlamaDecoderLayer}
    ),
    use_orig_params=True,  # LoRA와 호환
    cpu_offload=CPUOffload(offload_params=True)
)
```

---

## Instruction Tuning 기법

### Self-Instruct [Wang et al., 2023]

```python
# Alpaca 스타일의 자동 데이터 생성
def self_instruct_pipeline(seed_instructions, llm, n_new=1000):
    """
    소수 seed instruction에서 대규모 데이터 자동 생성
    """
    instructions = list(seed_instructions)

    while len(instructions) < n_new:
        # 랜덤하게 seed 선택
        selected = random.sample(instructions, min(3, len(instructions)))

        # LLM으로 새 instruction 생성
        prompt = f"""Given these examples:
{chr(10).join(selected)}
Generate a new, diverse instruction following similar style:"""

        new_instruction = llm.generate(prompt)

        # 품질 필터링
        if (len(new_instruction.split()) > 3 and  # 너무 짧지 않음
            rouge_overlap(new_instruction, instructions) < 0.7):  # 중복 아님
            instructions.append(new_instruction)

    return instructions
```

### Evol-Instruct [WizardLM]

```
기존 instruction을 더 복잡하게 진화:

진화 방법 (랜덤 선택):
  1. Add Constraints: 조건 추가
     "Write a story" → "Write a 500-word story with no adjectives"
  2. Deepening: 더 깊은 이해 요구
     "Explain X" → "Explain X and its implications for Y"
  3. Concretizing: 추상적 → 구체적
     "Improve this" → "Rewrite with 3 specific improvements"
  4. Increase Reasoning: 추론 단계 증가
     "Solve X" → "Solve X step-by-step with verification"
  5. Complicating Input: 입력 복잡화

여러 라운드 반복 → 점점 어려운 데이터 생성
WizardCoder, WizardMath에서 활용
```

### Magpie (Meta, 2024) - 자동 고품질 데이터 생성

```python
# LLaMA-3-Instruct에서 자기 지시 데이터 자동 생성

def generate_magpie_data(model, tokenizer, n_samples=100000):
    """
    모델이 스스로 질문을 생성하게 만드는 방법:
    instruction 시작 token까지만 주면 모델이 질문을 완성
    """
    # 사용자 turn 시작 토큰까지만 제공
    # <|start_header_id|>user<|end_header_id|>
    prompt_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"

    dataset = []
    for _ in range(n_samples):
        # 모델이 자연스럽게 질문 생성
        question_ids = model.generate(
            tokenizer(prompt_template, return_tensors="pt").input_ids,
            max_new_tokens=200,
            temperature=1.0,
        )
        question = tokenizer.decode(question_ids[0])

        # 전체 대화로 답변도 생성
        answer_ids = model.generate(
            question_ids,
            max_new_tokens=512,
        )
        answer = tokenizer.decode(answer_ids[0])

        dataset.append({
            "instruction": extract_question(question),
            "response": extract_answer(answer)
        })

    return dataset

# 결과: 3M+ 고품질 SFT 데이터 자동 생성
# 인간 annotation 없이 모델 수준의 데이터 품질
```

### Rejection Sampling Fine-Tuning (RSF)

```
자기 개선 루프:
  1. 현재 모델로 각 문제에 K개 응답 생성
  2. Reward model (또는 검증 가능한 신호)로 최고 선택
  3. 선택된 (문제, 최고응답) 쌍으로 SFT 추가 학습
  4. 더 좋은 모델 → 다시 반복

LLaMA-2에서 활용:
  - RM으로 응답 필터링 → 고품질 SFT 데이터
  - PPO 이전 단계로 활용

DeepSeek-R1에서 활용:
  - RL 학습 모델로 N=32 샘플링
  - 정답 확인 후 올바른 응답만 선택
  - 선별된 응답으로 추가 SFT (Step 3)
```

---

## 학습 프레임워크

### TRL (Transformer Reinforcement Learning)

```python
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# LoRA 설정
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM"
)

# 학습 설정
training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # effective batch = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    save_steps=200,
    max_seq_length=2048,
    packing=True,           # Sequence packing (GPU 활용률 ↑)
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
)
trainer.train()
```

---

## Chat Template & System Prompt

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct"
)

# 대화 형식
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What is it known for?"}
]

# 학습용 (assistant 응답 포함)
train_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False  # 학습 시
)

# 추론용 (다음 응답 생성을 위한 프롬프트)
infer_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  # 추론 시 → "<|start_header_id|>assistant<|end_header_id|>"
)
```

---

## SFT 품질 평가

```python
# SFT 후 모델 품질 체크
quality_checks = {
    "instruction_following": "답변이 지시를 따르는가?",
    "factual_accuracy": "사실이 정확한가?",
    "format_compliance": "요청한 형식으로 답변했는가?",
    "length_appropriateness": "답변 길이가 적절한가?",
    "no_hallucination": "없는 내용을 꾸며내지 않았는가?",
}

def evaluate_sft_quality(model, eval_dataset, judge_model):
    results = {}
    for criterion, description in quality_checks.items():
        scores = []
        for example in eval_dataset:
            response = model.generate(example["instruction"])
            score = judge_model.score(
                instruction=example["instruction"],
                response=response,
                criterion=description
            )
            scores.append(score)
        results[criterion] = sum(scores) / len(scores)
    return results
```

---

## 학습 팁 & 트러블슈팅

```
Overfitting 징후:
  - Train loss ↓ 지속, Eval loss ↑
  - 특정 형식/표현 반복 (format overfitting)
  해결: 데이터 다양성 증가, dropout, 에폭 수 감소

Underfitting 징후:
  - Loss가 충분히 내려가지 않음
  - 지시를 제대로 따르지 못함
  해결: lr ↑, 더 긴 학습, 더 많은 데이터

Format Overfitting:
  특정 응답 형식에 과도하게 학습
  예: 항상 "As an AI assistant..." 시작
  해결: 다양한 형식의 응답 데이터 포함

Sycophancy (아첨):
  사용자 의견에 무조건 동의하는 경향
  원인: SFT 데이터가 긍정적 응답에 편향
  해결: 적절히 거부하는 데이터 포함, RLHF

학습률 선택:
  LoRA: 1e-4 ~ 3e-4 (높아도 ok, LoRA만 학습)
  Full FT: 1e-5 ~ 5e-5 (낮게 유지)
  QLoRA: 2e-4 (BNB 4bit에서 높아도 안정)

배치 크기:
  Effective batch = per_device × num_gpu × grad_accum
  보통 64~256 권장 (안정적 학습)
```

---

## Further Questions

**Q. LoRA의 rank r을 크게/작게 하면?**
> r 크게 → 표현력 증가, 파라미터 증가, 학습 시간 증가, Full FT에 수렴. r 작게 → 경량, 빠름, 간단한 태스크에 적합. 일반적으로 r=8~16으로 시작, 성능 부족 시 32~64. 도메인 특화 파인튜닝은 r=8로도 충분.

**Q. QLoRA vs LoRA 선택 기준은?**
> GPU 메모리가 제한적이면 QLoRA. 최고 품질이 필요하면 LoRA 또는 Full FT. 70B 모델을 단일 GPU에서 파인튜닝 → QLoRA 필수. 7B 모델에 GPU 여러 개 → LoRA 또는 Full FT 권장. QLoRA: 품질 차이 5% 미만, 메모리 절약 4×.

**Q. SFT 후 모델이 거짓말을 더 잘하는 이유는?**
> SFT는 주어진 예시를 따르도록 학습 → 사용자 기대에 맞는 답을 생성하려는 경향 (sycophancy). 정확성보다 형식이 우선될 수 있음. RLHF/DPO로 추가 교정 필요. LIMA 논문: 품질 높은 소량 데이터로 이 문제 완화 가능.

**Q. Multi-turn SFT 데이터의 학습 마스킹은 어떻게 하나?**
> 방법 1: Assistant 응답만 학습 (User turn -100). 방법 2: 마지막 turn만 학습 (이전 대화 -100). 방법 3: 전체 학습 (User turn도 포함). 방법 1이 일반적. 방법 3은 모델이 User 역할도 모방 → 원치 않는 동작. 구현: response_start_pos로 마스킹 경계 설정.
