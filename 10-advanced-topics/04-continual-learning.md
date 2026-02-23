# Continual Learning & 지속 학습

## 핵심 문제: Catastrophic Forgetting

```
새로운 태스크 학습 → 이전 태스크 성능 급락

예시:
  Step 1: 의료 도메인 파인튜닝 → 의료 성능 90% ↑
  Step 2: 법률 도메인 파인튜닝 → 의료 성능 → 45%로 급락!

원인: Stability-Plasticity Dilemma
  SGD: 새 loss 최소화 → 기존 task 최적 weight 덮어씀
  Weight interference: 같은 파라미터가 두 태스크 모두 담당
  → 새 태스크 최적화 = 기존 태스크 최적 지점에서 이탈

측정:
  Backward Transfer: T_new(T_old) - T_baseline(T_old)
    → 음수 = 망각
  Forward Transfer: T_new(T_upcoming) - T_baseline
    → 양수 = 사전 학습 효과
```

---

## 연속 학습 시나리오

```
Task-incremental:
  태스크 ID 알려줌 → 태스크별 모델 헤드 선택 가능
  가장 쉬움 (태스크 ID = 힌트)
  실용적 의미 적음

Class-incremental:
  새 클래스 추가, 태스크 ID 미제공
  → 기존 + 새 클래스 모두 분류
  가장 어려운 시나리오
  예: Continual Image Classification

Domain-incremental:
  같은 태스크, 다른 도메인 데이터
  LLM에서 가장 흔한 시나리오
  예: 의료 도메인 → 법률 도메인 → 금융 도메인

Data-incremental (Temporal):
  시간에 따라 새 데이터 추가
  → LLM 지식 업데이트 (2023년 → 2024년 → 2025년)
```

---

## 방법론 분류

### 1. Regularization 기반

#### EWC (Elastic Weight Consolidation)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EWCModel(nn.Module):
    """Elastic Weight Consolidation 구현"""

    def __init__(self, base_model: nn.Module, ewc_lambda: float = 100.0):
        super().__init__()
        self.model = base_model
        self.ewc_lambda = ewc_lambda
        self.fisher_matrices = {}   # 태스크별 Fisher 행렬
        self.optimal_params = {}    # 태스크별 최적 파라미터

    def compute_fisher(self, dataloader, task_id: int):
        """Fisher Information Matrix 계산 (대각 근사)"""
        fisher = {name: torch.zeros_like(p)
                  for name, p in self.model.named_parameters()}

        self.model.eval()
        for batch in dataloader:
            self.model.zero_grad()
            inputs, labels = batch
            output = self.model(inputs)
            loss = F.cross_entropy(output, labels)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # Fisher = E[(grad log p)²] ≈ (grad)²
                    fisher[name] += param.grad.pow(2) / len(dataloader)

        self.fisher_matrices[task_id] = fisher
        # 현재 최적 파라미터 저장
        self.optimal_params[task_id] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

    def ewc_loss(self) -> torch.Tensor:
        """EWC Regularization Term"""
        ewc_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for task_id in self.fisher_matrices:
            for name, param in self.model.named_parameters():
                fisher_i = self.fisher_matrices[task_id][name]
                optimal_i = self.optimal_params[task_id][name]

                # L_EWC = Σ F_i × (θ_i - θ*_i)²
                ewc_loss += (fisher_i * (param - optimal_i).pow(2)).sum()

        return self.ewc_lambda / 2 * ewc_loss

    def train_step(self, batch, optimizer) -> float:
        """EWC를 포함한 학습 스텝"""
        inputs, labels = batch
        output = self.model(inputs)

        # Task loss + EWC regularization
        task_loss = F.cross_entropy(output, labels)
        ewc_reg = self.ewc_loss()
        total_loss = task_loss + ewc_reg

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss.item()
```

```
EWC 한계 (LLM에서):
  Fisher 행렬 계산: O(p²) → 수십억 파라미터에서 비실용적
  대각 근사 사용하지만 여전히 큰 메모리
  태스크 수 많아질수록 규제항 복잡
  → LLM에서는 간소화 버전만 실용적
```

#### Online EWC / SI (Synaptic Intelligence)

```python
class SI_Model:
    """Synaptic Intelligence - 온라인 EWC 변형"""

    def __init__(self, model: nn.Module, c: float = 0.1, xi: float = 0.1):
        self.model = model
        self.c = c    # SI 강도
        self.xi = xi  # 분모 안정화

        # 파라미터별 중요도 추적
        self.W = {name: torch.zeros_like(p)
                  for name, p in model.named_parameters()}
        self.prev_params = {name: p.data.clone()
                           for name, p in model.named_parameters()}
        self.Omega = {}  # 누적 중요도

    def update_W(self, loss: torch.Tensor, lr: float):
        """학습 중 실시간 중요도 업데이트"""
        loss.backward(retain_graph=True)

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # W += -grad × delta_theta (파라미터 이동에 따른 loss 감소)
                delta = param.data - self.prev_params[name]
                self.W[name] += -param.grad * delta

            self.prev_params[name] = param.data.clone()

    def consolidate(self):
        """태스크 완료 후 중요도 업데이트"""
        for name, param in self.model.named_parameters():
            delta = param.data - self.prev_params.get(name, param.data)

            if name not in self.Omega:
                self.Omega[name] = torch.zeros_like(param)

            # Ω_i = Σ W_i / (Δθ_i² + ξ)
            self.Omega[name] += self.W[name].clamp(min=0) / (delta.pow(2) + self.xi)
            self.W[name] = torch.zeros_like(param)
            self.prev_params[name] = param.data.clone()
```

---

### 2. Replay 기반

#### Experience Replay

```python
import random
from collections import deque

class ReplayBuffer:
    """이전 태스크 데이터 버퍼"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, item):
        self.buffer.append(item)

    def sample(self, n: int) -> list:
        return random.sample(list(self.buffer), min(n, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class ContinualTrainer:
    """Replay 기반 연속 학습"""

    def __init__(self, model: nn.Module, buffer_size: int = 1000):
        self.model = model
        self.replay_buffer = ReplayBuffer(buffer_size)

    def train_task(self, task_data: list, replay_ratio: float = 0.3):
        """새 태스크 학습 + 이전 태스크 replay"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        for epoch in range(10):
            for batch in task_data:
                # 새 태스크 데이터
                new_loss = self._compute_loss(batch)

                # Replay (이전 태스크 데이터 혼합)
                replay_size = int(len(batch[0]) * replay_ratio)
                if replay_size > 0 and len(self.replay_buffer) > 0:
                    replay_samples = self.replay_buffer.sample(replay_size)
                    replay_batch = self._collate(replay_samples)
                    replay_loss = self._compute_loss(replay_batch)
                    total_loss = new_loss + replay_loss
                else:
                    total_loss = new_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # 현재 태스크 데이터 버퍼에 추가 (Reservoir Sampling)
            for item in task_data:
                self.replay_buffer.add(item)
```

#### GDumb (Greedy Sampler and Dumb Learner)

```python
class GDumb:
    """
    GDumb: Balanced Replay Buffer + 처음부터 재학습
    매우 단순하지만 효과적
    """

    def __init__(self, buffer_size: int, model_factory):
        self.buffer_size = buffer_size
        self.class_buffer: dict[int, list] = {}  # 클래스별 balanced buffer
        self.model_factory = model_factory

    def add_data(self, samples: list, labels: list):
        """Greedy 균형 샘플링으로 버퍼 추가"""
        for sample, label in zip(samples, labels):
            if label not in self.class_buffer:
                self.class_buffer[label] = []

            num_classes = len(self.class_buffer)
            per_class = self.buffer_size // num_classes

            if len(self.class_buffer[label]) < per_class:
                self.class_buffer[label].append(sample)
            else:
                # 임의 교체 (다른 클래스 버퍼가 너무 크면 줄임)
                for c in self.class_buffer:
                    while len(self.class_buffer[c]) > per_class:
                        self.class_buffer[c].pop()

    def learn(self, epochs: int = 100):
        """버퍼 데이터로 처음부터 재학습"""
        model = self.model_factory()  # 새 모델 (망각 걱정 없음)
        optimizer = torch.optim.Adam(model.parameters())

        all_data = [(s, l) for l, samples in self.class_buffer.items()
                    for s in samples]

        for epoch in range(epochs):
            random.shuffle(all_data)
            for sample, label in all_data:
                # ... 표준 학습
                pass
        return model
```

#### Pseudo Replay (생성 모델 기반)

```python
class GenerativeReplay:
    """이전 모델로 synthetic 데이터 생성하여 replay"""

    def __init__(self, model: nn.Module):
        self.current_model = model
        self.previous_model = None

    def start_new_task(self):
        """새 태스크 시작 시 현재 모델 저장"""
        self.previous_model = type(self.current_model)()
        self.previous_model.load_state_dict(self.current_model.state_dict())
        self.previous_model.eval()
        for p in self.previous_model.parameters():
            p.requires_grad_(False)

    def generate_replay_data(self, num_samples: int) -> list:
        """이전 모델로 synthetic 이전 태스크 데이터 생성"""
        if self.previous_model is None:
            return []

        with torch.no_grad():
            # LLM의 경우: 이전 태스크 도메인 텍스트 생성
            prompts = self._sample_prompts(num_samples)
            generated = [self.previous_model.generate(p) for p in prompts]

        return generated

    def train_with_replay(self, task_data, replay_count: int = 100):
        """새 태스크 + 생성된 replay 데이터로 학습"""
        replay_data = self.generate_replay_data(replay_count)
        combined_data = list(task_data) + replay_data
        random.shuffle(combined_data)
        # ... 학습
```

---

### 3. Architecture 기반

#### LoRA 기반 Continual Learning

```python
from peft import LoraConfig, get_peft_model

class LoRATaskSwitch:
    """태스크별 LoRA adapter, 필요 시 전환"""

    def __init__(self, base_model, lora_rank: int = 16):
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.task_adapters: dict[str, dict] = {}

    def train_task(self, task_name: str, task_data, epochs: int = 3):
        """새 태스크를 위한 LoRA 학습"""
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank * 2,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(self.base_model, lora_config)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

        for epoch in range(epochs):
            for batch in task_data:
                outputs = model(**batch)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # LoRA 가중치 저장
        self.task_adapters[task_name] = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if "lora_" in name
        }

        return model

    def switch_task(self, task_name: str):
        """태스크 전환 (adapter swap)"""
        if task_name not in self.task_adapters:
            raise ValueError(f"Task {task_name} not trained")

        # Base 모델 → 해당 태스크 adapter 로드
        saved_adapter = self.task_adapters[task_name]
        # ... adapter 로드 로직

    def merge_adapters(self, task_names: list[str]) -> dict:
        """여러 태스크 adapter를 평균으로 병합 (Task Arithmetic)"""
        merged = {}
        for name in self.task_adapters[task_names[0]]:
            merged[name] = torch.stack([
                self.task_adapters[t][name] for t in task_names
            ]).mean(dim=0)
        return merged
```

#### O-LoRA (Orthogonal LoRA)

```python
def orthogonal_lora_update(model, lora_A: torch.Tensor, lora_B: torch.Tensor,
                           prev_subspace: torch.Tensor = None) -> tuple:
    """
    O-LoRA: 이전 태스크 subspace에 직교하는 방향으로만 업데이트
    망각 방지 + 새 태스크 학습 가능
    """
    if prev_subspace is None:
        return lora_A, lora_B

    # 이전 태스크 subspace에 직교 투영
    # projection = A - V(VᵀA) where V = prev_subspace
    def project_away(matrix: torch.Tensor, subspace: torch.Tensor) -> torch.Tensor:
        # 이전 subspace 방향 제거
        projection = subspace @ (subspace.T @ matrix)
        return matrix - projection

    new_lora_A = project_away(lora_A, prev_subspace)

    # 직교 방향에서만 업데이트 → 이전 태스크 보존
    return new_lora_A, lora_B
```

---

## Continual Pretraining (도메인 적응)

### 목적 및 전략

```python
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

def continual_pretrain(
    base_model_id: str,
    domain_data_path: str,
    general_data_path: str,
    output_dir: str,
    domain_ratio: float = 0.9,
    learning_rate: float = 2e-5,
    num_epochs: int = 1
):
    """도메인 특화 Continual Pretraining"""

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # 데이터셋 구성
    domain_dataset = load_dataset("json", data_files=domain_data_path)["train"]
    general_dataset = load_dataset("json", data_files=general_data_path)["train"]

    # 도메인:일반 = domain_ratio:(1-domain_ratio) 혼합
    weights = (
        [domain_ratio / len(domain_dataset)] * len(domain_dataset) +
        [(1 - domain_ratio) / len(general_dataset)] * len(general_dataset)
    )
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(domain_dataset) * 2,  # 총 샘플 수
        replacement=True
    )
    combined = ConcatDataset([domain_dataset, general_dataset])

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,  # 원래의 1/10 ~ 1/100
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,  # 짧은 warmup
        save_steps=1000,
        logging_steps=100,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined,
        # custom sampler: 도메인/일반 비율 조정
    )
    trainer.train()
    return model
```

### Domain-Adaptive Pretraining (DAPT) 예시

```python
# 의료 도메인 적응 (PubMed + 일반 텍스트 혼합)
domain_configs = {
    "medical": {
        "data": ["pubmed_abstracts", "medical_textbooks"],
        "domain_ratio": 0.85,
        "lr": 1e-5,
        "epochs": 1
    },
    "code": {
        "data": ["github_code", "stackoverflow"],
        "domain_ratio": 0.9,
        "lr": 2e-5,
        "epochs": 2
    },
    "legal": {
        "data": ["court_opinions", "legal_texts"],
        "domain_ratio": 0.8,
        "lr": 5e-6,
        "epochs": 1
    }
}

# 순서: Base LLM → Continual Pretraining → SFT → RLHF
# 이 순서가 각 단계의 효과 극대화
```

---

## 지식 편집 (Knowledge Editing)

### ROME (Rank-One Model Editing)

```python
def rome_edit(model, subject: str, relation: str, target_new: str, target_old: str):
    """
    ROME: 특정 사실을 (subject, relation, target_new)로 업데이트
    예: ("에펠탑", "위치한 도시", "파리") → ("베를린")

    원리: MLP가 사실 저장소 역할 → 특정 레이어의 W만 수정
    방법: Rank-1 업데이트로 정확한 수정 (다른 지식 최소 영향)
    """

    # 1. 타겟 레이어 찾기 (해당 사실이 저장된 레이어)
    layer_idx = find_fact_layer(model, subject, relation)

    # 2. 키 벡터 계산 (subject의 표현)
    k = compute_key_vector(model, subject, layer_idx)

    # 3. 값 벡터 계산 (새 타겟 정보가 포함된 표현)
    v_new = compute_value_vector(model, subject, relation, target_new, layer_idx)
    v_old = compute_value_vector(model, subject, relation, target_old, layer_idx)

    # 4. Rank-1 업데이트
    W = model.layers[layer_idx].mlp.W_out  # MLP 출력 행렬

    # C = covariance matrix (보정 행렬)
    C_inv = compute_covariance_inverse(model, layer_idx)

    # 업데이트: ΔW = (v_new - v_old - W_old × k) × (C_inv × k)ᵀ / (kᵀ × C_inv × k)
    delta = (v_new - v_old - W @ k) / (k.T @ C_inv @ k)
    W_new = W + torch.outer(delta, C_inv @ k)

    model.layers[layer_idx].mlp.W_out = nn.Parameter(W_new)
    return model

# 한계:
# - 단일 사실 편집은 잘 됨
# - 연속 편집 (수천 개) → 품질 저하 ("Ripple Effects")
# - 편집된 사실과 관련된 다른 사실들은 업데이트 안 됨
```

### MEMIT (Mass-Editing Memory in a Transformer)

```python
def memit_batch_edit(model, edit_requests: list[dict]):
    """
    MEMIT: 수천 개 사실을 동시에 편집
    ROME의 배치 버전 (여러 레이어에 분산 업데이트)

    편집 요청: [{"subject": ..., "relation": ..., "target": ...}, ...]
    """
    # ROME과 차이: 여러 레이어에 걸쳐 업데이트 분산
    # → 단일 레이어 과부하 방지

    target_layers = [8, 10, 12, 14, 16]  # 여러 레이어

    for layer_idx in target_layers:
        W = model.layers[layer_idx].mlp.W_out
        C_inv = compute_covariance_inverse(model, layer_idx)

        K = torch.stack([compute_key_vector(model, req["subject"], layer_idx)
                        for req in edit_requests], dim=1)  # [d, N]
        V_delta = torch.stack([
            compute_value_delta(model, req, layer_idx)
            for req in edit_requests
        ], dim=1)  # [d, N]

        # 배치 업데이트: ΔW = V_delta × Kᵀ × (K × Kᵀ + λI)⁻¹
        regularizer = 1e-4 * torch.eye(len(edit_requests), device=K.device)
        delta_W = V_delta @ K.T @ torch.linalg.inv(K @ K.T + regularizer)

        model.layers[layer_idx].mlp.W_out = nn.Parameter(W + delta_W)

    return model
```

---

## 지식 업데이트 (Temporal)

### 시간적 연속 학습

```python
class TemporalContinualLearner:
    """시간 순서로 데이터가 오는 설정"""

    def __init__(self, model, replay_buffer_size: int = 10000):
        self.model = model
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.drift_detector = StatisticalDriftDetector()

    def process_new_data(self, new_data: list, timestamp: str):
        """새 시간대 데이터 처리"""

        # 1. 분포 변화 감지
        if self.drift_detector.detect_drift(new_data):
            print(f"Distribution shift detected at {timestamp}")
            # 더 많은 새 데이터 비중 / 빠른 적응

        # 2. 새 데이터 + replay 혼합 학습
        replay_data = self.replay_buffer.sample(len(new_data) // 5)  # 20% replay
        combined = new_data + replay_data

        # 3. 학습 (낮은 LR로 점진적 업데이트)
        self._fine_tune(combined, lr=5e-6)

        # 4. 새 데이터를 버퍼에 추가 (Reservoir Sampling)
        for item in new_data:
            self.replay_buffer.add(item)

    def _fine_tune(self, data: list, lr: float = 5e-6):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        # ... 학습 루프
```

---

## Instruction Tuning과 Continual Learning

### Alignment Tax 분석

```python
class AlignmentEvaluator:
    """RLHF/SFT 후 능력 손실 (Alignment Tax) 측정"""

    def __init__(self, benchmarks: list):
        self.benchmarks = benchmarks  # [MMLU, HumanEval, GSM8K, ...]

    def measure_alignment_tax(self, base_model, aligned_model) -> dict:
        """정렬 전후 벤치마크 성능 비교"""
        results = {}
        for benchmark in self.benchmarks:
            base_score = self.evaluate(base_model, benchmark)
            aligned_score = self.evaluate(aligned_model, benchmark)
            tax = base_score - aligned_score  # 양수 = 성능 저하

            results[benchmark] = {
                "base": base_score,
                "aligned": aligned_score,
                "tax": tax
            }

        return results

# 관찰된 Alignment Tax 패턴:
# 일반적으로 SFT 후:
#   MMLU 5-shot: 약간 감소 (few-shot → 0-shot 전환)
#   MMLU 0-shot: 증가 (instruction following 학습)
#   HumanEval: 보통 향상 또는 유지
#   CreativeWriting: 향상

# RLHF 후 추가 tax:
#   안전 필터로 인한 특정 태스크 성능 감소
#   Verbosity 증가 (더 길게 답변)
```

### Continual Instruction Tuning 전략

```python
class ContinualInstructionTuner:
    """여러 라운드 instruction tuning에서 망각 방지"""

    def __init__(self, model, strategy: str = "replay"):
        self.model = model
        self.strategy = strategy
        self.task_datasets = {}  # 누적 데이터셋

    def add_task(self, task_name: str, task_data: list):
        self.task_datasets[task_name] = task_data

    def train_round(self, new_task: str, new_data: list) -> None:
        """새 라운드 학습 (망각 방지 포함)"""

        if self.strategy == "replay":
            # 이전 모든 태스크 데이터 + 새 태스크 데이터 혼합
            all_data = []
            for task, data in self.task_datasets.items():
                # 이전 태스크는 더 작은 비율 (균형)
                sample_size = min(len(new_data) // len(self.task_datasets), len(data))
                all_data.extend(random.sample(data, sample_size))
            all_data.extend(new_data)
            self._train(all_data)

        elif self.strategy == "joint":
            # 모든 태스크 데이터 함께 학습 (가장 효과적이지만 비용)
            all_data = []
            for data in self.task_datasets.values():
                all_data.extend(data)
            all_data.extend(new_data)
            self._train(all_data)

        elif self.strategy == "lora_per_task":
            # 태스크별 LoRA (base 망각 없음)
            lora_model = self._apply_lora(self.model)
            self._train_lora(lora_model, new_data)
            self.task_adapters[new_task] = self._extract_lora(lora_model)

        self.task_datasets[new_task] = new_data

    def _train(self, data: list, epochs: int = 1):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        # ... 표준 학습 루프
```

---

## 실전 권장사항

```
LLM 도메인 적응 시:
  1. 데이터 혼합: 도메인:일반 = 9:1 (시작점)
  2. LR: 원래 pretraining의 1/10 (2e-5 이하)
  3. Epochs: < 3 (overfitting, forgetting 주의)
  4. 평가: 도메인 + 일반 벤치마크 동시 추적
     → 일반 성능 10% 이상 하락 시 일반 데이터 비율 증가

LoRA로 지식 추가:
  망각 관점에서 가장 안전
  rank: 8~64 (도메인 복잡도에 따라)
  → rank 높이면 더 많은 도메인 지식 흡수
  → base 모델은 완전히 보존
  도메인 특화 adapter 여러 개 → 스위칭

대규모 지속 학습:
  Replay buffer 유지 (이전 데이터 10-20%)
  Reservoir Sampling으로 버퍼 다양성 유지
  주기적 감사: 모든 도메인 성능 추적
  점진적 LR 감소: 새 태스크마다 LR 조금 낮춤

실용 우선순위:
  1. Joint training (모든 데이터 함께) → 최고 성능, 비용 큼
  2. LoRA per task + 스위칭 → 깔끔, 태스크 ID 필요
  3. Replay 기반 → 균형적, 버퍼 관리 필요
  4. EWC/SI → 구현 복잡, LLM에서 실용성 제한
```

---

## Further Questions

**Q1. Catastrophic Forgetting을 완전히 없앨 수 있나?**
> 이론적으로: Progressive networks (태스크별 별도 컬럼) → 가능하지만 파라미터 선형 증가. 실용적으로: 완전 방지 어려움. Stability-Plasticity Dilemma: 망각 방지 ↑ = 새 학습 속도 ↓. 실무 접근: "허용 가능한 수준" 정의 → 일반+도메인 혼합 학습 + 정기적 평가로 모니터링.

**Q2. Continual Pretraining vs Finetuning 차이는?**
> CP: 비지도 CLM, 대량 도메인 텍스트, LR 낮음(2e-5), 도메인 지식 주입. SFT: 지도 instruction-response, 수천~수만 고품질 예시, LR 더 낮음(1e-5), 태스크/형식 학습. 권장 순서: Base → CP(도메인 지식) → SFT(태스크 학습). CodeLLaMA: LLaMA-2 → 코드 CP → Code SFT.

**Q3. LLM의 지식 커트오프 문제를 어떻게 해결하나?**
> 단기(즉각): RAG - 외부 DB에 새 지식, 검색 후 컨텍스트로 제공. 중기: 지속 학습 - 주기적 (월간/분기) 새 데이터로 학습, 새 버전 릴리즈. 장기: Knowledge editing (ROME, MEMIT) - 특정 사실만 targeted 수정 가능. 현실: RAG가 가장 실용적. 근본적으로 static weight vs dynamic world 불일치 문제.

**Q4. 지식 편집(Knowledge Editing)의 한계는?**
> ROME: 단일 사실 편집은 잘 됨, 수천 개 배치 편집은 품질 급락. Ripple Effects: 편집된 사실과 관련된 다른 사실들 자동 업데이트 안 됨 ("에펠탑은 파리에 있다"를 베를린으로 바꿔도 관련 지식 업데이트 안 됨). MEMIT: 배치 편집 가능하지만 여전히 수천 개 이상에서 품질 저하. 실용적: 소수 중요 사실 편집에만 적용, 대규모 지식 업데이트는 재학습이 안전.
