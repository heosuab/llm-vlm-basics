# 심화 학습 기법

## Curriculum Learning

### 개념
```
인간 학습처럼 쉬운 것부터 어려운 것 순서로 학습
vs Random shuffling (표준)

이점:
  빠른 초기 수렴
  더 나은 최종 성능 (일부 태스크)
  학습 안정성 향상

적용 예:
  문장 길이 기준: 짧은 시퀀스 먼저 → 긴 시퀀스
  난이도 기준: 쉬운 예시 먼저 → 어려운 예시
```

### 구현 방법
```python
# 길이 기반 커리큘럼
class CurriculumDataLoader:
    def __init__(self, dataset, schedule):
        self.dataset = sorted(dataset, key=lambda x: len(x['input_ids']))
        self.schedule = schedule  # [(step, max_length), ...]

    def get_batch(self, step):
        max_len = self._get_max_len(step)
        # max_len 이하 시퀀스만 샘플링
        filtered = [x for x in self.dataset
                   if len(x['input_ids']) <= max_len]
        return random.sample(filtered, batch_size)

    def _get_max_len(self, step):
        for threshold, max_len in self.schedule:
            if step < threshold:
                return max_len
        return float('inf')

# 스케줄 예: 처음 1000 steps는 512, 이후 2048, 나중엔 8192
schedule = [(1000, 512), (5000, 2048), (float('inf'), 8192)]
```

### LLM에서의 커리큘럼
```
데이터 품질 커리큘럼:
  처음: 모든 품질 데이터
  나중: 고품질 데이터만 (FineWeb filtering)

도메인 커리큘럼:
  처음: 일반 웹 텍스트 (쉬운 언어 패턴)
  나중: 과학 논문, 코드 (복잡한 패턴)

실제로는 단순 혼합(mixing ratio)이 커리큘럼보다 효과적인 경우 많음
→ 커리큘럼 효과는 태스크/데이터 의존적
```

---

## Activation Checkpointing (Gradient Checkpointing)

### 원리
```
문제: Backward pass를 위해 모든 레이어 activation 보관
  LLaMA-3-8B, seq=4096, batch=4:
    레이어당 activation: 4096 × 8192 × 4 × 2 bytes ≈ 256 MB
    32 레이어: ~8 GB activation만으로

해결: Forward 시 일부만 저장, Backward 시 재계산

표준 전략:
  매 n번째 레이어만 저장 (n=2 or √num_layers)
  재계산: O(1) 추가 시간 per 저장 레이어

메모리-속도 트레이드오프:
  저장 레이어 수 ↓ = 메모리 ↓, 속도 ↓
  최적: O(√n) 레이어 저장 → 메모리 O(√n), 속도 ~30% 감소
```

### 구현
```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerLayer(nn.Module):
    def __init__(self, layer):
        self.layer = layer

    def forward(self, x, use_checkpoint=True):
        if use_checkpoint:
            # 이 레이어의 forward를 나중에 재계산
            return checkpoint(self.layer, x,
                            use_reentrant=False)  # PyTorch 2.0+
        else:
            return self.layer(x)

# Selective checkpointing: attention만 체크포인트
# (attention이 메모리 많이 사용)
class TransformerBlock(nn.Module):
    def forward(self, x):
        # Attention: checkpoint (메모리 집약적)
        x = x + checkpoint(self.attention, x)
        # FFN: 직접 계산 (상대적으로 작음)
        x = x + self.ffn(x)
        return x
```

### Activation Checkpointing 최적화
```
Selective Activation Checkpointing:
  모든 레이어가 아닌 특정 레이어만
  FlashAttention: attention은 이미 메모리 효율적
  → FFN, attention 이외 레이어는 체크포인트

Micro-batching과의 관계:
  PP에서 micro-batch는 inflight activation 유지
  Checkpointing은 각 micro-batch 내 저장량 감소

Recomputation-aware 스케줄링:
  재계산할 레이어 = 계산이 빠른 레이어 (LayerNorm 등)
  저장할 레이어 = 재계산이 느린 레이어 (attention score 등)
```

---

## Mixed Precision Training 심화

### FP32, BF16, FP16, FP8 비교
```
FP32: 1 sign + 8 exp + 23 mantissa, max ≈ 3.4×10³⁸
BF16: 1 sign + 8 exp + 7 mantissa,  max ≈ 3.4×10³⁸  (FP32와 동일 범위)
FP16: 1 sign + 5 exp + 10 mantissa, max ≈ 65504       (범위 제한)
FP8 E4M3: 1+4+3, max = 448
FP8 E5M2: 1+5+2, max ≈ 57344  (gradient용, 더 큰 범위)

Mixed Precision 표준 (BF16):
  Forward: BF16 (빠름, 메모리 절반)
  Backward: BF16 gradient
  Loss scaling: 불필요 (BF16은 FP32와 같은 범위)
  Optimizer: FP32 master weights + FP32 states
  Weight update: FP32로 계산 후 BF16으로 변환
```

### Loss Scaling (FP16 사용 시)
```
FP16의 문제:
  작은 gradient: underflow (< 6e-5 → 0으로 표현)
  큰 gradient: overflow (> 65504 → NaN)

Loss scaling:
  loss = scaled_loss / scale_factor
  gradient: 자동으로 scale_factor배 커짐
  → underflow 방지

Dynamic loss scaling:
  scale_factor를 동적 조정
  - NaN/Inf 없으면 scale 증가 (2× per 2000 steps)
  - NaN/Inf 발생하면 scale 감소 (0.5×)

PyTorch:
  scaler = torch.cuda.amp.GradScaler()
  with torch.cuda.amp.autocast():
      loss = model(input)
  scaler.scale(loss).backward()
  scaler.unscale_(optimizer)
  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  scaler.step(optimizer)
  scaler.update()
```

### FP8 학습 (최신)
```
목적: BF16 대비 2× 처리량, 절반 메모리

어려움:
  FP8의 좁은 표현 범위 → scaling 필수
  Activation 이상치 → 양자화 오류

DeepSeek-V3 FP8 학습:
  W (weights): FP8 E4M3
  A (activation): FP8 E4M3
  G (gradient): FP8 E5M2 (더 넓은 범위 필요)
  Master weights: BF16

Fine-grained quantization:
  블록별 스케일 팩터 (전체 tensor 하나의 scale X)
  128-element block마다 별도 scale
  → 이상치 영향 최소화

Accumulation:
  FP8 곱셈 → FP32 누적 (정확도 유지)
  최종: FP32 → BF16 변환
```

---

## 학습 안정성 기법

### Gradient Clipping
```
torch.nn.utils.clip_grad_norm_(parameters, max_norm)

작동:
  1. 전체 gradient의 L2 norm 계산
  2. norm > max_norm이면 전체 gradient를 max_norm/norm로 스케일

왜 필요한가:
  Loss spike → 큰 gradient → weight 급변 → 발산
  gradient clipping: 방향은 유지, 크기만 제한

max_norm 선택:
  1.0 (표준)
  학습 초기: gradient norm 분포 확인 후 설정
  너무 작으면 학습 느림, 너무 크면 효과 없음

관련 모니터링:
  gradient norm이 max_norm 근처: 자주 clip 발생 → LR 낮추기
  gradient norm이 매우 작음: vanishing gradient 주의
```

### Warm-up 전략
```
왜 warm-up이 필요한가:
  초기 weight: 무작위, 큰 gradient
  높은 LR → 초기에 발산 위험
  Adam: 초기 m, v 통계가 부정확 (bias correction 있지만 불완전)

Linear warmup (표준):
  lr = max_lr × (step / warmup_steps)
  warmup_steps = 1000 ~ 4000 (모델 크기에 따라)

Cosine warmup:
  초반에 더 느리게 증가
  더 안정적이지만 큰 차이는 없음

warmup 없이 가능한 경우:
  Pre-LN 아키텍처 (gradient flow 안정)
  매우 낮은 LR로 시작
  μP (Maximal Update Parameterization) 사용 시
```

### μP (Maximal Update Parameterization)
```
문제: 모델 크기에 따른 LR 재조정 필요
  작은 모델: LR=3e-4 최적
  큰 모델: LR=?  (단순 스케일링 불가)

μP: width를 바꿔도 최적 LR이 같도록 초기화/스케일링 조정

핵심 변경:
  임베딩 레이어: 1/width로 LR 스케일
  hidden 레이어: 1/width로 LR 스케일
  출력 레이어: 1/width로 초기화 스케일

이점:
  작은 모델에서 찾은 최적 하이퍼파라미터 → 큰 모델에 그대로 적용
  "Hyperparameter Transfer" 가능
  LR, momentum, weight decay 모두 전달

실제 사용:
  Cerebras, EleutherAI 등에서 채택
  GPT-3 학습 비용의 작은 분수로 최적 HP 탐색
```

---

## 효율적 어텐션 변형

### Flash Attention 3 (2024)
```
H100 최적화:
  Warp specialization: 각 warp가 softmax vs matmul 역할 분리
    softmax warp: score 계산 + 정규화
    matmul warp: 행렬 곱 계산
    → 두 연산 파이프라인 병렬화

  Pingpong scheduling:
    WGMMA (Warpgroup Matrix Multiply Accumulate)와
    softmax를 교대 실행 → TMA 활용

  FP8 지원: FP8 matmul + FP32 accumulation
    → 2× 처리량 향상

성능 (H100):
  FA2 대비 1.5-2× 향상
  BF16: 최대 750 TFLOPS (이론 989의 76%)
```

### Multi-Token Prediction (MTP)
```
표준: 한 번에 1 토큰 예측
MTP: 한 번에 n 토큰 예측 (보조 head 사용)

DeepSeek-V3 적용:
  주 head: 다음 1 토큰
  보조 head 1: 다음 2 토큰
  보조 head 2: 다음 3 토큰

학습:
  L = L_main + λ·L_mtp1 + λ·L_mtp2
  → 더 풍부한 학습 신호

추론:
  보조 head를 Speculative Decoding의 draft로 사용
  → 추론 속도 향상 (주 모델이 draft 검증)

이점:
  학습: 더 많은 gradient signal → 표현력 향상
  추론: draft 모델 없이 Speculative Decoding
```

---

## Scaling Law 심화

### Chinchilla Law 재해석
```
Chinchilla (Hoffmann et al., 2022):
  Compute-optimal: N_params = 20 × N_tokens
  7B 모델 → 140B tokens

하지만 실제 모델들:
  LLaMA-3-8B: 15T tokens (compute-suboptimal)
  이유: inference cost 최적화

올바른 프레임:
  Training compute budget = N × D × C_fwd
  Inference compute budget = N × N_queries × C_fwd

  훈련 비용: 상수 (일회성)
  추론 비용: N × (무한히 늘어나는 쿼리 수)

  → 더 작은 N, 더 많은 D 학습이 inference 관점에서 최적!
```

### Neural Scaling Laws 수식
```
손실 스케일링:
  L(N, D) = L_∞ + A/N^α + B/D^β

  L_∞: 달성 가능한 최소 손실 (데이터 엔트로피)
  A, B: 상수
  α ≈ 0.076, β ≈ 0.095 (Chinchilla 추정)
  N: 파라미터 수, D: 학습 토큰 수

Compute budget C = 6ND (학습 FLOPs 근사):
  최적 N_opt = (A·α/B·β)^(1/(α+β)) · C^(β/(α+β))
  최적 D_opt = (B·β/A·α)^(1/(α+β)) · C^(α/(α+β))

  C가 2배: N과 D 각각 ≈ 1.4배씩 증가
```

### Emergent Ability와 Scaling
```
특정 능력이 갑자기 나타나는 임계점:

관찰된 임계점:
  Arithmetic (3-digit): ~100B params
  Chain-of-thought: ~100B params
  Code generation: ~50B params
  다국어 번역: ~10B params

이게 진짜 emergent인가?:
  Schaeffer (2023): 측정 지표의 비선형성 때문
  이진 지표(pass/fail) → 급격해 보임
  연속 지표 → 점진적 향상

실용적 의미:
  특정 기능 필요 → 최소 파라미터 규모 추정 가능
  근거: 유사 태스크의 임계점 참고
```

---

## 학습 데이터 최적화

### Token Budget과 반복
```
표준 pretraining: 각 토큰 1번 학습 (epoch=1)
데이터 부족 시:
  epoch 반복 → 같은 데이터 여러 번
  과도한 반복 → 암기/과적합

연구 결과 (Muennighoff et al., 2023):
  최대 4 epoch까지는 성능 저하 미미
  그 이상: 점진적 성능 저하
  → 데이터 부족 시 4× 반복이 실용적 한계

예외:
  수학/코딩 데이터: 더 많은 반복도 효과적
  이유: 정확한 패턴 학습 필요 (암기가 도움)
```

### 합성 데이터 (Synthetic Data)
```
Phi 계열 (Microsoft):
  "Textbook quality" 합성 데이터
  GPT-4로 생성한 교육적 텍스트
  3.5B 모델로 70B 대비 우수한 성능

Magpie (2024):
  Instruction-following 데이터 자동 생성
  LLM에게 자신의 instruction 생성하도록 유도
  → 다양한 instruction + 자동 품질 필터

WizardLM (Evol-Instruct):
  기존 instruction을 복잡하게 변형
  더 어렵고 다양한 instruction 생성

주의점:
  Model collapse: 합성 데이터만 학습 → 다양성 감소
  반드시 실제 데이터와 혼합 필요
```

---

## Further Questions

**Q1. Activation Checkpointing의 메모리-속도 트레이드오프를 최적화하는 방법은?**
```
Selective checkpointing:
  - FlashAttention은 이미 메모리 효율적 → 체크포인트 불필요
  - LayerNorm, Dropout: 빠르게 재계산 → 체크포인트 선택적
  - FFN: 가장 메모리 많이 쓰는 구간 → 체크포인트 우선

블록 수준 최적화:
  - Transformer block 단위로 체크포인트
  - Block 내에서는 full activation 유지

ZeRO-3 조합:
  파라미터 샤딩 + activation 체크포인트 조합
  → 최대 메모리 효율 (속도 트레이드오프 가장 큼)
```

**Q2. BF16이 FP16보다 학습에 유리한 이유는?**
```
핵심: exponent bits 수
  BF16: 8 exp bits → FP32와 같은 표현 범위 (10^-38 ~ 10^38)
  FP16: 5 exp bits → 좁은 범위 (6×10^-5 ~ 65504)

FP16 문제:
  작은 gradient → underflow (0으로 표현)
  LR schedule, BN 통계 등 작은 값 처리 어려움
  → Loss scaling 필수 (복잡, 실패 가능)

BF16:
  Loss scaling 불필요
  FP32 대비 mantissa 정밀도 낮지만 실용적으로 충분
  A100 이후: BF16 기본 권장

단점: BF16은 더 낮은 mantissa 정밀도 (7 bits vs 10)
  → 매우 정밀한 계산 필요 시 FP16 또는 FP32
```

**Q3. μP가 실용적으로 중요한 이유는?**
```
비용 절감:
  기존: 새 모델 크기마다 하이퍼파라미터 탐색 필요
    70B 모델 HP 탐색 = 수십만 달러
  μP: 소형 모델(7B)에서 탐색 → 70B에 그대로 적용
    7B HP 탐색 = 수천 달러

신뢰성:
  스케일에 무관한 최적 HP → 예측 가능한 스케일링
  안정적인 loss curve

한계:
  구현 비표준 (라이브러리 지원 필요)
  일부 아키텍처 변경 필요 (임베딩, 출력 레이어)
  MoE, 특수 아키텍처에서 검증 부족
```
