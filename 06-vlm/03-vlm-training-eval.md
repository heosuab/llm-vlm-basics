# VLM 학습 전략 & 평가

## VLM 학습 단계

### 표준 3단계 학습 (LLaVA 스타일)

```
Stage 1: Pretraining (Alignment / Feature Learning)
  데이터: 이미지-캡션 쌍 (CC3M, LAION 등)
  목표: Vision Encoder와 LLM의 임베딩 공간 정렬
  학습: Projection layer (MLP Connector)만 학습
  Vision Encoder: 고정 (frozen)
  LLM: 고정 (frozen)
  Loss: 캡션 생성 NLL (next token prediction)
  크기: 보통 수백만~수천만 이미지-캡션 쌍

Stage 2: Visual Instruction Tuning (SFT)
  데이터: VQA, 이미지 대화, 다중 이미지 instruction pairs
  목표: 지시 따르기, 시각적 질의응답, 추론 능력
  학습: Projection layer + LLM (전체 또는 LoRA)
  Vision Encoder: 고정 또는 부분 학습
  크기: 수십만~수백만 샘플

Stage 3: RLHF/DPO (선택적 정렬)
  인간 선호도 기반 추가 정렬
  VLFeedback, RLAIF-V, RLHF-V 등
  환각 감소, 안전성 향상
```

### InternVL2 스타일 (강력한 Vision Encoder 사용)

```
Stage 1: Contrastive Pretraining (Vision-Language Alignment)
  CLIP/SigLIP 방식으로 Vision Encoder 사전학습
  대규모 이미지-텍스트 쌍으로 대조 학습

Stage 2: Generative Pretraining
  Vision Encoder + MLP + LLM 전체 학습
  이미지 캡션 생성 태스크

Stage 3: Supervised Fine-Tuning (SFT)
  혼합 멀티모달 instruction 데이터
  Vision Encoder를 LLM과 함께 파인튜닝
  → 더 나은 시각적 이해

핵심: Vision Encoder도 함께 학습 → 더 강한 표현력
```

---

## 학습 구현 예시

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from torch.utils.data import DataLoader
import torch

# LLaVA 스타일 학습
class VLMTrainer:
    def __init__(self, model, processor, stage=1):
        self.model = model
        self.processor = processor
        self.stage = stage
        self._freeze_params()

    def _freeze_params(self):
        if self.stage == 1:
            # Stage 1: Connector만 학습
            for name, param in self.model.named_parameters():
                if "connector" in name or "projection" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif self.stage == 2:
            # Stage 2: LLM + Connector 학습
            for name, param in self.model.named_parameters():
                if "vision_encoder" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def compute_loss(self, batch):
        images = batch["images"]
        input_ids = batch["input_ids"]
        labels = batch["labels"]  # user 메시지 부분 -100으로 마스킹

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=images,
            labels=labels
        )
        return outputs.loss

    def train_step(self, batch, optimizer):
        loss = self.compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
```

---

## 학습 데이터

### Pretraining 데이터 (Stage 1)

| 데이터셋 | 크기 | 설명 |
|---------|------|------|
| LAION-5B | 5B 쌍 | 웹 크롤링 이미지-Alt text |
| COYO-700M | 700M 쌍 | 고품질 필터링 |
| CC3M/CC12M | 3M/12M | 개념적 캡션 |
| YFCC100M | 100M | Flickr 이미지 |
| DataComp-1B | 1B 쌍 | CLIP 필터링된 고품질 |
| WIT (Wikipedia Image-Text) | 37.5M | Wikipedia 이미지-텍스트 |

### SFT 데이터 (Stage 2)

| 데이터셋 | 크기 | 설명 |
|---------|------|------|
| LLaVA-665K | 665K | GPT-4 생성 instruction |
| ShareGPT4V | 1.2M | 고품질 GPT-4V 캡션 |
| LVIS-Instruct4V | 220K | 세부 설명 instruction |
| ALLaVA | 1.5M | 합성 고품질 데이터 |
| Cambrian-10M | 10M | 다양한 혼합 데이터 |
| M3IT | 2.4M | 다국어 멀티모달 |
| OKVQA, VQA-v2 | 합산 2M+ | 시각적 질의응답 |

### 특화 데이터 (도메인별)

```
OCR/문서: TextVQA, DocVQA, OCRBench, FUNSD
수학: MathVista, Geo170K, CLEVR
과학: ScienceQA, MMStar, SCIENCEQA
차트: ChartQA, FigureQA, Chart2Text
비디오: VideoChat2, VideoInstruct, MSRVTT
그라운딩: RefCOCO, GQA, Visual Genome
의료: PMC-VQA, SLAKE, VQA-RAD
코드/UI: Screen2Words, ScreenQA
```

---

## Dynamic Resolution 처리

### AnyRes / Dynamic Tiling

```python
def process_high_res_image(image, max_tiles=9, tile_size=336):
    """
    고해상도 이미지를 타일로 분할하여 처리
    LLaVA-Next, InternVL2, Qwen2-VL에서 사용
    """
    w, h = image.size

    # 최적 그리드 계산
    # 예: 1344x336 이미지 → 4×1 그리드
    aspect_ratio = w / h
    best_grid = find_closest_aspect_ratio(
        aspect_ratio, w, h, tile_size, max_tiles
    )

    # 이미지를 그리드에 맞게 리사이즈
    target_w = best_grid[0] * tile_size
    target_h = best_grid[1] * tile_size
    resized = image.resize((target_w, target_h))

    # 타일 분할
    tiles = []
    for row in range(best_grid[1]):
        for col in range(best_grid[0]):
            tile = resized.crop((
                col * tile_size, row * tile_size,
                (col+1) * tile_size, (row+1) * tile_size
            ))
            tiles.append(tile)

    # 썸네일도 추가 (전체 맥락)
    thumbnail = image.resize((tile_size, tile_size))
    return [thumbnail] + tiles  # [global] + [local tiles]

# 결과: 각 336x336 타일 → ViT 처리 → 각 144 토큰 → 전체 수천 토큰
# InternVL2: Pixel Shuffle로 4개 패치 → 1개로 합쳐 토큰 수 감소
```

### Pixel Shuffle (InternVL2)

```python
def pixel_shuffle(x, scale_factor=0.5):
    """
    인접 4개 패치를 채널로 합쳐 토큰 수를 1/4로 감소
    예: [1, 1024, 1024] → pixel_shuffle → [1, 256, 4096]
    """
    n, h, w, c = x.shape
    h_new = int(h * scale_factor)
    w_new = int(w * scale_factor)
    x = x.reshape(n, h_new, 2, w_new, 2, c)
    x = x.transpose(2, 3).reshape(n, h_new, w_new, 4 * c)
    return x

# 256 토큰(16×16 패치)을 64 토큰으로 압축
# LLM의 계산 부담 감소
```

---

## 데이터 혼합 전략

```python
# VLM 학습 데이터 혼합 예시
data_mix = {
    "general_caption": 0.30,    # 일반 이미지-텍스트 정렬
    "vqa_conversation": 0.25,   # 다중 턴 시각적 QA
    "ocr_document": 0.15,       # 문서/OCR 이해
    "math_chart": 0.10,         # 수학/차트 이해
    "pure_text": 0.10,          # 텍스트만 (언어 능력 유지)
    "grounding": 0.05,          # 위치/좌표 태스크
    "video": 0.05,              # 비디오 이해
}

# 주의: pure_text 데이터 포함 → "catastrophic forgetting" 방지
# LLM의 언어 능력이 vision 학습으로 저하되는 현상 방지
```

---

## VLM RLHF & 정렬

### RLHF-V (2023)

```
인간이 VLM 출력 비교 → Preference 데이터 수집
Bradley-Terry 모델로 Reward 학습
PPO로 VLM 정렬

특화: 환각(hallucination) 감소에 집중
데이터: 환각-비환각 응답 쌍
```

### RLAIF-V

```
AI(강력한 VLM)가 응답 품질 평가
인간 annotation 없이 선호도 데이터 생성
→ 대규모 VLM 정렬 가능
```

### VCD (Visual Contrastive Decoding)

```python
# 이미지 있을 때 vs 없을 때 logit 차이 활용
# 시각적 정보에 더 집중하도록

def vcd_decode(model, input_ids, pixel_values, alpha=1.0):
    # 정상 (이미지 있음)
    logits_with_image = model(
        input_ids=input_ids,
        pixel_values=pixel_values
    ).logits

    # 이미지 없음 (또는 노이즈 이미지)
    logits_without_image = model(
        input_ids=input_ids,
        pixel_values=None  # 또는 noise_image
    ).logits

    # 대조 디코딩: 이미지 있을 때 - 없을 때
    # 이미지 정보에만 의존하는 logit 강조
    adjusted_logits = logits_with_image + alpha * (
        logits_with_image - logits_without_image
    )
    return adjusted_logits
```

---

## 주요 벤치마크

### 종합 이해력

| 벤치마크 | 측정 내용 | 특징 |
|---------|---------|------|
| MMBench | 종합 VLM 평가 (200+ 능력 차원) | MCQ, 체계적 |
| MMMU | 대학 수준 멀티모달 이해 (30개 과목) | 전문 지식 필요 |
| MMStar | 환각 방지, 정밀 평가 | 언어 편향 제거 |
| Blink | 인간에게 쉽지만 VLM에 어려운 시각 태스크 | 3D, 공간 추론 |
| SEED-Bench | 이미지/비디오 이해, 12개 차원 | 대규모 (19K 문제) |

### OCR & 문서 이해

| 벤치마크 | 측정 내용 |
|---------|---------|
| TextVQA | 장면 내 텍스트 읽기 |
| DocVQA | 문서 레이아웃 이해 |
| OCRBench | 종합 OCR 능력 (1K 과제) |
| InfoVQA | 인포그래픽 이해 |
| ChartQA | 차트 해석 |
| AI2D | 과학 다이어그램 |

### 수학/과학

| 벤치마크 | 측정 내용 |
|---------|---------|
| MathVista | 수학 시각 추론 |
| MathVision | 고급 경시 수학 |
| MMMU-Pro | 전문가 수준 (더 어려운 MMMU) |
| ScienceQA | 과학 QA (이미지+텍스트) |

### 환각 평가

| 벤치마크 | 측정 내용 |
|---------|---------|
| HallusionBench | 시각적 환각, 언어 편향 분리 평가 |
| POPE | 객체 존재 여부 환각 (Popular/Adversarial) |
| MMHal-Bench | 다양한 환각 유형 (8개 범주) |
| AMBER | 환각 종합 평가 |

### 비디오 이해

| 벤치마크 | 측정 내용 |
|---------|---------|
| MVBench | 비디오 이해 (20개 유형) |
| Video-MME | 비디오 종합 평가 |
| EgoSchema | 긴 형식 비디오 추론 |
| MSVD-QA | 비디오 QA |

---

## 평가 도구

```python
# lmms-eval (EleutherAI) - VLM 평가 표준 도구
from lmms_eval import evaluator

results = evaluator.simple_evaluate(
    model="llava",
    model_args="pretrained=llava-hf/llava-1.5-7b-hf",
    tasks=["mmbench_en", "mmmu_val", "pope"],
    batch_size=1
)

# 주요 VLM 평가 지표
print(f"MMBench: {results['results']['mmbench_en']['acc']:.2%}")
print(f"MMMU: {results['results']['mmmu_val']['acc']:.2%}")
print(f"POPE: {results['results']['pope']['acc']:.2%}")
```

---

## VLM 특유의 문제들

### Hallucination (환각)

```
이미지에 없는 객체를 있다고 말함
언어 편향: 이미지 무시하고 LLM 사전 확률로 답변

원인:
  1. SFT 데이터의 노이즈 (잘못된 레이블)
  2. 이미지 인코딩의 정보 손실 (토큰 수 제한)
  3. LLM의 강한 언어 사전 (prior)
  4. 시각 정보보다 텍스트 패턴에 의존하는 학습

해결:
  1. RLHF/DPO로 환각에 패널티
  2. 이미지 토큰 수 증가 (Dynamic Resolution)
  3. VCD (Visual Contrastive Decoding)
     → 이미지 있을 때 vs 없을 때 logit 차이 활용
  4. 세밀한 grounding 데이터로 SFT
  5. POPE, HallusionBench로 측정 후 타겟 학습

POPE 측정:
  "Is there a [object] in the image?" (Yes/No)
  Popular: 자주 등장하는 객체 (쉬운 환각)
  Adversarial: 공존하는 객체 (어려운 환각)
```

### 언어 vs 시각 편향

```
문제: "이 사과는 무슨 색인가?" → 이미지 무시하고 "빨간색"
  → LLM이 "사과 = 빨간색"이라고 학습되어 있어 이미지 무시

HallusionBench로 측정:
  이미지를 잘못 조작 + 질문 → 언어 사전대로 답하는지 체크

해결:
  1. 시각 정보를 더 강조하는 학습
  2. Counter-intuitive VQA 데이터 (예: 파란 사과 이미지)
  3. Grounding-dense 데이터 (객체 위치 기반 QA)
```

### 위치/공간 이해

```
"왼쪽 사람이 누구인가?" → 좌우 혼동
좌표 기반 QA에서 정확도 낮음

해결:
  RefCOCO 등 좌표 레이블이 있는 SFT 데이터
  SpatialVQA, VSR (Visual Spatial Reasoning) 데이터

Blink 벤치마크: 3D 공간 이해, 깊이 추정, 카메라 각도
대부분의 VLM이 인간 성능의 절반에도 못 미침
```

### 멀티이미지 이해

```
여러 이미지 비교 → 차이/공통점 찾기
이미지 시퀀스 이해 (스토리, 비디오 프레임)

해결:
  Multi-image interleaved 데이터
  Mantis, MMC 같은 멀티이미지 데이터셋
  Cross-image attention 메커니즘
```

---

## 최신 VLM 성능 비교 (2024-2025)

| 모델 | Vision Encoder | LLM | MMBench | MMMU | TextVQA |
|------|---------------|-----|---------|------|---------|
| GPT-4V/4o | 미공개 | GPT-4 | 81.0 | 56.8 | 78.0 |
| Gemini 1.5 Pro | 미공개 | Gemini | 82.2 | 58.5 | 73.5 |
| Claude 3.5 Sonnet | 미공개 | Claude 3.5 | - | 68.3 | - |
| LLaVA-OneVision-72B | SigLIP | Qwen2-72B | 85.4 | 56.8 | 77.4 |
| InternVL2-76B | InternViT-6B | InternLM2-20B | 86.5 | 58.3 | 84.4 |
| Qwen2-VL-72B | 내장 ViT | Qwen2-72B | 89.0 | 64.5 | 85.5 |
| LLaMA-3.2-90B-V | 미공개 | LLaMA-3.2-90B | - | 60.3 | - |

---

## 학습 안정성 & 팁

```
Vision Encoder 학습률:
  LLM보다 낮은 lr 사용 (보통 1/10)
  이유: Vision Encoder는 이미 좋은 표현 학습되어 있음
  과도한 학습 → CLIP 임베딩 공간 붕괴

연결자(Connector) 초기화:
  Random init 가능하지만 pretrained 시작이 더 좋음
  Stage 1에서 충분히 학습 필수

데이터 순서:
  Stage 1: 쉬운 단순 캡션 → 나중에 복잡한 VQA
  혼합 비율: Curriculum learning 적용 권장

Catastrophic Forgetting:
  순수 텍스트 데이터를 학습 데이터에 10~20% 포함
  LLM 언어 능력 유지
```

---

## Further Questions

**Q. VLM에서 환각을 줄이는 방법은?**
> 1) 고품질 정확한 SFT 데이터. 2) RLHF로 환각에 패널티 (RLHF-V, RLAIF-V). 3) VCD 같은 디코딩 전략. 4) 이미지 토큰 수 증가 (Dynamic Resolution). 5) 세밀한 시각 grounding 데이터 추가. 6) POPE, HallusionBench로 측정 후 약점 데이터 보강.

**Q. VLM의 vision encoder를 교체하면 어떻게 되나?**
> 임베딩 공간이 달라져 Projection Layer 재학습 필요. Stage 1부터 다시 학습 권장. Vision Encoder도 VLM 태스크에 맞게 파인튜닝하면 더 좋음. InternVL의 접근: VLM 전용 Vision Encoder(InternViT) 개발로 성능 향상.

**Q. Dynamic Resolution이 왜 중요한가?**
> 고정 크기 처리(LLaVA-1.5 방식)는 고해상도 이미지에서 세부 정보 손실. OCR, 문서 이해, 차트 해석에서 필수. Dynamic tiling: 이미지를 여러 타일로 분할해 고해상도 유지. 하지만 토큰 수 증가 → 계산 비용 증가. Pixel Shuffle로 토큰 압축하여 균형 유지.

**Q. VLM에서 비디오를 어떻게 처리하나?**
> 비디오 = 프레임 시퀀스. 균일 샘플링 또는 내용 기반 키프레임 선택. 시간 임베딩 추가 (M-ROPE, 타임스탬프 텍스트). 긴 비디오: 청크 처리 → 요약 → 전체 이해. 현재 한계: 매우 긴 비디오(1시간+)는 모든 모델에 어려움.
