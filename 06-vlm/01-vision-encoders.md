# Vision Encoders & Vision-Language Pretraining

## Vision Transformer (ViT) [Dosovitskiy et al., 2020]

```
핵심: 이미지를 패치(patch)로 나눠 Transformer 입력으로

처리 과정:
  이미지 (H × W × C)
    ↓ 패치 분할 (P × P 크기)
  N = HW/P² 개의 패치
    ↓ Linear Projection (flatten + 선형 변환)
  패치 임베딩 (N × D)
    ↓ + Learnable Position Embedding + [CLS] 토큰
  Transformer Encoder (L개 레이어)
    ↓
  [CLS] 토큰 출력 → 이미지 표현

일반적 설정:
  Patch size: 14×14 또는 16×16
  ViT-B/16: 12레이어, 12헤드, d=768, 86M params
  ViT-L/14: 24레이어, 16헤드, d=1024, 307M params
  ViT-H/14: 32레이어, 16헤드, d=1280, 632M params

224×224 이미지, patch 16×16:
  N = (224/16)² = 196 패치 + 1 CLS = 197 토큰
```

### ViT 구현

```python
import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size

        # 패치를 한 번에 선형 변환 (Conv2d 활용)
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)        # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)        # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS 토큰과 위치 임베딩
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                batch_first=True
            ),
            num_layers=depth
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, D]

        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, x], dim=1)           # [B, N+1, D]
        x = x + self.pos_embed

        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x[:, 0])  # CLS 토큰 사용
```

---

## CLIP (Contrastive Language-Image Pretraining) [Radford et al., 2021]

```
학습 목표: 이미지-텍스트 쌍 매칭

구조:
  Image Encoder (ViT 또는 ResNet) → 이미지 임베딩 iᵢ
  Text Encoder (Transformer) → 텍스트 임베딩 tᵢ
  → 각 임베딩을 같은 공간으로 projection + L2 정규화

손실 함수 (InfoNCE / NT-Xent):
  배치 내 N개 이미지-텍스트 쌍
  올바른 쌍의 유사도 최대화
  잘못된 쌍(N²-N개)의 유사도 최소화

  L_i→t = -log( exp(sim(iᵢ, tᵢ)/τ) / Σⱼ exp(sim(iᵢ, tⱼ)/τ) )
  L_t→i = -log( exp(sim(tᵢ, iᵢ)/τ) / Σⱼ exp(sim(tⱼ, iᵢ)/τ) )
  L = (L_i→t + L_t→i) / 2

  τ: 학습 가능한 온도 파라미터 (초기값 0.07)

학습 데이터: 400M 이미지-텍스트 쌍 (웹 크롤링)
```

### CLIP 구현 및 Zero-shot 분류

```python
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def zero_shot_classify(image, class_names: list):
    """
    CLIP zero-shot 이미지 분류
    클래스 레이블 없이 텍스트 유사도로 분류
    """
    # 텍스트 템플릿
    texts = [f"a photo of a {name}" for name in class_names]

    inputs = processor(
        images=image,
        text=texts,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # logits_per_image: [1, num_classes]
    probs = F.softmax(outputs.logits_per_image, dim=-1)
    predicted_class = class_names[probs.argmax().item()]
    return predicted_class, probs

# 이미지 검색: 이미지 임베딩 vs 텍스트 임베딩 코사인 유사도
def image_text_similarity(image, text):
    inputs = processor(images=image, text=text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.logits_per_image.item()
```

---

## SigLIP (Sigmoid Loss for Language Image Pre-Training) [Zhai et al., 2023]

```
CLIP과의 핵심 차이:
  CLIP: Softmax over 배치 (전체 배치를 분모로)
  SigLIP: Sigmoid (독립적 이진 분류)

SigLIP 손실:
  L = -1/N² · Σᵢ,ⱼ log σ(zᵢⱼ · yᵢⱼ · γ - δ)

  yᵢⱼ = +1 if (i==j, 매칭 쌍) else -1
  zᵢⱼ = sim(iᵢ, tⱼ) (코사인 유사도)
  γ, δ: 학습 가능한 스케일/편향

  → 각 쌍을 독립적으로 이진 분류
  → 배치 내 모든 N² 쌍을 균등하게 처리

장점:
  배치 크기에 덜 의존
    CLIP: 큰 배치 필요 (분모에 많은 negative 필요)
    SigLIP: 작은 배치도 동작
  TPU-friendly (배치 across devices 불필요)
  분산 학습에서 배치를 GPU별로 분리 가능
  같거나 더 좋은 성능

주요 모델:
  SigLIP-400M: 400M params ViT-B/16
  SigLIP-SO400M: Shape-Optimized, VLM에 최적화
    → Gemini, PaliGemma, LLaVA-OneVision에서 사용
```

---

## DINOv2 (Meta, 2023)

```
Self-supervised vision pretraining:
  텍스트 없이 이미지만으로 표현 학습

두 가지 손실 결합:
  1. DINO (self-distillation with no labels):
     Student-Teacher 구조
     Teacher: student의 EMA (momentum 업데이트)
     다양한 augmentation으로 동일 이미지의 다른 뷰
     Student가 Teacher 출력 예측

  2. iBOT (Image BERT Pre-Training with Online Tokenizer):
     패치 마스킹 + masked prediction
     MAE와 유사하지만 self-supervised 토크나이저 사용

Curated data: 142M 이미지 (자동 필터링된 고품질)

특성:
  세밀한 visual feature (dense prediction 유리)
  Segmentation, depth estimation에 탁월
  Spatial feature 강함

vs CLIP:
  CLIP: 고수준 의미 이해 (분류, 검색, VQA)
  DINOv2: 저수준~중수준 visual feature (검출, 분할, 깊이)
  Cambrian-1: 두 encoder 앙상블로 상호 보완
```

---

## EVA-CLIP (BAAI, 2022)

```
CLIP loss + Masked Image Modeling (MIM) 결합

EVA (Exploratory Vision Autoencoder):
  1단계: MAE로 masked image reconstruction 사전학습
  2단계: CLIP loss로 vision-language alignment

EVA-CLIP-18B (2024):
  18B 파라미터 ViT
  데이터: 15B 이미지-텍스트 쌍
  역대 최대 규모 CLIP 모델

성능:
  Zero-shot 이미지 분류: 최고 수준
  InternVL에서 활용 (InternVL2의 backbone)
```

---

## InternViT (InternVL)

```
VLM을 위해 설계된 대형 Vision Encoder

InternViT-300M: 효율적 버전
  ViT-L 수준 성능, 더 작은 크기

InternViT-6B: 대형 버전
  6B 파라미터
  EVA-CLIP 기반 + VLM 전용 학습
  InternVL2의 핵심 구성요소

Dynamic Resolution:
  이미지를 여러 해상도로 분할하여 처리
  최대 12개 타일 (각 448×448)
  고해상도 이미지에서 세부 정보 보존

Pixel Shuffle:
  4개 인접 패치 → 1개 패치로 합성 (채널로)
  토큰 수 4× 감소 → LLM 처리 효율
  해상도 손실 없이 토큰 압축

  예: 448×448 이미지
    ViT 출력: 1024 토큰
    Pixel Shuffle: 256 토큰으로 압축
```

---

## Image Tokenization (생성 모델용)

### VQ-VAE (Vector Quantized VAE)

```
이미지를 이산 토큰으로 변환:

Encoder: 이미지 → 연속 latent z
Codebook: K개의 code vector
Quantizer: z → 가장 가까운 code 인덱스
Decoder: code 인덱스 → 이미지 재구성

손실:
  L = ||x - x̂||² +  ||sg[z] - e||² + β||z - sg[e]||²
  sg: stop gradient

활용:
  DALL-E 1: VQ-VAE + Transformer
  학습 후 이미지 = 토큰 시퀀스 → AR 생성 가능
```

### VQ-GAN

```
VQ-VAE + GAN 조합 (더 높은 품질):
  VQ-VAE: 구조적 품질
  PatchGAN Discriminator: 세부 사항 품질
  Perceptual loss: 지각적 유사성

코드북 크기: 16,384
더 다양한 이미지 표현 가능

VQVAE-2: 계층적 코드북 (거칠고 세밀한 코드)
```

### Chameleon (Meta, 2024)

```
이미지 토큰 + 텍스트 토큰을 동일 공간에서 처리
Early fusion 아키텍처

이미지 → VQ-VAE → 이산 토큰 (8192 vocab)
텍스트 → BPE → 이산 토큰 (32K vocab)
합쳐진 vocab으로 하나의 Transformer

장점:
  진정한 멀티모달 (이미지 생성 + 이해 동시)
  이미지와 텍스트 자유롭게 인터리빙

단점:
  이산 토큰화의 정보 손실
  복잡한 학습 파이프라인
```

---

## Vision Encoder 학습 비교

| Encoder | 학습 방식 | 강점 | 약점 |
|---------|---------|------|------|
| CLIP | 대조학습 (text-image) | 의미 이해, 분류 | spatial feature 약함 |
| SigLIP | Sigmoid 대조학습 | CLIP 개선, 효율 | - |
| DINOv2 | 자기지도 (SSL) | 세밀한 feature | 언어 정보 없음 |
| EVA-CLIP | CLIP + MAE | 의미+구조 | 대형 모델 |
| MAE | 마스킹 재구성 | 표현력 | 고수준 의미 약함 |
| InternViT | VLM 특화 | VLM 성능 최고 | proprietary |
| ConvNeXt | ConvNet 방식 | 귀납적 편향 | 긴 시퀀스 느림 |

---

## AnyRes / Dynamic Resolution 처리

```python
def select_best_resolution(image_size, possible_resolutions):
    """
    이미지 크기에 맞는 최적 그리드 선택
    예: possible_resolutions = [(336,336), (672,336), (336,672), ...]
    """
    original_width, original_height = image_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width = int(original_width * scale)
        downscaled_height = int(original_height * scale)
        effective_resolution = downscaled_width * downscaled_height
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or \
           (effective_resolution == max_effective_resolution and
            wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit

# LLaVA-Next/NeXT 스타일
# 이미지를 타일로 분할 + 전체 축소본 추가
# [thumbnail] + [tile_0_0, tile_0_1, tile_1_0, tile_1_1, ...]
```

---

## Vision Encoder 선택 기준

```
이미지 이해 (VQA, 캡션, 일반 대화):
  SigLIP-SO400M → 의미 이해 중심
  CLIP ViT-L/14 → 범용 선택

공간 추론 (OCR, 객체 검출, 위치):
  DINOv2 → spatial feature 중심
  InternViT → 복합적 이해

고해상도 처리 (문서, 차트):
  Dynamic resolution 지원 필수
  InternVL2, Qwen2-VL의 방식 참고

생성 모델 (이미지 생성):
  VQ-GAN, VQ-VAE 계열
  Stable Diffusion의 KL-VAE

멀티 인코더 앙상블:
  Cambrian-1: CLIP + DINOv2 + ConvNeXt
  → 다양한 특성 보완

파라미터 효율:
  SigLIP-400M: 성능/크기 균형
  ViT-L: 대형 모델 기준
  InternViT-6B: 최고 성능 (비용 높음)
```

---

## Further Questions

**Q. CLIP이 zero-shot 분류를 할 수 있는 이유?**
> 이미지-텍스트 임베딩이 같은 공간에 있어 유사도 비교 가능. "a photo of a [class]" 형식의 텍스트 프롬프트와 이미지를 비교. 클래스 레이블을 직접 학습하지 않아도 텍스트 유사도로 분류. 프롬프트 엔지니어링으로 성능 향상 가능 ("a photo of a [class], a type of [category]").

**Q. SigLIP이 CLIP보다 효율적인 이유?**
> CLIP의 softmax는 배치 전체를 분모로 → 큰 배치 필요 (많은 negative 필요). SigLIP은 독립적 sigmoid → 배치 크기 유연. 분산 학습에서 배치를 GPU별로 분리 가능 (all-gather 불필요). 같은 데이터로 더 나은 성능, 더 적은 메모리.

**Q. DINOv2를 VLM에서 추가로 사용하는 이유?**
> CLIP은 고수준 의미(분류, 검색)에 강하지만 spatial/dense feature 약함. DINOv2는 공간적 이해(위치, 깊이, 세그멘테이션)에 강함. Cambrian-1: 두 encoder의 특성이 상호보완적임을 실험으로 보임. 특히 "왼쪽/오른쪽", 거리 추정 등 공간 질문에서 DINOv2 feature 중요.

**Q. ViT의 위치 임베딩을 고해상도로 어떻게 확장하나?**
> 학습 시 196개 패치(224×224) → 추론 시 더 많은 패치. 방법 1: Bicubic interpolation (2D grid에서 위치 임베딩 보간). 방법 2: 2D RoPE (InternViT, Qwen2-VL). 방법 3: Dynamic resolution → 타일로 분할 (각 타일은 원래 해상도 유지). FlexiViT: 다양한 패치 크기로 학습.
