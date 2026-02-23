# VLM 아키텍처 & Multimodal Fusion

## VLM 기본 구조

```
이미지
  ↓ Vision Encoder (CLIP, SigLIP, ViT 등)
Visual Features (패치 임베딩)
  ↓ Connector / Projection
Visual Tokens (LLM 입력 공간으로 변환)
  ↓
LLM (Decoder-only: LLaMA, Qwen 등)
텍스트 + 이미지 통합 처리
  ↓
텍스트 생성

핵심 설계 결정:
  1. Vision Encoder 선택 (CLIP vs SigLIP vs DINOv2 등)
  2. Connector 방식 (Linear, MLP, Q-Former, Cross-attention)
  3. LLM 선택 (LLaMA, Qwen, Gemma, Mistral 등)
  4. 학습 전략 (몇 단계? 무엇을 frozen?)
  5. 해상도 처리 방식
```

---

## Connector 방법들 비교

### Linear Projection
```
Visual Features (N × D_v) → W (D_v × D_l) → Visual Tokens (N × D_l)

특징:
  - 단순, 빠름
  - LLaVA 초기 버전 사용
  - 낮은 파라미터 수

한계:
  - 선형 변환만 → 복잡한 modality 간 매핑 어려움
```

### MLP Projection (LLaVA-1.5+)
```
2-layer MLP:
  Linear(D_v → D_hidden) → GELU → Linear(D_hidden → D_l)
  D_hidden: 보통 D_v 또는 D_l

장점:
  - 비선형 매핑으로 더 풍부한 변환
  - LLaVA-1.5에서 Linear → MLP로만 바꿨는데 성능 크게 향상
```

### Q-Former (BLIP-2)
```
32개의 learnable query 토큰 (Q)
  각 query: 특정 시각 개념 담당

Cross-attention:
  Q가 Visual Features에 attend → 필요한 정보만 추출
  + Self-attention으로 query들 간 상호작용

특징:
  가변 길이 시각 입력 → 고정 32 토큰 출력
  LLM 처리 부담 감소

단점:
  32토큰으로 압축 → 세부 정보 손실
  고해상도 이미지 처리 어려움
```

### Perceiver Resampler (Flamingo)
```
N개의 latent query + Cross-attention
  latent queries (고정 크기) ← Image Features

매 Transformer 레이어마다:
  1. Self-attention (latent 간)
  2. Cross-attention (latent ← image features)

특징:
  가변 길이 입력 → 고정 크기 출력
  Gated Cross-attention으로 LLM 레이어 내부에 통합
```

### Cross-attention (LLM 레이어 내부)
```
Flamingo/Idefics 방식:
  일반 LLM self-attention + 시각 Cross-attention
  매 k번째 레이어마다 cross-attention 삽입

  text tokens → Self-Attention → + Cross-Attention(visual) → ...

  x_attn = tanh(α) × CrossAttn(x, visual_features) + x
  α: 학습 가능한 게이팅 파라미터 (초기 0 → 점진적 통합)

장점:
  텍스트가 직접 시각 정보 참조 → 정보 접근성 높음
  긴 시각 컨텍스트 처리 가능

단점:
  LLM 구조 변경 → 기존 LLM 직접 사용 불가
  추가 cross-attention 파라미터 필요
```

---

## 주요 VLM 아키텍처

### LLaVA 시리즈

#### LLaVA-1 (2023)
```
구조: CLIP ViT-L/14 → Linear Projection → LLaMA-7B/13B

2단계 학습:
  Stage 1 (Feature Alignment): CC3M 595K 이미지-텍스트
    - Projection Layer만 학습 (Vision Encoder, LLM frozen)
    - 단순 image-caption 매핑

  Stage 2 (Visual Instruction Tuning): 158K GPT-4 생성 데이터
    - 전체 파인튜닝 (or LLM frozen + projection tuning)
    - 다양한 시각 instruction 학습

핵심 기여:
  GPT-4로 생성한 시각 instruction 데이터
  → 고비용 인간 annotation 없이 고품질 VLM
```

#### LLaVA-1.5 (2023)
```
변경점:
  Linear → MLP Projection (2-layer)
  더 강한 LLM (Vicuna-13B)
  ShareGPT4V 등 더 많은/다양한 데이터

성능:
  비슷한 스케일에서 당시 최고 수준 달성
  → MLP projection의 중요성 입증

파라미터:
  CLIP ViT-L/14 (336px): 428M
  MLP Connector: ~2M
  Vicuna-13B: 13B
```

#### LLaVA-Next / LLaVA-1.6 (2024)
```
AnyRes (고해상도):
  이미지를 고정 해상도 타일로 분할
  Grid: {1×1, 1×2, 2×1, 2×2, ...} 중 비율 맞는 선택
  각 타일을 독립적으로 처리 후 concat

  [thumbnail 토큰] + [tile1 토큰] + <image_newline> + [tile2 토큰] ...

효과:
  336px → 1344px (2×2 grid) 해상도 향상
  OCR, 세부 이해 능력 향상

LLaVA-1.6-34B: 대형 LLM으로 성능 향상
```

#### LLaVA-OneVision (2024)
```
단일 모델로 이미지, 비디오, 멀티이미지 지원
LLM: Qwen2-7B/72B
Vision: SigLIP-SO400M (14px patch)

주요 특징:
  Single image → Multi-image → Video 순차 학습
  비디오: 프레임 샘플링 + 위치 인코딩

성능:
  이미지/비디오 벤치마크 모두 당시 오픈소스 최고
```

---

### BLIP / BLIP-2 / InstructBLIP

#### BLIP (2022)
```
Bootstrap Language-Image Pretraining

주요 기여: CapFilt
  1. 웹 데이터의 노이즈 캡션 → Captioner(이미지→캡션)로 새 캡션 생성
  2. Filter(텍스트→이미지 매칭)로 노이즈 필터링
  → 고품질 이미지-텍스트 쌍으로 자가 부트스트랩

손실:
  ITC (Image-Text Contrastive): CLIP 스타일 대조 학습
  ITM (Image-Text Matching): 이진 분류 (매칭 여부)
  LM (Language Modeling): 이미지 기반 캡션 생성
```

#### BLIP-2 (2023)
```
핵심 기여: Q-Former + 2단계 pretraining

구조:
  Frozen Vision Encoder (ViT-g/14 from EVA-CLIP)
    ↓ Q-Former (32 learnable queries)
      Stage 1: Vision-Language 표현 학습
        → ITC + ITM + ITG (Image to Text Generation)
      Stage 2: LLM과 연결
        → Q-Former 출력 → Linear → Frozen LLM
  Frozen LLM (OPT-2.7B/6.7B or FlanT5-XL/XXL)

비전의 역할 분리:
  Vision Encoder: 시각 특징 추출 (frozen)
  Q-Former: 시각-언어 인터페이스 역할
  LLM: 언어 이해/생성 (frozen)
  → 각 컴포넌트 독립적으로 발전 가능

효율성:
  32 query 토큰만 LLM에 전달 → LLM 처리 효율적
  대형 Vision Encoder + LLM frozen → 학습 비용 절감
```

#### InstructBLIP (2023)
```
BLIP-2 + Instruction Tuning

Q-Former에 instruction 통합:
  Q-Former의 cross-attention 시 instruction도 참조
  → instruction에 맞는 시각 정보 선택적 추출

  예: "이 사진에서 글자를 읽어봐" → OCR 관련 feature 추출
  예: "이 그림의 분위기는?" → 감정/분위기 feature 추출

13개 시각 언어 데이터셋으로 instruction tuning
```

---

### Flamingo (DeepMind, 2022)

```
Large-scale vision-language pretraining
Chinchilla-level LLM + Cross-attention 통합

핵심 아이디어:
  1. 강력한 LLM (Chinchilla)을 frozen으로 유지
  2. Vision 정보를 Cross-attention으로 주입
  3. Gated cross-attention: tanh(α) 게이팅

  매 FF 레이어 앞에 GATED XATTN-DENSE 삽입:
  y = tanh(α_g) · GATED_XATTN(x, visual) + x

Perceiver Resampler:
  가변 프레임(비디오) → 고정 64 토큰
  시간적 시각 정보 효율적 압축

학습 데이터:
  MultiModal MassiveWeb (M3W): 웹의 인터리브드 이미지-텍스트
  ALIGN, LTIP: 이미지-텍스트 쌍

In-context learning:
  Few-shot: 몇 개 이미지-질문-답 쌍 제공 후 새 질문
  → VLM의 in-context learning 첫 시연

크기: Flamingo-3B, 9B, 80B
```

---

### PaliGemma (Google, 2024)

```
SigLIP-So400M + Gemma-2B/9B/27B 조합

구조:
  이미지 → SigLIP-So400M → 이미지 토큰
  텍스트 → Gemma 토크나이저
  → Gemma가 이미지+텍스트 모두 처리

특징:
  Full attention (이미지 토큰 ↔ 텍스트 토큰)
  이미지 토큰에 prefix LM attention (서로 attend 가능)
  텍스트 토큰은 이미지에 attend, 이미지는 텍스트 attend 불가

학습 3단계:
  1. Unimodal pretraining (별도로)
  2. Multimodal pretraining:
     LAION, WebLI 등 대규모 데이터
  3. Task transfer:
     다양한 시각 태스크 파인튜닝 mix

해상도:
  224px, 448px, 896px 지원
  448px: 1568 이미지 토큰 (기본)

특이점:
  "Transfer" 모델: 다양한 downstream 태스크 전이 최적화
  단순 구조 + 고품질 학습이 핵심
```

---

### InternVL 시리즈 (Shanghai AI Lab, 2023-)

```
InternVL-1 (2023):
  InternViT-6B: VLM을 위해 설계된 대형 Vision Encoder
  EVA-CLIP 기반, VLM에 맞게 추가 학습
  Cascaded architecture: InternViT → MLP → LLM

InternVL-1.5 / InternVL-2 (2024):
  Dynamic Resolution:
    이미지를 최대 12개 타일로 동적 분할
    각 타일: 448×448 → 256 토큰
    전체 썸네일: 1 타일 (개요)
    → 최대 3072 시각 토큰

  Pixel Shuffle (토큰 압축):
    인접 2×2 패치를 하나의 토큰으로 합침
    채널 차원 확장: D → 4D
    → 토큰 수 4× 감소
    → 더 긴 시퀀스 처리 가능

  모델 크기 다양성:
    InternVL2-2B: MiniInternVL (엣지 배포)
    InternVL2-8B: 균형
    InternVL2-40B: 고성능
    InternVL2-76B: 최고 성능

InternLM-XComposer (InternLM + InternViT):
  중국어 강함, 문서 이해 특화
```

---

### Qwen-VL 시리즈 (Alibaba, 2023-)

```
Qwen-VL (2023):
  Vision Encoder: ViT-bigG (1.9B params)
  Connector: Cross-attention (Q-Former 스타일)
  LLM: Qwen-7B
  3단계 학습: 이미지-텍스트 사전학습 → SFT → RLHF

  특징:
    256×256 원본 해상도 학습
    위치 인식: Bounding box 생성 능력
    다국어 (중국어/영어) 강함

Qwen2-VL (2024):
  Naive Dynamic Resolution:
    어떤 해상도든 모두 처리 (AnyRes 확장)
    최소/최대 픽셀 설정만

  M-ROPE (Multimodal Rotary Position Embedding):
    텍스트: 1D RoPE (시퀀스 위치)
    이미지: 2D RoPE (height_id, width_id)
    비디오: 3D RoPE (time_id, height_id, width_id)
    → 공간/시간 정보를 위치 인코딩에 통합

  비디오 이해:
    1분, 20분 비디오까지 지원
    각 프레임에 time_id 할당

  크기: Qwen2-VL-2B, 7B, 72B
```

---

### Phi-3.5-Vision (Microsoft, 2024)

```
경량 고성능 VLM:
  Visual Encoder: CLIP ViT-L/14
  LLM: Phi-3.5-mini (3.8B)
  Connector: HD-Transform (고해상도 처리)

HD-Transform:
  448×448 기준 타일 분할
  Global + Local 처리:
    Global: 전체 이미지 low-res 개요
    Local: 각 타일 high-res 세부

교육 데이터:
  고품질 합성 데이터 중심 (Phi 시리즈 특성)
  STEM, OCR, 다이어그램 특화

특징:
  소형(3.8B)이지만 7B 이상 경쟁
  엣지 배포에 최적
```

---

### Pixtral (Mistral AI, 2024)

```
Pixtral-12B:
  Vision Encoder: Pixtral-ViT (400M)
    - 새로운 ViT 아키텍처 (Mistral 자체 개발)
    - 고해상도 natively 지원
  LLM: Mistral-Nemo-12B

특징:
  Native resolution: 임의 해상도 이미지 처리
  이미지 내 행 구분: [IMG_BREAK] 토큰
  이미지 끝: [IMG_END] 토큰

  다중 이미지:
    여러 이미지를 컨텍스트에 포함 가능
    이미지-텍스트 인터리빙

  Long context: 128K 토큰 컨텍스트
```

---

### LLaMA-3.2 Vision (Meta, 2024)

```
크기: 11B, 90B

구조:
  Vision Encoder: 별도 cross-attention 방식 (Flamingo 스타일)
  LLM: LLaMA-3.1 (frozen)
  Cross-attention layers: 매 4번째 LLM 레이어

특징:
  기존 LLaMA-3.1 Instruct 능력 유지 (text frozen)
  이미지 처리만 추가학습
  16개 타일 지원 (고해상도)

학습:
  Stage 1: Cross-attention layers만 학습
  Stage 2: Cross-attention + vision encoder 파인튜닝
```

---

### Cambrian-1 (NYU, 2024)

```
핵심 기여: 다중 Vision Encoder 앙상블

SVA (Spatial Vision Aggregator):
  여러 Vision Encoder 출력을 통합
    CLIP: 의미 이해
    DINOv2: 공간 정보
    ConvNeXt: 지역적 특징
    → 다양한 시각 표현 결합

  Spatial-aware Q-Former:
    가변 개수의 spatial grid 유지
    각 grid cell이 해당 위치 시각 정보 담당

  → 공간 추론, 객체 검출 벤치마크에서 강함

LLM: LLaMA-3-8B/34B
학습 데이터: Cambrian-10M (10M 큐레이션 데이터)
```

---

## 고해상도 처리 전략

### AnyRes / Dynamic Tiling (LLaVA-Next, InternVL)
```
이미지를 고정 해상도 타일로 분할:
  입력: 임의 크기 이미지
  처리:
    1. 이미지 비율에 맞는 grid 선택 (1×1, 2×1, 2×2, ...)
    2. 각 타일을 독립적으로 Vision Encoder 처리
    3. 전체 썸네일 (global context)
    4. concat + <image_newline> 토큰으로 위치 표시

토큰 수 계산:
  타일당 256 토큰 (14×14 patches, 16px → 256)
  2×2 grid: 4 tiles × 256 + 1 thumbnail × 256 = 1280 tokens
  최대 2×2: LLaVA-Next
  최대 3×4: InternVL2 (최대 3072 tokens)
```

### Pixel Shuffle (InternVL)
```
인접 2×2 패치를 하나로 병합:
  [p1, p2, p3, p4] (각 D차원) → [p1||p2||p3||p4] (4D차원)
  → 토큰 수 4× 감소, 차원 4× 증가

후처리 Linear:
  4D → D (차원 복원)

효과:
  1024 tokens → 256 tokens (2×2 pixel shuffle)
  LLM 처리 속도 4× 향상
  성능 손실 최소 (인접 정보 보존)
```

### Naive Dynamic Resolution (Qwen2-VL)
```
패치 수를 이미지 내용에 맞게 동적 결정:
  min_pixels, max_pixels로 범위 설정
  이미지 크기에 비례한 토큰 수

  예:
    224×224: 256 tokens
    448×448: 1024 tokens
    1024×1024: ~5376 tokens

장점:
  작은 이미지 → 적은 토큰 (효율)
  큰 이미지 → 많은 토큰 (품질)
  해상도에 따른 적응적 처리
```

---

## 비디오 이해

```
방법 1: 프레임 샘플링
  비디오 → 균등 간격 N개 프레임 추출 → 각 이미지로 처리
  단순, 빠름
  단점: 시간 정보 손실, 긴 비디오 어려움

방법 2: 시간적 위치 인코딩
  Qwen2-VL M-ROPE:
    각 프레임에 time_id 할당
    → 시간 순서 인식

  InternVL-2: 비디오 프레임을 이미지 시퀀스로 처리 후
             frame 위치 태그 추가

방법 3: 전용 시간 인코딩
  Video-LLaVA: visual 특징에 temporal embedding 추가
  VideoChat: 시간 attention mask

방법 4: 긴 컨텍스트 활용
  Gemini 1.5: 1M 토큰 컨텍스트로 전체 비디오 처리
  LongVA: LLaMA + 긴 컨텍스트로 긴 비디오

실용적 선택:
  짧은 비디오 (<1분): 프레임 샘플링 (4-8 fps)
  긴 비디오: 프레임 압축 + 긴 컨텍스트
  실시간: 경량 모델 + 낮은 해상도
```

---

## 모델 비교

| 모델 | Vision Encoder | Connector | LLM | 특징 |
|------|---------------|-----------|-----|------|
| LLaVA-1.5 | CLIP ViT-L | MLP | Vicuna-13B | 단순, 강력 |
| BLIP-2 | EVA-CLIP ViT-g | Q-Former | OPT/Flan-T5 | 압축 효율 |
| Flamingo | NFNet | Perceiver | Chinchilla | 인터리브드 |
| InternVL2 | InternViT-6B | Pixel Shuffle | InternLM2 | 다국어, 고해상도 |
| Qwen2-VL | 자체 ViT | Dynamic | Qwen2 | M-ROPE, 비디오 |
| PaliGemma | SigLIP-So400M | Full Attention | Gemma | Transfer 최적 |
| Phi-3.5-V | CLIP ViT-L | HD-Transform | Phi-3.5 | 경량, 고성능 |
| LLaMA-3.2-V | 별도 ViT | Cross-Attention | LLaMA-3.2 | Text 능력 보존 |
| GPT-4V/4o | 미공개 | 미공개 | GPT-4 | SOTA |
| Gemini 1.5 | 미공개 | 미공개 | Gemini | 1M 컨텍스트 |

---

## Further Questions

**Q. LLaVA의 2단계 학습 이유는?**
> Stage 1: Vision Encoder와 LLM 임베딩 공간 정렬 (alignment). 시각 특징을 LLM이 이해할 수 있는 토큰으로 변환하는 projection 학습. Stage 2: 실제 지시 따르기 학습 (instruction tuning). 처음부터 instruction tuning하면 정렬 안 된 시각 토큰으로 불안정.

**Q. Q-Former vs MLP Projection 어느 것이 좋은가?**
> 일반적으로 MLP가 최근 더 선호됨. Q-Former는 고정 32토큰으로 압축 → 세부 정보 손실. MLP는 모든 패치 토큰 유지 → 정보 보존. 하지만 MLP는 더 많은 LLM 토큰 → 긴 시퀀스 처리 비용 증가. 고해상도: Q-Former 토큰 절약 유리, 하지만 성능 손실.

**Q. 고해상도 이미지 처리가 왜 중요한가?**
> OCR: 작은 글자 읽기, 문서 이해. 차트/다이어그램: 세밀한 시각 정보. 물체 검출: 작은 물체 인식. LLaVA-1.5 이후 고해상도 지원이 VLM 성능의 핵심. 기본 336px에서 1344px(2×2 tiling)으로 OCR 정확도 2배 이상 향상.

**Q. Flamingo와 LLaVA의 근본적 차이는?**
> 아키텍처: Flamingo는 LLM 레이어 내부에 Cross-attention 삽입 (LLM 구조 변경 필요). LLaVA는 Vision 토큰을 텍스트 토큰처럼 LLM 입력으로 제공 (LLM 구조 그대로). 사용성: LLaVA 방식이 구현 단순, 최신 LLM 바로 사용 가능. Flamingo 방식은 강력한 시각 통합 but 새 LLM으로 교체 어려움.

**Q. 비디오 VLM에서 가장 큰 도전은?**
> 시간적 이해: 단순 프레임 나열로는 "무슨 일이 벌어지는가" 파악 어려움. 토큰 폭발: 30fps×10분 = 18000 프레임 → 수백만 토큰. 시간적 위치: 어느 시점에 무슨 일인지 인식 필요. 해결: 프레임 샘플링 + temporal positional encoding (M-ROPE) + 긴 컨텍스트 (Gemini 1.5).
