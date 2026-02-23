# 필독 논문 리스트

## 기초 (Foundation)

### Attention is All You Need (2017)
```
핵심 기여: Transformer 아키텍처
핵심 아이디어:
  - Self-Attention: Q, K, V 행렬
  - Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V
  - Multi-Head Attention: h개 헤드 병렬
  - Position Encoding: sin/cos 함수
  - 인코더-디코더 구조
  - RNN/LSTM 없이 시퀀스 처리

이해해야 할 것:
  - Self-Attention 복잡도: O(n²d)
  - Cross-Attention vs Self-Attention 차이
  - Position Encoding의 필요성
  - Residual Connection + LayerNorm의 역할
```

### GPT-3 (2020) - Language Models are Few-Shot Learners
```
핵심 기여: 대규모 스케일링 + In-context Learning
핵심 아이디어:
  - 1750억 파라미터 (당시 최대)
  - Few-shot: 예시 몇 개만으로 새 태스크 해결
  - One-shot, Zero-shot 성능 입증
  - 사전학습만으로 다양한 NLP 태스크 해결

핵심 통찰:
  - 모델 크기가 커질수록 In-context Learning 능력 급상승
  - Fine-tuning 없이도 강력한 성능 (Prompt Engineering)
  - Emergent Abilities: 특정 규모에서 갑자기 새 능력 등장
```

### Scaling Laws for Neural Language Models (Kaplan et al., 2020)
```
핵심 기여: 스케일링 법칙 정량화

핵심 공식:
  L(N) = (Nc/N)^αN      # 파라미터 수에 따른 손실
  L(C) = (Cc/C)^αC      # 연산량에 따른 손실
  L(D) = (Dc/D)^αD      # 데이터 크기에 따른 손실
  αN ≈ αD ≈ αC ≈ 0.076

핵심 결론:
  - 고정 연산량 하에서 모델 크기와 데이터 크기의 최적 비율 존재
  - 파라미터 > 데이터 증가 권장 (당시 권장)
  - 아키텍처 선택보다 규모가 더 중요
  - (후에 Chinchilla가 데이터 중요성 재발견)
```

### Chinchilla (2022) - Training Compute-Optimal LLMs
```
핵심 기여: Compute-Optimal 학습 비율 재정립

핵심 공식:
  N_opt = C^0.49 / 1.7  # 최적 파라미터 수
  D_opt = C^0.51 / 7.7  # 최적 데이터 크기
  → N과 D가 거의 동등하게 중요

Kaplan과 차이:
  Kaplan: 고정 데이터에서 모델 크기 키우기 권장
  Chinchilla: 같은 연산량에서 모델↓, 데이터↑ (균형)

결과:
  Gopher 280B와 동일 연산량으로 Chinchilla 70B 학습
  → Chinchilla 70B가 모든 벤치마크에서 Gopher 280B 능가

실용적 의미:
  - 추론 비용 고려: 더 작고 데이터 많이 학습한 모델이 유리
  - LLaMA: Chinchilla보다 더 많은 데이터 (추론 최적화)
  - "Inference-optimal" vs "Training-optimal" 구분
```

---

## 아키텍처 개선

### Flash Attention (Dao et al., 2022)
```
핵심 기여: IO-Aware 메모리 효율 Attention

문제: 표준 Attention의 O(n²) HBM 접근
  - Softmax: 전체 QKᵀ 행렬 HBM에 써야 함
  - HBM ↔ SRAM 병목

해결:
  1. 타일링(Tiling): SRAM에 들어가는 블록으로 분할
  2. 온라인 소프트맥스: 전체 행 보지 않고 점진적 계산
     m_new = max(m_old, max(x_new))
     l_new = l_old × exp(m_old - m_new) + exp(x_new - m_new)
  3. Recomputation: 역전파 시 HBM에서 재계산 (저장 안 함)

결과:
  - 메모리: O(n²) → O(n) (Intermediate 저장 없음)
  - 속도: 2-4× 빠름 (HBM 접근 대폭 감소)
  - 긴 시퀀스 가능 (이전에 OOM이던 4K+ 처리 가능)

Flash Attention 2 개선:
  - Work partitioning 개선 (Q 병렬화)
  - Warp간 synchronization 최소화
  - 2배 추가 속도 향상

Flash Attention 3 (H100):
  - FP8 지원
  - Async pipeline (FP16/FP8 혼합)
  - 실제 하드웨어 최대 성능의 ~75%
```

### RoPE (Rotary Position Embedding, 2021)
```
핵심 기여: 상대적 위치 정보를 회전 행렬로 인코딩

아이디어:
  위치 m의 쿼리, 위치 n의 키에 대해:
  qₘᵀkₙ = (Rₘq)ᵀ(Rₙk) = qᵀRₘᵀRₙk = qᵀR(n-m)k

  → attention score가 상대 위치 (n-m)에만 의존

구현:
  θₖ = 10000^(-2k/d), k = 0,1,...,d/2
  q'₂ₖ   = q₂ₖ·cos(mθₖ) - q₂ₖ₊₁·sin(mθₖ)
  q'₂ₖ₊₁ = q₂ₖ·sin(mθₖ) + q₂ₖ₊₁·cos(mθₖ)

장점:
  - 학습 시 보지 못한 위치로 외삽 (extrapolation)
  - 상대 위치 자연스럽게 인코딩
  - LLaMA, Mistral, Qwen 등 거의 모든 현대 LLM 채택

컨텍스트 길이 확장 (YaRN):
  λ(r) = {1                    if r < α
          {1/scale(r)           if r ≥ α
  → 저주파 성분 조정으로 긴 컨텍스트 외삽
```

### GQA (Grouped-Query Attention, 2023)
```
핵심 기여: KV Cache 크기 축소 (추론 효율화)

MHA (Multi-Head Attention):
  h개 Query Head, h개 KV Head → 중복 많음
  KV Cache: O(h × seq_len × head_dim)

MQA (Multi-Query Attention):
  h개 Query, 1개 KV 공유 → 너무 큰 품질 손실

GQA (Grouped-Query Attention):
  h개 Query를 G개 그룹, 그룹당 1개 KV 공유
  KV Cache: O(G × seq_len × head_dim), G << h

예시 (LLaMA-3-8B):
  h=32 Query Heads, G=8 KV Heads
  KV Cache 크기: 1/4로 감소
  품질: MHA와 거의 동일

효과:
  - 같은 배치 크기에 4× 긴 시퀀스 처리
  - 또는 4× 더 큰 배치 크기 허용
  - 처리량 대폭 향상
```

### DeepSeek-V2 (2024) - MLA + DeepSeekMoE
```
MLA (Multi-head Latent Attention):
  기존 KV Cache: num_heads × head_dim × 2 (K+V)
  MLA: 저차원 잠재 벡터 c_kv로 압축 후 KV 복원
    c_kv = W_DKV · h (압축: 고차원 → 저차원)
    K_C, V_C = W_UK · c_kv, W_UV · c_kv (복원)

  효과: KV Cache를 최대 93.3% 절감

DeepSeekMoE:
  - Fine-grained Expert: 더 작고 많은 전문가 (2B → 256M)
  - Shared Expert: 항상 활성화되는 공유 전문가
  h(x) = Expert_shared(x) + Σₖ G(x)ₖ · Expert_routed_k(x)
```

### Mamba (2023) - Selective SSM
```
핵심 기여: Linear 복잡도 시퀀스 모델

기존 SSM (S4):
  h(t) = Ah(t-1) + Bx(t)
  y(t) = Ch(t)
  A, B, C가 시간 불변 → 콘텐츠 의존 불가

Mamba 개선:
  A, B, C를 입력 x(t)의 함수로 만듦
  B(t) = s_B(x(t)), C(t) = s_C(x(t))
  → Selective: 중요한 정보만 상태에 저장

특징:
  - O(n) 추론 복잡도 (vs Transformer O(n²))
  - 병렬 스캔으로 학습 가능
  - 긴 시퀀스에서 Transformer와 유사한 성능
  - 하지만 In-context Learning은 Transformer보다 약함
  - Jamba: Mamba + Transformer 하이브리드 (2024)
```

---

## 학습 & 정렬

### InstructGPT (2022) - RLHF 실용화
```
핵심 기여: RLHF 파이프라인 최초 실용화

3단계:
  1. SFT: 고품질 시연 데이터로 지도 학습
  2. RM 학습: 인간 비교 피드백으로 보상 모델
     r(x,y) = Bradley-Terry 모델로 선호도 학습
  3. PPO: RM 피드백으로 정책 최적화
     max_π E[r(x,y)] - β·KL(π||π_ref)

결과:
  - GPT-3 175B보다 1.3B InstructGPT 선호
  - Alignment Tax: 일부 성능 저하 발생
  - RLHF의 실용성 최초 입증
```

### DPO (Direct Preference Optimization, 2023)
```
핵심 기여: PPO 없이 선호 학습

RLHF의 문제: RM 별도 학습, PPO 불안정

DPO 핵심 수식:
  L_DPO = -E[log σ(β·log(π_θ(y_w|x)/π_ref(y_w|x))
                  - β·log(π_θ(y_l|x)/π_ref(y_l|x)))]

  y_w: 선호 응답, y_l: 비선호 응답
  β: 정책이 참조 모델에서 멀어지는 정도 제어

핵심 통찰:
  - 최적 정책과 보상 모델 간의 닫힌 형태 관계 발견
  - RM 없이 선호 데이터에서 직접 최적화
  - 더 안정적, 구현 간단

파생 연구:
  - SimPO (2024): Reference-free, 길이 정규화
  - ORPO (2024): SFT + 정렬 동시 학습
  - IPO (2024): DPO 과적합 문제 해결
```

### QLoRA (2023)
```
핵심 기여: 4비트 양자화 + LoRA 조합

기술:
  1. NF4 (Normal Float 4): 정규분포 가정한 4비트 양자화
     최적 분위점 사용 → 4비트 중 최고 품질
  2. Double Quantization: 양자화 상수도 다시 양자화
     추가 ~0.37 bits/param 절약
  3. Paged Optimizer: CPU offloading으로 OOM 방지

결과:
  - 65B 모델을 48GB GPU 1개에서 학습 가능
  - BF16 SFT와 동등한 성능
  - 4비트 저장 → 16비트로 계산 (BnB 자동 변환)

사용:
  from peft import prepare_model_for_kbit_training
  model = AutoModelForCausalLM.from_pretrained(
      model_id, load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
  )
  model = prepare_model_for_kbit_training(model)
```

### GRPO (DeepSeekMath, 2024)
```
핵심 기여: PPO 대체 그룹 상대 정책 최적화

PPO 문제: 별도 Critic 모델 필요, 불안정

GRPO:
  - 각 질문에 G개 응답 생성 → 검증
  - 그룹 내 상대적 장점으로 보상 정규화
  - Critic 불필요 (그룹 평균이 베이스라인)

L_GRPO = -E[Σᵢ min(rᵢAᵢ, clip(rᵢ,1-ε,1+ε)Aᵢ)] + β·KL

  Aᵢ = (rᵢ - mean(r_group)) / std(r_group)

DeepSeek-R1 훈련:
  1. Cold Start: 소량 추론 데이터 SFT
  2. GRPO with Verifiable Rewards
  3. Rejection Sampling (고품질 필터링)
  4. 최종 GRPO + 일반 데이터 혼합
```

### Constitutional AI (Anthropic, 2022)
```
핵심 기여: AI 스스로 원칙 기반 자기 개선

두 단계:
  1. SL-CAI (Supervised Learning from AI Feedback):
     - 해로운 응답 생성 → "원칙에 따라 수정" → 수정 응답 학습
     - 원칙 예시: "무해하고, 정직하며, 도움이 되어야 함"

  2. RLAIF (RL from AI Feedback):
     - AI가 두 응답 중 원칙에 더 맞는 것 선택
     - 사람 없이 선호 데이터 생성 → DPO/PPO 학습

장점:
  - 사람 레이블러 의존성 감소
  - 원칙이 명시적 → 투명성
  - 스케일 가능
```

---

## 추론 최적화

### vLLM / PagedAttention (2023)
```
핵심 기여: 가상 메모리 방식 KV Cache 관리

문제: 기존 KV Cache 할당
  - 최대 시퀀스 길이로 사전 할당 → 낭비
  - 요청마다 연속 메모리 필요 → 단편화
  - 메모리 이용률 60-80%

PagedAttention:
  - KV Cache를 고정 크기 Page로 분할 (예: 16 토큰)
  - Block Table로 비연속 Page 관리
  - Copy-on-Write: 병렬 샘플링 시 Page 공유

결과:
  - 메모리 이용률 ~100%
  - 처리량 2-4× 향상
  - 동시 서빙 요청 수 대폭 증가
```

### Speculative Decoding (2022)
```
핵심 기여: LLM 추론 속도 향상 (품질 유지)

핵심 아이디어:
  1. 작은 Draft 모델로 K개 토큰 빠르게 생성
  2. 큰 Target 모델로 K개 토큰 한 번에 검증
  3. Rejection Sampling으로 분포 유지

수학:
  Draft 분포 q(x|ctx), Target 분포 p(x|ctx)
  수락 확률: min(1, p(x)/q(x))
  거부 시: Target 분포의 수정 버전에서 재샘플링

  결과: Target 모델과 동일한 분포 보장 (lossless)

속도 향상:
  드래프트 토큰 대부분 수락 시: K배 빠름
  초안 모델이 타겟과 유사할수록 효과적

변형:
  - Medusa: Draft head 여러 개 병렬 사용
  - EAGLE: 동일 모델의 Early Exit으로 Draft
  - SpecTr: Tree 기반 Speculative
```

### GPTQ (2022) / AWQ (2023)
```
GPTQ (Post-Training Quantization):
  OBC(Optimal Brain Compression) 기반
  가중치 w_q를 양자화할 때, 보상하도록 다른 가중치 조정
  H⁻¹ (역 헤세 행렬)으로 중요도 측정
  4비트로 GPT-3.5 수준 품질 유지

AWQ (Activation-aware Weight Quantization):
  핵심 관찰: 가중치 일부가 훨씬 중요
  활성화 크기로 중요 가중치 식별
  중요 가중치 채널 스케일링 후 양자화
  → 정보 손실 최소화

  결과: GPTQ보다 품질 우수 (특히 지시따르기 모델)

SmoothQuant (W8A8):
  활성화 이상치 가중치로 이전 → W와 A 모두 양자화 쉬워짐
  smoothing: W' = W × diag(s), X' = X × diag(s)⁻¹
```

---

## Reasoning & Agents

### Chain-of-Thought (Wei et al., 2022)
```
핵심 기여: 단계별 추론 프롬프팅

아이디어:
  기존: Q → A (직접 답변)
  CoT: Q → [중간 추론 단계] → A

예시:
  "Roger has 5 tennis balls. He buys 2 more cans.
   Each can has 3 balls. How many balls does he have?"

  CoT: "Roger started with 5 balls.
        2 cans × 3 balls = 6 more balls.
        5 + 6 = 11 balls. The answer is 11."

효과:
  - 수학 문제: 어시스턴트 Turbo급에서 두드러짐
  - Zero-shot CoT: "Let's think step by step"만 추가
  - Emergent: 소규모 모델에선 효과 없음 (>100B부터)
```

### Self-Consistency (Wang et al., 2022)
```
핵심 기여: 다수결 추론

아이디어:
  - 동일 질문에 CoT로 여러 추론 경로 생성
  - 각 경로의 최종 답변 수집
  - 다수결로 최종 답변 결정

효과:
  - 단일 CoT보다 10-20% 향상
  - 비용: N배 추론 (N=5-10)
  - 이후 MCTS 기반 방법의 원형
```

### Tree of Thoughts (Yao et al., 2023)
```
핵심 기여: 체계적 추론 탐색 (BFS/DFS)

아이디어:
  - 추론을 트리로 모델링 (Thought Tree)
  - 각 노드: 중간 추론 상태
  - 상태 평가기(Evaluator): 각 상태 점수화
  - BFS/DFS로 유망한 경로 탐색

구현:
  thought_tree = TreeOfThoughts(llm, evaluator)
  answer = thought_tree.solve(problem, method="BFS", n_select=5)

vs Self-Consistency: 더 체계적, 역추적 가능
vs CoT: 잘못된 추론 경로 중간에 수정 가능
```

### DeepSeek-R1 (2025)
```
핵심 기여: 오픈소스 RL 기반 강력한 추론 모델

학습 파이프라인:
  1. Cold Start: 소량(수천 개) 고품질 추론 데이터 SFT
  2. 대규모 GRPO:
     - 검증 가능한 보상 (수학, 코드 정확성)
     - 형식 보상 (Chain-of-Thought 사용 보너스)
     - 언어 일관성 보상
  3. Rejection Sampling: 위 모델로 고품질 데이터 생성
  4. 최종 SFT + GRPO: 추론 + 일반 능력 통합

Emergent 능력:
  - Self-reflection: 스스로 오류 인식 후 수정
  - Aha Moment: 문제 접근법 갑자기 변경
  - 긴 CoT: 수십~수백 단계 추론

결과:
  - 수학/코딩: GPT-4o 수준 (오픈소스 최강)
  - 蒸溜(Distillation): R1-7B, R1-14B 등 소형 버전 공개
```

---

## VLM

### ViT (Vision Transformer, 2020)
```
핵심 기여: 이미지를 패치 시퀀스로 처리

아이디어:
  - 이미지를 16×16 패치로 분할 (예: 224→14×14=196 패치)
  - 각 패치를 Linear Projection → 토큰
  - [CLS] 토큰 추가 → Transformer Encoder
  - 분류: [CLS] 토큰의 최종 표현 사용

공식:
  z₀ = [x_class; E×p₁; E×p₂; ...] + E_pos
  zₗ = Transformer(z_{l-1})
  ŷ = MLP(z_L[0])  # [CLS] 토큰

특징:
  - CNN보다 대용량 데이터에서 우수
  - 전이학습 효율적
  - LLM 기반 VLM의 비전 인코더로 채택
```

### CLIP (2021) - Learning Transferable Visual Models
```
핵심 기여: 이미지-텍스트 대조 학습

학습:
  - 4억 (이미지, 텍스트) 쌍 수집
  - Image Encoder + Text Encoder
  - InfoNCE 손실:
    L = -log(exp(sim(i,t)/τ) / Σⱼ exp(sim(i,tⱼ)/τ))

Zero-shot:
  - "a photo of {class}" 프롬프트로 모든 분류 가능
  - 학습 없이 1000개 ImageNet 클래스 분류: 76.2%

적용:
  - VLM 비전 인코더 (LLaVA, InstructBLIP)
  - 이미지 검색
  - DALL-E 2, Stable Diffusion의 텍스트 인코더
```

### LLaVA (2023) - Visual Instruction Tuning
```
핵심 기여: 시각 instruction 따르기 능력

구조:
  CLIP ViT-L/14 → Linear Projection → LLaMA-7B
  (MLP projection으로 LLaVA-1.5에서 개선)

데이터:
  GPT-4로 (이미지, 설명) → 158K instruction pairs 생성
  대화, 상세설명, 추론 3가지 유형

훈련:
  Stage 1: Projection만 학습 (Visual Feature Alignment)
  Stage 2: Projection + LLM 전체 Fine-tuning

결과:
  - 최초로 시각 instruction tuning 체계 확립
  - GPT-4V 대비 85% 성능 (LLaVA-1.5)
  - 이후 InstructBLIP, Qwen-VL, InternVL 등의 원형
```

### Qwen2-VL (2024)
```
핵심 기여: M-RoPE, 다해상도, 비디오 이해

M-RoPE (Multimodal RoPE):
  텍스트: 1D 위치 정보
  이미지: 2D 위치 정보 (행, 열)
  비디오: 3D 위치 정보 (행, 열, 시간)

Dynamic Resolution:
  임의 해상도 이미지를 VLM이 직접 처리
  해상도별 다른 패치 수 → 패킹으로 배치 처리

비디오 이해:
  Time Compression: 연속 프레임 특징 합산
  동적 FPS 샘플링 (내용에 따라 조정)

성능:
  DocVQA: 96.4% (최고 수준)
  Video 이해: GPT-4o 능가
```

---

## RAG & 지식

### RAG (Lewis et al., 2020)
```
핵심 기여: 검색 기반 생성 (최초 체계화)

구조:
  Parametric Memory: LLM의 가중치
  Non-parametric Memory: 외부 문서 검색

두 변형:
  RAG-Sequence: 전체 답변에 하나의 문서
  RAG-Token: 각 토큰마다 다른 문서 가능

학습:
  p(y|x) = Σ_z p(z|x) × p(y|x,z)
  Retriever (DPR) + Generator (BART) 함께 학습

현대적 RAG와 차이:
  - 현재는 고정 임베딩 모델 + 별도 LLM 주로 사용
  - Dense Passage Retrieval (DPR) 방식
```

### Self-RAG (2023)
```
핵심 기여: 선택적 검색 + 자기 반성

특수 토큰 (Reflection Tokens):
  [Retrieve]: 검색 필요 여부 결정
  [IsRel]: 검색 결과 관련성 평가
  [IsSup]: 생성 내용이 문서로 지지되는가
  [IsUse]: 전체 응답 유용성 평가

학습:
  - 특수 토큰을 데이터에 삽입하여 SFT
  - 검색이 필요없는 질문: 직접 답변
  - 검색이 필요한 질문: 검색 후 답변

효과:
  - 모든 쿼리에 검색하는 naive RAG보다 효율적
  - 자기 비판으로 품질 향상
```

---

## 긴 컨텍스트

### YaRN (2023) - Yet Another RoPE Extension
```
핵심 기여: RoPE 외삽의 개선된 방법

NTK-aware 스케일링의 문제:
  고주파 성분만 스케일링 → 저주파도 조정 필요

YaRN:
  λ(r) = {1           if r < α (고주파, 변경 없음)
          {ramp(r)     if α ≤ r ≤ β (전환 구간)
          {1/scale     if r > β (저주파, 스케일링)

  Temperature 조정: 소프트맥스 온도도 함께 수정
    softmax(qkᵀ/√d × 1/t)  # t: 외삽 비율 기반

결과:
  - Code Llama: 4K → 100K 컨텍스트 확장
  - 미세조정 없이 0.1% 데이터로 추가 적응
  - Perplexity: 외삽 범위에서도 안정적
```

### Lost in the Middle (2023)
```
핵심 기여: LLM의 컨텍스트 위치 편향 발견

실험:
  20개 문서 중 1개에 정답 포함
  정답 문서 위치를 변경하며 Q&A 성능 측정

결과:
  - 처음과 끝에 있을 때 성능 최고
  - 중간에 있을 때 성능 최저 (U자형 곡선)
  - 문서 수가 많을수록 중간 정보 손실 심화

실용적 의미:
  - 중요 정보는 프롬프트 시작이나 끝에 배치
  - RAG: 가장 관련 있는 문서를 앞/뒤에 배치
  - 평균 풀링보다 MaxPool 또는 선택적 주의 필요
```

---

## 데이터

### FineWeb (2024)
```
핵심 기여: 고품질 웹 데이터 파이프라인 공개

파이프라인 (CommonCrawl → 고품질 텍스트):
  1. 언어 감지: fastText → 영어만
  2. URL 필터링: 성인/스팸 도메인 제거
  3. trafilatura: HTML에서 주요 텍스트 추출
  4. 품질 필터: 문자 반복비, 단어 반복비, 특수문자
  5. MinHash 중복 제거 (5-gram, threshold=0.7)
  6. C4 필터: 욕설, JS 코드, 불완전 문장 제거

FineWeb-Edu:
  - 교육적 가치 점수 (LLM Annotator)
  - 5점 만점, 임계값 3 이상만 사용
  - 15T 토큰 → 1.3T 토큰 (고품질 교육 데이터)
  - Phi 모델이 "Textbooks are all you need"의 근거
```

### Magpie (2024)
```
핵심 기여: 자동 SFT 데이터 생성

방법:
  Instruction-tuned LLM에게 User 턴만 제공
  → 모델이 스스로 사용자 질문 생성
  → 같은 모델로 응답 생성

vs Self-Instruct:
  Self-Instruct: 시드 태스크 기반, 다양성 한계
  Magpie: 모델의 instruction space 직접 탐색
         더 자연스럽고 다양한 질문 생성

성능:
  300K Magpie 데이터로 학습한 모델이
  Llama-3-8B-Instruct와 동등 또는 우수
```

---

## 평가

### MT-Bench (2023) / LLM-as-Judge
```
핵심 기여: LLM으로 LLM 평가 자동화

MT-Bench:
  - 80개 다회차 대화 질문 (8가지 범주)
  - GPT-4로 1-10점 평가
  - 인간 평가와 0.8+ 상관관계

편향 분석:
  - Position Bias: 첫 번째 응답 선호
  - Verbosity Bias: 긴 응답 선호
  - Self-bias: 같은 모델 생성 선호

  해결: A/B + B/A 양방향 평가, 평균

강점: 확장 가능, 비용 효율, 빠름
약점: 평가자 LLM의 편향, 주관적 품질 기준
```

### Chatbot Arena / ELO (2023)
```
핵심 기여: 실제 사용자 기반 LLM 랭킹

방법:
  - 두 모델이 같은 질문에 익명으로 응답
  - 사용자가 더 좋은 응답 선택
  - ELO 점수 시스템으로 랭킹 계산

ELO 업데이트:
  E(A wins) = 1 / (1 + 10^((ELO_B - ELO_A)/400))
  ELO_A_new = ELO_A + K × (result - E(A wins))
  K=32 (k-factor)

장점: 실제 사용자 선호 반영, 조작 어려움
단점: 느린 수렴, 사용자 편향, 도메인 편향
```

---

## 읽기 순서 추천

```
1단계 (반드시):
  Attention is All You Need → 모든 것의 기초
  Chinchilla → 스케일링 이해
  InstructGPT → RLHF 실용화
  LLaMA-2 → 현대 오픈소스 LLM 표준

2단계 (정렬):
  DPO → InstructGPT 이후 정렬 표준
  Constitutional AI → RLAIF 이해
  Flash Attention → 효율성 이해

3단계 (추론 최적화):
  vLLM/PagedAttention → 서빙 이해
  Speculative Decoding → 추론 가속
  GPTQ 또는 AWQ → 양자화 이해

4단계 (최신):
  DeepSeek-V2/V3 → MLA, MoE 최신 기술
  DeepSeek-R1 → RL 기반 추론
  관심 분야 ArXiv daily (Hugging Face Papers 추천)
```

---

## 논문 읽는 법

```
빠른 독법 (30분):
  1. 제목 + 초록: 무엇을 하는가?
  2. Figure 1-3: 직관적 이해
  3. 실험 테이블: 얼마나 좋은가?
  4. 결론/한계: 약점은?

심층 독법 (2-3시간):
  1. 서론: 왜 이 문제인가? 기여가 무엇인가?
  2. 관련 연구: 기존 방법과 차이?
  3. 방법론: 핵심 수식과 알고리즘 완전 이해
  4. 실험: 어떤 기준으로 평가? 공정한가?
  5. 부록: 구현 디테일 (재현에 중요)

재현 (며칠):
  1. 공식 코드 읽기
  2. 핵심 아이디어만 간단히 구현
  3. Paper의 ablation 재현 시도
  → 완전히 이해 가능

ArXiv 트래킹 도구:
  - Hugging Face Papers (daily summary)
  - Semantic Scholar
  - Papers With Code (코드 있는 논문)
  - Twitter/X의 ML 연구자 팔로우
```

---

## 2025년 주목 논문 (최신)

```
추론 모델:
  - DeepSeek-R1: RL로 오픈소스 추론 모델
  - QwQ-32B: Qwen 추론 모델
  - Gemini Flash Thinking: Google 추론

효율화:
  - DeepSeek-V3: FP8 학습, MoE 효율화
  - MiniMax-Text-01: 초장문 컨텍스트 (4M)

VLM:
  - Gemini 2.0: 멀티모달 네이티브
  - GPT-4o 멀티모달 강화
  - InternVL3: 오픈소스 최강 VLM

에이전트:
  - Claude 3.7 Sonnet: Extended Thinking
  - SWE-bench SOTA 향상 (에이전트 코딩)

핵심 트렌드:
  - Test-time Compute (추론 시 계산 증가)
  - Thinking/Reasoning 모델 확산
  - 긴 컨텍스트 (1M+) 표준화
  - 소형 고성능 모델 (Qwen2.5-7B, Gemma-3)
```
