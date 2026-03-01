# K-EXAONE (LG)

## 1) 효율성 확보: 학습/추론 비용 절감 전략

### 1.1 혼합 전문가(MoE) 구조
- Dense 대신 **MoE 채택**으로 확장성과 효율을 동시에 확보
- **전문가 128개** 중  
  - **상위 8개 + 공유 전문가 1개**를 활성화 → **총 9개 전문가 동작**
- 파라미터 규모:
  - **전체 파라미터:** 2360억 (236B)
  - **활성 파라미터:** 약 230억 (23B, 약 10%)
- ✅ 효과: **필요 계산량을 줄이면서도 대형 모델급 성능을 목표**

### 1.2 멀티 토큰 예측(MTP)
- 기존: 한 번에 **1 토큰씩** 예측하는 **자율회귀(autoregressive)** 방식 → 느림
- K-EXAONE: **다음 토큰 + 그다음 토큰을 동시에 예측**
- ✅ 효과: **추론 속도 약 1.5배 향상**

### 1.3 토크나이저 고도화 (SuperBPE)
- 기존 EXAONE: 약 **10만 vocab**
- K-EXAONE: 약 **15만 vocab**로 확장
- **SuperBPE 전략**: 자주 등장하는 단어 조합을 **한 토큰(Superword Token)**으로 묶음  
  - 예: `New` + `York` → `New_York`
  - Superword Token이 **전체 vocab의 약 20%**
- ✅ 효과: bytes per token 기준 **평균 약 30% 효율 개선**  
  → 같은 문장을 더 적은 토큰으로 처리 가능(다국어/전문 분야에서도 유리)

---

## 2) 성능 확보: 체계적인 학습 프로세스

### 2.1 3단계 사전 학습(Pre-training)
- 단계별 목표:
  1) 기초 지식
  2) 전문 지식
  3) **추론 능력 강화**
- 3단계에서 **사고 궤적(Thinking Trajectory)** 데이터를 합성해 학습
- ✅ 효과: 정답만 내는 게 아니라 **단계적 사고 흐름을 학습** → 추론 강화에 기여

### 2.2 컨텍스트 길이 확장 (최대 약 26만 토큰)
- 점진적 확장:
  - 1단계: **8k → 33k**
  - 2단계: **33k → 260k(26만)**
- 문제: 긴 컨텍스트만 학습하면 **짧은 컨텍스트 성능 저하 가능**
- 해결: **리허설 데이터셋(고품질 짧은 샘플)**을 함께 학습
- 검증: 학습 중 **NIAH(Needle-In-A-Haystack)** 테스트로 장문 이해 능력 점검

### 2.3 3단계 사후 학습(Post-training)
사전학습=교과서 읽기, 사후학습=실전 문제 풀이 능력 강화

1) **SFT (지도학습)**
- Agentic AI를 위해 “도구 사용 능력”이 중요하지만 현실 환경 구축은 비용 큼
- **가상 도구 사용 환경**을 만들어 데이터 생성 + 검증 가능한 성공 기준 설정
- ✅ 효과: **수백 개의 도구 사용 작업** 학습

2) **RL (강화학습)**
- 수학/코딩/과학/지시 따르기 등에서 강화학습 수행
- **검증 가능한 보상(Verifiable Reward)** 사용  
  - 예: 코드가 통과하면 1점, 실패하면 0점
- 기존 GRPO의 한계를 보완한 **AGAPO** 제안:
  - GRPO: “전부 정답 / 전부 오답” 샘플은 학습에서 제외될 수 있음
  - AGAPO: **전부 오답 샘플도 버리지 않고** 작은 음의 보상을 부여해 학습에 활용
- ✅ 효과: **오답 경로를 피하는 학습**까지 포함해 추론 안정성 강화

3) **Preference Alignment (선호 학습)**
- “똑똑함”뿐 아니라 **친근함/안전/자연스러움**을 맞추기 위한 단계
- SimPER 발전형 **GrouPER** 도입:
  - 여러 답변을 **그룹 단위로 비교**해 학습
  - 개별 평가 노이즈를 상쇄해 더 안정적
- 평가 기준:
  - 규칙 기반: 안전성, 지시 준수 등
  - 루브릭 기반: 창의성, 논리성, 자연스러움 등

---

## 3) 성능 평가: 어디서 잘하나?
평가 영역(벤치마크):
- **지식:** MMLU-Pro  
- **수학:** AIME 2025  
- **코딩:** LiveCodeBench v6  
- **에이전트 도구 사용:** τ²-Bench  
- **지시 따르기:** IFBench  
- **한국어:** KoBALT  
- **다국어:** MMMLU  
- **안전성:** KGC-Safety  

비교 대상:
- EXAONE 4.0 32B, gpt-oss-120b, Qwen3, DeepSeek 등

결과 요지:
- 지식/수학/코딩에서 경쟁력 있는 성과를 강조
- τ²-Bench에서도 우수한 성능 → **멀티스텝 도구 선택/탐색 능력** 강조
- **KGC-Safety에서 최고 성적** → 한국 사회문화 맥락 + 글로벌 윤리 기준을 함께 만족한다고 주장
- Artificial Analysis Intelligence Index 기준  
  **오픈 웨이트 모델 세계 7위·국내 1위**로 등재되었다고 소개

---

## 결론: K-EXAONE이 보여주는 포인트
- **MoE + MTP + 토크나이저**로 **효율**을 극대화
- **추론 강화 사전학습 + 26만 토큰 장문 컨텍스트 + SFT/RL/선호학습**으로 **실전 성능**을 확보
- 특히 **안전성 지표에서 강점**을 강조하며 “한국형 국가대표 AI” 포지셔닝

---

# HyperCLOVA X 8B Omni (OMNI) 

> 출처: *HyperCLOVA X 8B Omni* 기술 리포트(모델 카드/테크니컬 리포트)

---

## 한 줄 요약
**HyperCLOVA X 8B Omni(8B)는 text·audio·vision을 “입력/출력 모두” 지원하는 any-to-any Omni 모델**로,  
서로 다른 modality를 **하나의 decoder-only Transformer** 안에서 **shared next-token prediction interface**로 통합해 end-to-end로 처리한다.

---

## 1) 무엇을 만들었나? (핵심 목표)
- text만으로는 현실 정보를 모두 담기 어렵고, 실제 서비스는 **multimodal I/O**가 필수 → “통합 Omni assistant” 지향
- modality별 pipeline을 따로 두기보다 **하나의 모델에서 understanding + generation**을 함께 수행하도록 설계
- 특히 **Korean/English**를 함께 커버하며 다양한 조합(text→text, vision→text, text→image, speech→text, speech→speech 등)을 목표로 함

---

## 2) 전체 architecture 한눈에 보기
### 핵심 아이디어: “모든 modality를 하나의 sequence로 interleave해서 next-token prediction으로 처리”
- text: **discrete tokens**
- vision/audio:
  - generation 친화성을 위해 **discrete tokens(codebook)**을 LM vocabulary처럼 확장해 함께 예측
  - understanding/grounding을 위해 **continuous embeddings**도 함께 주입
- 최종 출력(image/audio)은 **modality-specific decoder**가 tokens를 pixel/waveform으로 복원

---

## 3) vision 설계 포인트
**“continuous vision encoder + discrete vision tokenizer + diffusion decoder” 조합**
- continuous vision encoder: visual understanding을 위해 LLM backbone과 aligned된 dense feature 제공
- discrete vision tokens: AR(autoregressive) backbone에 맞도록 **semantic-level tokenization** 강조
- diffusion-based vision decoder: semantic tokens의 정보 손실을 보완하며 detail 복원
- vision token budget(예: image/video token count)을 최적화해 **training cost를 크게 줄이는 방향**을 언급

---

## 4) audio 설계 포인트
**continuous audio encoder + discrete audio tokenizer + audio decoder(Unit-BigVGAN)**
- continuous audio encoder: (Whisper 계열 초기화 언급) acoustic 정보를 continuous embeddings로 제공
- discrete audio tokens: generation/AR modeling에 유리한 형태로 speech를 token sequence로 변환
- audio decoder: LLM이 예측한 audio tokens를 waveform으로 복원(speaker embedding conditioning)

---

## 5) training (Pre-training) 전략 요약
### 2-stage 큰 흐름
1) **discrete modality tokens 중심 training**으로 “shared token interface”를 먼저 구축  
2) 그 위에 **continuous (vision/audio) encoders**를 통합해 richer perception을 넣음

### text backbone 효율화: Multi-Token Prediction (MTP)
- 8B 규모에서 training signal density를 높이기 위해  
  **auxiliary prediction head(추가 layer 1개)** + **scaling factor**를 적용하는 **multi-token prediction**을 사용
  - 요지: 기본 next-token objective는 유지하면서 token당 supervision을 늘려 학습 효율을 높이는 목적

---

## 6) post-training 목표와 curriculum
- 목표: pre-trained backbone을 **Korean-centric Omni multimodal assistant**로 만들기
- **4-stage SFT curriculum**으로 점진적 확장:
  1) conversational alignment (text-heavy)  
  2) large-scale multimodal instruction/task 확장  
  3) video + long-context(temporal/long-context) 강화  
  4) intent understanding + multi-step reasoning (예: `<think>` block을 cognitive workspace처럼 활용)

---

## 7) evaluation에서 강조한 메시지
- 단일 모델로 **text·vision·audio의 다양한 input–output 조합**을 폭넓게 평가
- Korean/English benchmarks를 함께 두고, vision-language, ASR/TTS, speech-to-speech 등도 다룸
- 비교군으로 Qwen 계열 Omni 및 text-vision/audio 특화 모델들을 함께 두는 식으로 “지원 modality 범위”를 고려해 비교

---

## 결론: 이 리포트의 핵심 주장
- **shared next-token prediction + interleaved sequence**로 modalities를 통합하고  
- **discrete tokens (generation-friendly)** + **continuous embeddings (understanding/grounding)**을 결합해  
- **8B급에서도 any-to-any Omni를 실용적으로 구현**했다는 포지셔닝
