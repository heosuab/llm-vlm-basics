# Section 12: Emerging Trends (2024+)

## Multimodal Scaling Laws

### 단일 모달리티 Scaling Law를 넘어서

기존 LLM scaling law (Chinchilla): compute, parameters, data의 최적 비율
→ Multimodal에서는 텍스트 + 이미지 + 비디오 + 오디오를 어떻게 조합해야 하는가?

**연구 질문들**:
- Vision encoder의 크기와 LLM 크기의 최적 비율은?
- Vision data와 text data의 최적 혼합 비율은?
- Visual token의 수와 모델 크기의 관계는?

아직 명확한 답이 없으며, 활발히 연구 중인 영역.

---

## Large Multimodal Context

### 초장문 multimodal context

Gemini 1.5의 1M token context는 단순히 텍스트뿐 아니라 이미지, 비디오도 포함할 수 있음.

**실용적 활용**:
- 긴 동영상 전체를 컨텍스트로 (1시간 분량의 영상)
- 수백 페이지 문서 + 차트 + 이미지를 한번에 처리
- 코드베이스 전체를 컨텍스트로

**도전**:
- 시각 토큰 수가 폭발적으로 늘어남 (1280×720 비디오 1시간 = 수억 tokens)
- Attention의 O(n²) 문제가 더 심각
- "Lost in the middle"이 multimodal에서도 발생

---

## Video LLMs

### 비디오 이해의 도전

```
비디오 = 시간 차원이 추가된 이미지 시퀀스

도전:
1. 토큰 폭발: 30fps, 1분 = 1800 프레임 × 256 tokens = 460K tokens
2. 시간적 관계: "무엇이 먼저 일어났는가?"
3. 긴 비디오에서의 이벤트 위치 파악
4. Fine-grained 시간 이해 (정확한 timestamp)
```

### 주요 접근법

**프레임 서브샘플링**: 1fps, 0.5fps 등으로 줄임
**Temporal pooling**: 인접 프레임의 특징 집약
**Token compression**: 시간적으로 유사한 프레임의 token 병합 (LLaVA-Video의 temporal pooling)

### 대표 모델

- **Qwen2-VL**: 비디오 이해 최강 오픈소스
- **InternVL2**: 시간적 이해 우수
- **GPT-4o**: 실시간 비디오 스트리밍 처리

---

## Audio-Language Models

### 텍스트 + 이미지를 넘어 오디오까지

**GPT-4o**: 텍스트, 이미지, 오디오를 하나의 모델로 처리
- 오디오를 텍스트로 변환하지 않고 직접 처리
- 억양, 감정, 배경 소음까지 이해
- 실시간 대화 가능

**Gemini 2.0**: 오디오 입력 + 출력 모두 지원

### Audio Token

오디오를 token으로 변환하는 방법:
- **Mel spectrogram → CNN/Transformer**: Whisper 방식
- **Discrete audio codec**: EnCodec, DAC로 오디오를 discrete token으로
  → LLM vocabulary에 추가하여 autoregressive 생성

---

## Multimodal Reasoning Benchmarks

단순 인식(recognition)을 넘어 **추론(reasoning)** 능력을 평가.

**MathVista**: 수학 문제 + 이미지 (차트, 기하학, 그래프)
**MMSci**: 과학 논문의 이미지 + 텍스트 이해
**MMCode**: 코드 실행 결과 예측, 코드 + UI 이미지

---

## Self-Improving Models

### 핵심 아이디어

모델이 **자기 자신의 출력을 사용하여 스스로 개선**.

**방법 1: RLVR with Verifiable Rewards**
```
수학 문제 풀기 → 정답 여부 자동 확인 → reward signal
→ reward 기반으로 policy 개선 → 다시 문제 풀기
무한 반복
```

**방법 2: Rejection Sampling Fine-tuning**
```
모델이 N개 답변 생성 → verifier로 필터링 → 좋은 답변만 SFT
→ 개선된 모델로 다시 N개 생성 → 반복
```

**방법 3: Self-play**
```
모델이 문제도 만들고 답도 만들고, 자기 답을 자기가 평가
→ 점점 어려운 문제를 스스로 만들고 풀면서 성장
```

**한계**: Verifiable reward가 있는 도메인(수학, 코드)에서만 잘 작동.

---

## Tool-Augmented LLMs

### 개념

LLM이 external tool을 호출하여 지식과 능력을 확장.

```
LLM + Tools:
  - Web search: 최신 정보 검색
  - Code interpreter: 코드 실행, 수치 계산
  - Calculator: 정확한 산술
  - Database: 정형 데이터 조회
  - Image generation: DALL-E, Stable Diffusion 호출
  - File system: 파일 읽기/쓰기
```

### 구현 방식

**Function Calling** (OpenAI 방식):
```json
{
  "function": "search_web",
  "arguments": {"query": "latest earthquake in Japan 2025"}
}
→ 검색 결과 반환 → LLM이 결과를 처리하여 응답
```

**ReAct (Reason + Act)**:
```
Thought: I need to find the current exchange rate
Action: search("USD to KRW exchange rate today")
Observation: 1 USD = 1,350 KRW
Thought: Now I can calculate...
Answer: ...
```

---

## Agentic Systems

### LLM Agent

단순한 Q&A를 넘어 **장기적 목표를 달성하기 위해 자율적으로 계획하고 행동**.

```
Goal: "내 코드베이스를 분석하고 버그를 찾아서 수정하고 PR을 열어라"

Agent loop:
1. Read file system → 코드 파악
2. Run tests → 실패하는 테스트 발견
3. Analyze code → 버그 원인 파악
4. Write fix → 코드 수정
5. Run tests again → 통과 확인
6. Create PR → GitHub API 호출
```

**멀티 에이전트**: 여러 LLM agent가 협업 (planning agent, coding agent, testing agent 등)

**주요 프레임워크**: Claude Code, Devin, SWE-Agent, AutoGPT

---

## Structured Output Training

### 문제

LLM을 API로 사용할 때 항상 valid JSON을 반환해야 하는 경우가 많습니다. Inference-time constraint 외에 **학습 자체를 structured output에 맞게** 함.

```python
# 기존: 자유 텍스트 생성 → JSON parsing 시도 → 실패 가능
# 개선: 학습부터 JSON schema를 따르는 데이터로 SFT
```

**구조화 데이터 생성**:
- 정보 추출 (NER, relation extraction)
- 코드 생성 (항상 valid syntax)
- Function calling (valid JSON arguments)

---

## Modular LLM Systems

### 단일 거대 모델의 한계

모든 것을 하나의 모델로 처리하는 것이 항상 최선이 아님:
- 특정 task에 오버킬 (간단한 분류에 70B 모델 불필요)
- 특수 domain에서 전문 모델보다 약할 수 있음
- 모든 모달리티를 하나로 합치면 학습이 복잡

### 모듈화 방법

**Mixture-of-Experts**: 이미 일종의 모듈화 (전문화된 expert들)

**Routing**: 입력에 따라 적절한 모델로 라우팅
```
간단한 질문 → 8B 모델 (빠르고 저렴)
복잡한 추론 → 70B 모델
코딩 → Code-specialized model
```

**Pipeline**: 여러 모델의 조합
```
Vision encoder → VLM projector → LLM backbone
각 컴포넌트를 독립적으로 업그레이드 가능
```

**Retrieval + Generation**: RAG도 일종의 모듈화 (retriever + generator 분리)

---

## 2025-2026 Key Trends 요약

| 트렌드 | 상태 | 핵심 의미 |
|--------|------|---------|
| Test-time compute scaling | 성숙 중 | Reasoning 능력의 새 차원 |
| Native multimodal | 진행 중 | 텍스트/이미지/오디오 통합 |
| Agentic AI | 초기 | LLM이 진짜 "일"을 하기 시작 |
| Small-but-capable models | 가속 중 | 효율성과 접근성 향상 |
| World models | 초기 | Embodied AI의 핵심 |
