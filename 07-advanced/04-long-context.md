# Long Context & 컨텍스트 확장

## 긴 컨텍스트의 도전

```
Attention 복잡도: O(n²d) → 매우 긴 시퀀스에서 메모리/속도 문제
  - 128K 토큰: 128K × 128K × d 행렬 연산
  - 메모리: O(n²) → 128K 토큰 = 수십 GB

KV Cache: 시퀀스 길이에 비례한 메모리
  - 7B 모델, 128K 토큰: ~2-4GB/sequence
  - 배치 처리 시 수십~수백 GB

"Lost in the Middle" [Liu et al., 2023]:
  컨텍스트 중간 정보에 대한 집중도 저하
  앞/뒤 정보보다 중간 정보 성능 낮음

위치 외삽 (Position Extrapolation):
  학습 시 보지 못한 길이로 추론 → 성능 저하
  RoPE 등 위치 인코딩의 외삽 문제
```

---

## 컨텍스트 확장 방법

### RoPE 스케일링 기법

```
기본 원리: 학습 컨텍스트 4K → 128K로 확장

1. Linear Scaling (Meta):
   θ'_i = θ_i / s  (s = 목표길이/학습길이)
   단순하지만 고주파 성분 왜곡

2. NTK-aware Scaling (Reddit/blog, 2023):
   기저(base)를 조정:
   θ'_i = base^(2i/d) * s^(2i/(d-2))

   코드:
   new_base = base * (scale_factor ** (d / (d-2)))

   고주파 보존, 저주파 조정 → 더 나은 외삽

3. YaRN (Yet another RoPE extensioN, 2023):
   주파수 별로 다른 스케일링:
   - 고주파 (짧은 거리): 변경 없음 (충분히 학습됨)
   - 중간 주파수: 선형 스케일링
   - 저주파 (긴 거리): NTK-aware 스케일링

   구현:
   freq_threshold = 계산된 임계값
   for i in range(d//2):
       if freq_i > freq_threshold:
           scale_i = 1  # 변경 없음
       else:
           scale_i = 1/s  # 선형 스케일링

   또한 attention 온도 조정:
   score = (QK^T) / (sqrt(d_head) * t)  # t > 1 (attention 분산)

4. LongRoPE (Microsoft, 2024):
   비균일 스케일링: 각 head마다 다른 스케일
   탐색 알고리즘으로 최적 스케일 계수 찾기
   LLaMA-2를 2M 토큰까지 확장
```

### LongLoRA [Chen et al., 2023]

```
핵심: Shifted Sparse Attention (S²-Attn) + LoRA

Shifted Sparse Attention:
  - 전체 attention 대신 로컬 그룹 attention
  - 그룹 간 정보 교환: 절반 head를 half-window shift
  - O(n × w) 복잡도 (w: window size)

  구현:
    Head 절반: 일반 로컬 window attention
    Head 절반: window를 w/2 shift → 그룹 경계 정보 교환
    → 완전한 attention 없이 long-range dependency 포착

LoRA와 결합:
  - Embedding과 Normalization layer: 전체 학습
  - Attention/FFN: LoRA로 경량 학습

결과:
  - 7B 모델을 100K 컨텍스트로 확장
  - 단일 8× A100 GPU에서 학습 가능
  - 전체 파인튜닝 대비 95% 품질 유지
```

### LongLLaMA [Tworkowski et al., 2023]

```
Focused Transformer (FoT):
  메모리 레이어: KNN 기반 외부 메모리 조회
  로컬 attention과 조합

외부 메모리:
  (key, value) 쌍을 저장하는 외부 데이터베이스
  쿼리로 유사한 KV 검색 → attention 입력에 추가
  → 이론적으로 무한한 컨텍스트
```

---

## 효율적 Long Context 아키텍처

### Sliding Window Attention (SWA)

```
Mistral-7B에서 사용:
  각 토큰이 최근 w개만 attention
  w = 4,096 (기본값)

정보 전파:
  직접: window 내로 제한
  간접: 레이어를 통해 전파
  레이어 L에서: 실질적으로 L × w 범위 커버

Rolling KV Cache:
  최신 w개 토큰의 KV만 유지
  오래된 KV 버리기 → 고정 크기 메모리

한계:
  매우 먼 거리 정보는 손실 가능
  Mistral: 일부 레이어에서 Full attention 사용 (보완)

SWA 복잡도: O(n × w) 선형 (w 고정)
```

### Longformer & BigBird

```
Local Attention + Global Attention:
  Local: 모든 토큰에 window attention (O(n×w))
  Global: 특수 토큰(CLS, [SEP])에 전체 attention (O(n))
  Random: 일부 랜덤 attention (BigBird)

복잡도: O(n) (w, global tokens 고정)

Longformer: 문서 분류, QA에 강점
BigBird: 더 긴 문서, 수학적 완전성 증명

한계: Generation task에서 비효율적
→ Encoder 모델에 주로 사용
```

### StreamingLLM [Xiao et al., 2023]

```
발견: "Attention Sink" 현상
  첫 번째 토큰(BOS)이 항상 큰 attention 가중치 받음
  → "Sink" 역할: 무의미한 정보를 흡수

무한 길이 스트리밍 처리:
  Attention Sink 토큰: 처음 4개 토큰 항상 유지
  Sliding Window: 최근 N개 토큰 유지
  메모리: O(sink + window) = O(1) 고정

"초기 토큰 + 최근 토큰"만으로 안정적 추론 가능
→ 수백만 토큰 처리 가능, 성능은 근사적

적합한 용도: 실시간 처리, 스트리밍 대화, 온-디바이스 추론

주의:
  초기 토큰과 최근 토큰 사이의 중간 정보 손실
  정확한 long-range recall 불가
```

### Infini-Attention (Google, 2024)

```
무한 컨텍스트를 유한 메모리로:

  기존 Attention:
    최신 L 토큰 KV cache

  Infini-Attention 추가:
    Compressive Memory M (고정 크기 행렬)
    오래된 KV를 M에 압축하여 저장

  Compressive Memory 업데이트:
    M_s = M_{s-1} + K_s^T · V_s  (key^T × value 누적)
    (σ(Q_s) · K_s^T + 1)로 정규화

  Long-term Retrieval:
    A_long = σ(Q) · M / (σ(Q) · K^T + 1)

  최종 Output:
    A = sigmoid(β) ⊙ A_long + (1-sigmoid(β)) ⊙ A_local
    β: 학습 가능한 gate

장점:
  메모리 O(1) (고정 크기 M)
  이론적으로 무한 컨텍스트
  Local + Long-term 정보 조합

단점:
  압축 과정에서 정보 손실
  구현 복잡도
```

---

## "Lost in the Middle" 현상 [Liu et al., 2023]

```
실험:
  20개 문서 중 관련 문서 위치 변경
  맨 앞(position 0): 성능 높음 ✓
  맨 뒤(position 19): 성능 높음 ✓
  중간(position 7-13): 성능 저하 ✗

원인 분석:
  1. Recency bias: 최근 정보에 더 집중 (decoder 특성)
  2. Primacy bias: 앞쪽 정보에 더 집중 (attention sink)
  3. 긴 컨텍스트에서 attention이 분산됨

실용적 해결책:
  1. 중요 정보를 앞/뒤에 배치 (RAG에서 중요)
  2. Reranking으로 중요 문서 앞에 배치
  3. 더 긴 컨텍스트 모델 사용 (부분적 개선)
  4. Chunked processing: 각 청크 독립 처리 후 통합

Needle-in-a-Haystack 벤치마크:
  컨텍스트 × 삽입 위치 grid → 정확도 매핑
  시각화: "hot map" 형태
  Gemini 1.5 Pro: 거의 전 위치에서 높은 정확도 달성
```

---

## 긴 문서 처리 전략

### Map-Reduce

```
긴 문서 → 청크로 분할
  Map: 각 청크 독립적으로 처리
  Reduce: 결과 통합

예: 긴 소설 요약 (200K 토큰)
  청크 1 (0-4K): 요약 1
  청크 2 (4K-8K): 요약 2
  ...
  청크 N: 요약 N
  Reduce: 모든 요약 통합 → 전체 요약

단점: 청크 경계에서 맥락 단절
해결: 청크 간 overlap 추가
```

### Recursive Summarization

```
1단계: 각 청크 요약 (4K → 200 토큰)
2단계: 요약들을 다시 합쳐 요약
3단계: 최종 요약

Tree-like processing
단점: 세부 정보 손실 가능

활용: GPT-4 to summarize books (Paul Graham, 2023)
```

### Retrieval-Augmented Long Context (RAG)

```
긴 컨텍스트 전체를 넣는 대신
  → 관련 부분만 검색해서 넣기

Flow:
  1. 긴 문서를 청크로 분할
  2. 청크를 벡터 임베딩으로 변환
  3. 쿼리로 유사한 청크 검색 (Top-K)
  4. 검색된 청크만 LLM 컨텍스트에 삽입

장점:
  O(1) 컨텍스트 (검색된 청크만)
  비용 효율적

단점:
  전체 맥락 이해 어려움
  관련성 판단의 한계 (embedding의 한계)

Long Context vs RAG 비교:
  Long Context:
    - 전체 문서 접근 가능
    - 비용 비쌈 (O(n²) attention)
    - 정확한 정보 접근
  RAG:
    - 부분적 문서만 접근
    - 비용 저렴
    - 검색 실패 가능성
```

### RAPTOR [Sarthi et al., 2024]

```
계층적 요약 트리:
  1. 원본 청크 → 리프 노드
  2. 유사한 청크 클러스터링
  3. 각 클러스터 요약 → 상위 노드
  4. 반복 → 트리 형성

검색:
  다층 검색: 상위 요약 + 관련 세부 내용
  전체 맥락 + 세부 정보 동시에 접근
```

---

## Flash Attention과 Long Context

```
Flash Attention (Dao et al., 2022):
  SRAM을 활용한 메모리 효율적 attention
  O(n²) 계산 필요하지만 메모리 접근 최적화

  표준 Attention:
    Q, K, V → DRAM에서 읽기/쓰기 반복
    메모리: O(n²), IO: O(n² · d)

  Flash Attention:
    Tiling: Q, K, V를 블록으로 나눠 SRAM에 올림
    Online softmax: 전체 softmax 없이 점진적 계산
    IO: O(n · d) (n²에서 n으로!)
    → 긴 시퀀스에서 2-4× 빠름

Flash Attention 2 & 3:
  병렬화 개선 (head 병렬화)
  FP8 지원 (Hopper GPU)
  causal mask 처리 최적화

Ring Attention [Liu et al., 2023]:
  매우 긴 시퀀스 (1M+)를 여러 GPU에 분산
  GPU0: Q[0], K/V 전체 → attention
  Ring 통신: K/V 순환적 공유
  → 이론적으로 무한 긴 시퀀스 처리 가능
```

---

## 긴 컨텍스트 모델

| 모델 | 컨텍스트 | 특징 |
|------|---------|------|
| Gemini 1.5 Pro | 1M 토큰 | 실용적 1M, "Haystack" 테스트 통과 |
| Claude 3.5 Sonnet | 200K | 고품질, 중간 정보 처리 우수 |
| GPT-4 Turbo | 128K | OpenAI 플래그십 |
| LLaMA-3.1 | 128K | 오픈소스 128K |
| Qwen2.5 | 128K | 오픈소스 강자 |
| LongLLaMA | 무제한 (근사) | FoT 외부 메모리 |
| RWKV-6 | 무제한 | RNN 기반, O(1) 메모리 |
| Mamba | 무제한 | SSM 기반, 연속 상태 |

---

## Needle in a Haystack 테스트

```
평가 방법:
  긴 텍스트 내 특정 위치에 "needle" 삽입
  모델이 needle을 찾을 수 있는지 테스트

  Needle: "The best thing to do in San Francisco is eat a sandwich."
  Haystack: 수천~수백만 토큰의 무관한 텍스트 (Paul Graham essays 등)

  x축: 컨텍스트 길이 (1K ~ 1M)
  y축: Needle 삽입 위치 (0% ~ 100%)
  색상: 정확도

주요 결과:
  대부분 모델: 중간 위치 + 긴 컨텍스트에서 정확도 저하
  Gemini 1.5 Pro: 1M 토큰 전 위치에서 ~99% 달성
  Claude 3: 200K 내에서 안정적

Multi-needle 테스트:
  여러 개의 needle 삽입 → 모두 찾기
  훨씬 어려움 (대부분 모델이 어려워함)
```

---

## 실용적 Long Context 사용 가이드

```
언제 Long Context를 쓰는가:
  1. 코드베이스 분석 (전체 repo)
  2. 긴 문서 요약 (책, 법률 문서)
  3. 다중 문서 비교/합성
  4. 긴 대화 히스토리 유지

언제 RAG가 더 나은가:
  1. 매우 큰 지식베이스 (수백 GB)
  2. 실시간 업데이트 필요 (뉴스, 위키)
  3. 비용 제약
  4. 정확한 소스 추적 필요

비용 최적화:
  1. 짧은 시스템 프롬프트 사용
  2. 청크 재사용 (KV 캐시 활용)
  3. 길이 기반 라우팅: 짧은 쿼리 → 짧은 컨텍스트 모델

컨텍스트 구성 전략:
  1. 중요 정보: 앞 또는 뒤에 배치
  2. 관련도 높은 것 먼저 (Reranking)
  3. 필요 없는 부분 제거 (Context compression)
  4. 문서 간 구분자 명시
```

---

## Context Compression

```
긴 컨텍스트를 압축하여 짧게 만들기:

LLMLingua [Jiang et al., 2023]:
  소형 LM으로 중요하지 않은 토큰 식별 후 제거
  압축률: 10-20× (10K → 1K)
  약간의 정보 손실 허용

  작동 방식:
    1. 소형 LM으로 각 토큰의 perplexity 계산
    2. 높은 perplexity (놀라운) 토큰 → 중요
    3. 낮은 perplexity → 제거 가능
    4. 압축률 설정에 따라 제거

AutoCompressor:
  Transformer로 긴 컨텍스트를 짧은 표현으로 압축
  압축된 임베딩을 LLM에 주입
```

---

## Further Questions

**Q. 128K 컨텍스트 모델을 실제로 128K 전체 채우는 게 좋은가?**
> 꼭 그렇지 않음. 1) Lost in the Middle 문제 (중간 정보 무시). 2) 비용: attention은 O(n²)이므로 128K는 4K 대비 1024배 비쌈. 3) 속도 저하 심각. 최적은 필요한 정보만 RAG로 추출 + 충분한 컨텍스트. 전체 채우기는 정말 필요할 때만 (전체 맥락 필요한 요약 등).

**Q. Sliding Window Attention이 긴 시퀀스에서도 동작하는 이유?**
> 직접 접근은 window 내로 제한되지만, 레이어를 거치며 간접 전파 가능. 레이어 l에서의 receptive field = window × l 범위. LLaMA-3: 전체 레이어의 일부만 SWA 사용하고 나머지는 Full attention. 하지만 먼 거리 정보는 여전히 열화.

**Q. Gemini 1.5 Pro는 어떻게 1M 토큰을 처리하나?**
> 공식 발표된 세부 정보 없음. 추정: 효율적 attention (MQA + Flash), 모델 병렬화, 전문화된 하드웨어 (TPU). 실용적으로 "Needle in a Haystack" 테스트에서 1M 내 거의 모든 위치에서 높은 정확도 확인됨.

**Q. RAG vs Long Context: 어떤 것을 선택해야 하나?**
> 지식이 자주 업데이트되면 RAG. 비용이 중요하면 RAG. 전체 맥락 이해(코드 분석, 책 요약)가 필요하면 Long Context. 최신 트렌드: 두 접근법 조합 (Graph RAG + Long Context).
