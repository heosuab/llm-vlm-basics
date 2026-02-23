# 심화 개념 질문 & 답변 — 연구자 편

## Transformer & Attention

**Q1. Self-Attention의 시간/공간 복잡도는? Flash Attention은 어떻게 개선하나?**
```
시간: O(n²d), 공간: O(n²) — n은 시퀀스 길이

표준 Attention의 메모리 병목:
  1. QKᵀ 행렬: n×n float16 → n=8K 시 512MB
  2. Softmax(QKᵀ): 동일
  3. 각 행 전체를 읽어야 softmax 계산 가능 → HBM I/O 병목

Flash Attention (IO-aware):
  타일링: 블록 단위로 SRAM에서 계산 (전체 행렬 불필요)
  온라인 소프트맥스: 블록별로 점진적 계산
    m_new = max(m_old, max(current_block))
    l_new = l_old × exp(m_old - m_new) + sum(exp(x - m_new))
  Recomputation: backward 시 HBM에서 재계산 (중간 저장 X)

결과:
  메모리: O(n²) → O(n) (attention matrix HBM 저장 없음)
  속도: 2-4× 향상 (HBM I/O 대폭 감소)
  핵심: FLOP 감소가 아닌 IO 병목 해결
  Flash Attention 2: Q 방향 병렬화, 더 나은 warp 분할
  Flash Attention 3: H100 FP8 지원, async pipeline, ~75% 하드웨어 활용
```

**Q2. Pre-LN vs Post-LN 차이와 현대 모델이 Pre-LN 쓰는 이유는?**
```
Post-LN: x → Sub-layer → x + Sub-layer(x) → LayerNorm
Pre-LN:  x → LayerNorm → Sub-layer → x + Sub-layer(LN(x))

Post-LN 문제:
  Residual에 LayerNorm 적용 전 → gradient explosion/vanishing
  특히 깊은 모델에서 초기 학습 불안정
  Warmup 없이 학습 어려움

Pre-LN 장점:
  LayerNorm이 Residual Path 통과 전에 적용
  → gradient가 항상 normalized 된 채로 흐름
  → Warmup 없이도 안정적 학습 가능
  현대 LLM 표준 (LLaMA, GPT-3, PaLM 등)

단점:
  표현력 이론적으로 약간 제한 (일부 연구)
  Sandwich Norm: Pre-LN + 추가 Post-norm으로 보완

RMSNorm vs LayerNorm:
  LayerNorm: 평균과 분산 모두 계산, scale + shift
  RMSNorm: RMS만 계산 (평균 제거), scale만
  → 더 빠름, 품질 거의 동일
  → LLaMA, Mistral, Gemma 채택
```

**Q3. RoPE가 ALiBi보다 현대 모델에서 선호되는 이유는?**
```
ALiBi (Attention with Linear Biases):
  score(i, j) = qᵢᵀkⱼ/√d - m|i-j|  (거리 패널티 직접 추가)
  장점: 구현 단순, 외삽 시 성능 저하 완만
  단점: 거리 페널티가 전역 고정 → 내용 기반 위치 학습 불가
        Flash Attention과 통합 어색

RoPE (Rotary Position Embedding):
  qₘ = Rₘ·q, kₙ = Rₙ·k (회전 변환 적용)
  qₘᵀkₙ = qᵀRₘᵀRₙk = qᵀR(n-m)k (상대 위치만 의존)

  장점:
    내적이 상대 위치만의 함수 → attention이 자연스럽게 상대 위치 고려
    YaRN, NTK scaling 등으로 컨텍스트 확장 가능
    Flash Attention과 자연스럽게 결합
    학습 시 보지 못한 위치로 외삽 가능 (확장 기법 적용 시)
  단점: 단독으로는 외삽 어려움 (확장 기법 필요)

실제: 거의 모든 현대 LLM이 RoPE 채택 (LLaMA, Mistral, Qwen, Gemma)
```

**Q4. GQA(Grouped-Query Attention)의 메모리 절약 원리와 품질 영향은?**
```
MHA: h Query Heads, h KV Heads
  KV Cache: 2 × L × h × d_head × T bytes
  Llama-3-8B (32 heads): 32 × KV

GQA: h Query Heads, G KV Heads (G << h)
  각 KV Head를 h/G개의 Query Head가 공유
  KV Cache: 2 × L × G × d_head × T bytes

실제 (LLaMA-3-8B):
  h=32 Query, G=8 KV (4:1 비율)
  KV Cache: 1/4 감소
  배치 크기 4× 증가 가능 (또는 시퀀스 4× 길게)

품질:
  MQA (G=1): 메모리 1/h, 성능 저하 있음
  GQA (G=4~8): 메모리 감소, MHA와 거의 동일 품질
  Fine-tuning 기법: 기존 MHA 체크포인트를 GQA로 업스케일링 가능
    → mean pooling으로 KV head 초기화
```

**Q5. DeepSeek-V2의 MLA(Multi-head Latent Attention)가 GQA보다 뛰어난 이유는?**
```
GQA: K, V head를 그룹별로 공유 (단순 공유)
MLA: K, V를 저차원 잠재 벡터로 압축 후 복원

MLA 메커니즘:
  c^KV = W^DKV · h        (h: d_model → d_c, 압축)
  K = W^UK · c^KV         (d_c → (num_heads × d_head))
  V = W^UV · c^KV

  KV Cache: c^KV 저장 (d_c << num_heads × d_head)
  d_c = 512 (GQA 대비 훨씬 작음)

추론 효율:
  저장: c^KV만 저장 → 93.3% KV cache 감소 (DeepSeek-V2)
  계산: c^KV를 K,V로 복원 후 일반 attention 수행
         → 표현력 GQA보다 우수

학습 trick:
  행렬 흡수: W^UK를 Q projection에 흡수
  → 실제 attention 시 K 복원 없이 계산 가능
```

---

## 학습

**Q6. Chinchilla Scaling Law와 실제 모델의 차이는?**
```
Chinchilla (Hoffmann et al., 2022):
  L(N, D) = E + A/N^α + B/D^β
  E: 불가역 손실, N: 파라미터, D: 토큰 수

  compute-optimal: N ∝ D (대략 N_tokens = 20 × N_params)
  Chinchilla 70B: 1.4T 토큰 (70B × 20)

현실 모델:
  LLaMA-3-8B: 15T 토큰 (Chinchilla 대비 10×)
  LLaMA-3-70B: 15T 토큰 (Chinchilla 대비 ~3×)

이유: Inference-optimal vs Training-optimal
  더 작은 모델을 더 많이 학습하면
  → 추론 비용이 크게 감소 (같은 품질 대비)
  → 서비스 규모에서는 학습 비용 < 추론 비용

교훈:
  Scaling law는 학습 compute 최적화
  실제 서비스는 inference cost 최적화
  → Chinchilla "optimal"이 서비스에서는 "optimal"이 아님
  LLM.int8()처럼 더 작은 모델을 오래 → 양자화로 추론 경제성 달성
```

**Q7. Gradient Checkpointing의 원리와 trade-off는?**
```
표준 역전파:
  Forward: 모든 activation 저장 → O(n) 메모리 (n: 레이어 수)
  Backward: 저장된 activation으로 gradient 계산

Gradient Checkpointing:
  Forward: 체크포인트 레이어만 저장, 나머지 폐기
  Backward: 필요 시 체크포인트에서 재계산
  메모리: O(n) → O(√n) (√n 체크포인트 균등 배치)

트레이드오프:
  메모리: O(n) → O(√n) (33B 모델에서 ~40% 감소)
  속도: ~30% 느려짐 (재계산 비용)
  유용 시기: GPU OOM, 배치 크기 증가, 긴 시퀀스

PyTorch 구현:
  torch.utils.checkpoint.checkpoint(function, *inputs)
  Hugging Face: gradient_checkpointing_enable()

Activation Offloading (확장):
  GPU → CPU 비동기 offload → GPU에서 필요 시 복사
  속도는 더 느리지만 메모리 극단적으로 절약
```

**Q8. MoE 모델의 Load Balancing이 왜 중요하고 어떻게 구현하나?**
```
왜 중요한가:
  학습: 모든 Expert가 충분히 학습되어야 성능 좋음
    → "hot expert"만 계속 선택되면 나머지 무의미
  추론: GPU간 균등 분배 → 부하 균형
    → Expert Parallelism에서 특정 GPU만 과부하

측정:
  Load Imbalance = max_expert_load / avg_expert_load
  이상적: ~1.0, 현실: 2-5× 가능

방법 1: Auxiliary Load Balancing Loss
  L_aux = α × N × Σᵢ fᵢ × pᵢ
  fᵢ: expert i가 처리한 토큰 비율
  pᵢ: expert i로 라우팅된 평균 확률
  → fᵢ와 pᵢ 같이 작아야 함 (균등 분배)
  단점: α가 크면 성능 저하 (라우팅 자유도 감소)

방법 2: DeepSeek-V3 Loss-free
  동적 Expert Bias: b_i += α if f_i < target else -α
  라우팅: topk(gating + b_i)
  b_i는 균형용, gradient에 영향 없음
  → 성능 저하 없이 균형 달성
```

---

## Alignment

**Q9. DPO가 왜 PPO와 수학적으로 동일한 최적해를 갖나?**
```
KL-constrained RL 목표:
  max E_{y~π}[r(x,y)] - β·KL(π||π_ref)

라그랑주 풀면 최적 정책:
  π*(y|x) = π_ref(y|x) × exp(r(x,y)/β) / Z(x)
  Z(x): 정규화 상수

역으로 표현:
  r*(y|x) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x)

Bradley-Terry 선호 모델:
  P(y_w ≻ y_l) = σ(r(x,y_w) - r(x,y_l))
                = σ(β·log(π*(y_w|x)/π_ref(y_w|x))
                   - β·log(π*(y_l|x)/π_ref(y_l|x)))

DPO 손실 (MLE):
  L_DPO = -E[log σ(β·log(π_θ(y_w|x)/π_ref(y_w|x))
                  - β·log(π_θ(y_l|x)/π_ref(y_l|x)))]

→ 동일한 최적해, RM 학습 불필요, PPO 불필요
→ 하지만 오프라인 (고정 데이터), PPO는 온라인 (지속 탐색)
실제 차이: DPO는 분포 외 응답 처리 어려움
```

**Q10. GRPO와 PPO의 핵심 차이는?**
```
PPO 구조:
  - Actor (정책): 학습 대상
  - Critic (가치 함수): advantage 추정
  - Advantage = r - V(s)    (실제 - 예측)
  - 추가 모델 필요 (메모리/계산 2×)

GRPO 구조:
  - 같은 질문에 G개 응답 생성 (보통 G=8~16)
  - 그룹 내 상대적 advantage 계산:
    Aᵢ = (rᵢ - mean(r_group)) / std(r_group)
  - Critic 불필요 (그룹 평균이 baseline)
  - 손실: L = -E[min(rᵢAᵢ, clip(rᵢ,1-ε,1+ε)Aᵢ)] + β·KL

장점:
  메모리: Critic 모델 없음 (1.5× 절약)
  안정성: 그룹 정규화로 advantage 스케일 자동 조정
  검증 가능 태스크: 수학/코딩 정확성으로 간단한 보상

단점:
  G개 응답 생성 필요 → G배 추론 비용
  연속적 행동 공간(창의적 태스크)에서 PPO보다 효과 낮을 수 있음
```

**Q11. SimPO가 DPO보다 나은 이유는?**
```
DPO 문제점:
  1. Reference 모델 의존: 항상 π_ref 필요 (메모리, 계산)
  2. Verbosity bias: 더 긴 응답이 선호될 수 있음 (로그확률 합산)
  3. 분포 외 응답: offline 데이터의 분포 바깥에서 예측 불정확

SimPO (Reference-free DPO):
  보상: r(x,y) = (1/|y|) Σ log π_θ(yᵢ|x,y<i) - γ
        (평균 로그 우도 - 마진 γ)

  손실: L = -log σ(r(x,y_w) - r(x,y_l))

장점:
  Reference-free: π_ref 불필요 (메모리, 계산 절약)
  길이 정규화: 평균 로그우도 → verbosity bias 제거
  γ (마진): 선호 응답이 명확히 더 높아야 함 → calibration 개선

성능: AlpacaEval 2.0에서 DPO 대비 +4-8% 향상
```

---

## 아키텍처 심화

**Q12. Mamba의 Selective SSM이 기존 SSM과 어떻게 다른가?**
```
기존 S4 SSM: A, B, C 행렬이 고정 (입력과 무관)
  h(t) = Āh(t-1) + B̄u(t)   (고정된 Ā, B̄)
  y(t) = Ch(t)

Mamba Selective:
  B(t) = s_B(u(t)) = Linear(u(t))  [input-dependent]
  C(t) = s_C(u(t)) = Linear(u(t))  [input-dependent]
  Δ(t) = softplus(s_Δ(u(t)))       [input-dependent time step]

  Δ_t 클수록: 해당 입력 "기억" → state 크게 업데이트
  Δ_t 작을수록: 해당 입력 "무시" → state 거의 유지

직관: Transformer의 content-based attention과 유사한 선택성
  차이: attention은 O(n²), SSM은 O(n) (recurrent)

Hardware-Aware Algorithm:
  Selective scan: 입력 의존적 파라미터 → 병렬 scan 불가
  해결: CUDA parallel scan 알고리즘 직접 구현
       Recomputation으로 backward 메모리 절약

추론:
  상태 h ∈ R^{d_model × d_state} 유지 (고정 크기)
  O(1) 메모리, O(1) 시간 per step (vs O(n) KV cache)
```

**Q13. 왜 SwiGLU가 ReLU보다 LLM에서 선호되나?**
```
활성화 함수 발전:
  ReLU: max(0, x) → 50% neurons 항상 0, "dying ReLU" 문제
  GELU: x·Φ(x) → 부드러운 ReLU, BERT/GPT-2 표준
  SiLU (Swish): x·σ(x) → 자기 게이팅, smooth, β=1 Swish
  GLU: σ(xW₁)·(xW₂) → 게이팅 메커니즘 (source-target attention)
  SwiGLU: SiLU(xW₁)·(xW₂) → SiLU + GLU 결합

SwiGLU = Gate × Up:
  down(silu(linear_gate(x)) × linear_up(x))
  gate 경로: 무엇을 통과시킬지 결정
  up 경로: 실제 값

이유:
  게이팅: 각 뉴런이 정보 흐름 제어 → 표현력 향상
  부드러운 0 근처: gradient 소실 없음 (ReLU와 달리)
  경험적: 일관된 성능 우위 (Noam Shazeer 연구)

비용:
  W_gate + W_up + W_down = 3개 행렬 (vs ReLU 2개)
  → d_ff를 8/3 × d_model으로 줄여 파라미터 유지
  LLaMA-3-8B: d_ff = 14336 (= 8/3 × 4096 ≈ 10922, 여유 있게 설정)
```

---

## 최신 트렌드 & 연구

**Q14. Test-Time Compute Scaling이란 무엇이고 어떻게 활용하나?**
```
핵심 아이디어:
  더 큰 모델 대신 더 많은 "생각 시간"
  compute budget을 학습 시 고정 → 추론 시 가변으로 이동

방법들과 비교:
  1. Best-of-N:
     N개 샘플 → Verifier/PRM으로 최고 선택
     비용: N× 추론
     단순, 효과적

  2. MCTS (Monte Carlo Tree Search):
     State: 추론 단계
     Action: 다음 추론 단계 생성
     Reward: PRM 점수 또는 최종 정확도
     Selection: UCB1: argmax (V(s) + c√(ln N(parent)/N(s)))
     가장 강력, 가장 비쌈

  3. Sequential Revision (Reflexion):
     응답 → 비평 → 수정 → 반복
     메모리: 이전 시도 모두 context에

  4. Beam Search:
     K개 partial sequence 유지 + 가지치기
     중간: 비용과 품질 균형

스케일링 특성:
  수학/코딩: 검증 가능 → 확실한 보상 신호
  open-ended 태스크: 검증 어려움 → 효과 제한적
  Compute budget↑ → performance↑ (하지만 diminishing returns)

o1/R1의 혁신:
  명시적 Best-of-N이 아닌 암묵적 CoT 연장
  모델 자체가 필요한 만큼 생각
  → budget_tokens 파라미터로 계산량 제어
```

**Q15. Reasoning 모델(o1, R1)이 일반 모델과 다른 점은?**
```
일반 모델 (GPT-4o, Claude-3.5):
  Prompt → 직접 답변 생성
  SFT + RLHF로 학습
  빠르고 저렴
  단순~중간 난이도 태스크에 최적

Reasoning 모델 (o1, R1):
  Prompt → <think>장문 추론</think> → 답변
  학습: SFT (cold start) → RL (GRPO/PPO) → 반복
  보상: 검증 가능한 정답 (수학, 코딩)
  느리고 비쌈, 어려운 문제에 강함

핵심 차이:
  1. Extended Chain-of-Thought: 수십~수백 단계
  2. Self-correction: 틀린 추론 발견 → 수정 (Aha Moment)
  3. Test-time compute scaling: 어려울수록 더 많이 생각
  4. Verifiable Rewards: 수학/코딩 정확성으로 RL

한계:
  - 느림 (1-10분 response time 가능)
  - 비쌈 (20-100× 일반 모델 대비)
  - 단순 태스크 오버엔지니어링 ("Overthinking")
  - 프라이버시/보안: thinking trace 분석 가능

언제 사용:
  수학, 과학, 코딩: Reasoning 모델
  일반 대화, 요약, 번역: 일반 모델
```

**Q16. Knowledge Distillation의 LLM 적용 방법은?**
```
Teacher-Student:
  Teacher: 대형 모델 (70B, 405B, API 모델)
  Student: 소형 모델 (7B, 8B)

방법 1: Output Distillation (On-policy)
  Teacher 생성 데이터로 Student SFT
  LLaMA-3.1-405B-Instruct → LLaMA-3.1-8B-Instruct (Meta)
  DeepSeek-R1 → R1-Distill-{Qwen,LLaMA}

방법 2: Logit Distillation (KL Divergence)
  L = KL(p_teacher(·|x) || p_student(·|x))
    = -Σ_t p_teacher(t|x) log p_student(t|x) + const
  Teacher soft label = 더 많은 정보 (분포 전체)
  Temperature T 높이면 soft label 더 부드러움

방법 3: Feature Distillation
  중간 레이어 표현 맞추기
  Lfeature = ||f_teacher(x) - W·f_student(x)||²
  W: projection (차원 다를 수 있음)
  DistilBERT, TinyBERT에서 효과적

방법 4: Speculative Decoding as Distillation
  Draft 모델 = Student, Target 모델 = Teacher
  Target 검증 과정 → Draft 모델 암묵적 학습 신호

DeepSeek-R1 증류:
  671B R1-Zero의 추론 CoT → 소형 모델 SFT
  R1-Distill-{1.5B, 7B, 8B, 14B, 32B, 70B}
  7B 증류 모델이 일반 70B 모델 능가 (수학 태스크)
```

---

## 분산 학습 & 시스템

**Q17. TP와 PP를 조합할 때 최적 설정을 어떻게 결정하나?**
```
고려 요소:
  TP 제약: num_heads로 나눠야 함
    (LLaMA-3 70B: 64 heads → TP: 1,2,4,8,16,32,64)
  노드 내 GPU 수: TP는 NVLink 필요 (단일 노드 내로 제한)
  PP: 레이어 수로 균등 분할 (PP=4 → 32/4=8 layers each)

비용 분석:
  TP 통신 비용: 2 × num_layers × AllReduce(d_model)
    → 매 레이어마다 동기화
  PP 통신 비용: 각 stage 경계에서 activation 전달
    → 통신량: batch × seq × d_model

통신 속도:
  NVLink (같은 노드): ~900GB/s
  InfiniBand (노드 간): ~400GB/s
  → TP는 NVLink 필요 (매 레이어 동기화), PP는 IB로 충분

실용 규칙:
  노드 내 (8 GPU) → TP=8 (NVLink)
  노드 간 → PP (IB) 또는 DP
  DP는 통신 오버헤드 작음 → 가능하면 DP 최대화

예시 (70B, 16×A100 = 2 노드):
  TP=8 (노드 내) × PP=2 (노드 간) × DP=1
  또는 TP=4 × PP=1 × DP=4 (더 많은 데이터 병렬)
```

**Q18. ZeRO (Zero Redundancy Optimizer)의 3단계 차이는?**
```
데이터 병렬에서 메모리 낭비:
  각 GPU가 전체 모델 복사본 유지
  7B 모델: 가중치 14GB + optimizer state 56GB + gradient 14GB = 84GB

ZeRO-1: Optimizer State 분산
  N개 GPU로 optimizer state 나눔
  각 GPU: optimizer state N분의 1
  통신: AllReduce (gradient) + Scatter (optimizer update)

ZeRO-2: + Gradient 분산
  ZeRO-1 + gradient도 N분의 1
  통신: Reduce-Scatter (gradient) + AllGather (업데이트 후)

ZeRO-3: + Parameter 분산
  ZeRO-2 + parameter도 N분의 1
  각 레이어 forward/backward 시 AllGather로 파라미터 수집
  통신 오버헤드 가장 많지만 메모리 최소

메모리 절약:
  ZeRO-1: ~4× (optimizer state = 4/7 총 메모리)
  ZeRO-2: ~8×
  ZeRO-3: ~N× (N = GPU 수)

ZeRO-Infinity:
  ZeRO-3 + CPU/NVMe offloading
  사실상 무제한 모델 크기 가능 (속도 희생)

Deepspeed vs FSDP:
  Deepspeed: ZeRO-3 표준, 더 성숙
  FSDP: PyTorch 내장, 더 쉬운 통합
```

**Q19. Scaling Law가 가리키는 LLM 연구의 다음 방향은?**
```
현재 Scaling의 한계:
  Data Wall: 고품질 인터넷 텍스트 소진
  Compute Wall: 에너지 비용, 칩 공급 부족
  ROI Wall: 파라미터 추가 효용 감소

대안적 스케일링:
  Test-time compute:
    → 더 큰 모델 대신 더 많은 추론 (o1/R1 패러다임)
    → Scaling curves: verification 가능 태스크에서 강력

  Data quality:
    → 더 많은 데이터 대신 더 좋은 데이터 (Phi, FineWeb-Edu)
    → 합성 데이터 (교육적 가치 높은 데이터 생성)

  Architecture:
    → MoE: FLOPs 절약하며 파라미터 증가 (DeepSeek-V3)
    → Mixture of Depths: 토큰마다 다른 compute

  Multimodal:
    → 텍스트 이외 데이터 (비디오, 오디오, 시뮬레이션)
    → World model: 더 기반적인 이해

예측:
  순수 파라미터 스케일링 수익 감소
  Test-time compute + 데이터 효율 + 아키텍처 혁신 조합
  Domain specialization (범용 → 전문화) 중요성 증가
```

**Q20. 대규모 LLM 학습에서 데이터 오염(Data Contamination)을 어떻게 탐지/방지하나?**
```
탐지 방법:
  1. n-gram 중복 측정:
     benchmark 문제와 학습 데이터의 n-gram 겹침
     threshold: 13-gram 이상 겹치면 오염 의심
     도구: bloomfilter로 효율적 탐지

  2. Memorization 테스트:
     benchmark prefix 제공 → 정확한 suffix 완성 여부
     greedy decoding으로 정확히 완성하면 암기된 것

  3. Min-k% Prob:
     각 토큰의 학습 데이터 포함 확률 추정
     낮은 surprisal → 학습에서 본 가능성 높음

  4. 성능 분포 분석:
     오염된 데이터: 정확한 문자열 → 성능 분포 이상
     유사한 데이터: +10~20% 성능 향상 의심

방지:
  학습 커트오프를 benchmark 출시 이전으로 설정
  benchmark URL 블랙리스트 처리
  dynamic benchmarks 사용 (LiveCodeBench: 새 문제 매주)
  Model-blind evaluation: 평가자가 학습 데이터 접근 불가

실용:
  완전 방지 어려움 → 여러 버전 benchmark 동시 사용
  MMLU vs MMLU-Pro vs 직접 제작 내부 benchmark
```

---

## Further Questions

**Q. Transformer가 In-context Learning을 하는 메커니즘은?**
> 현재 이론: 1) Gradient descent in activation space: attention이 암묵적으로 few-shot 예시를 메모로 gradient update 수행. 2) Induction heads: "A B ... A →" 패턴 탐지 회로가 진화. 3) Implicit Bayesian inference: prior를 가중치에, likelihood를 context에 인코딩. 완전히 밝혀지지 않음. Mechanistic interpretability 활발 연구 중.

**Q. LLM에서 Emergent Abilities는 실재하는가?**
> 논쟁 중. Schaeffer et al. (2023): 선택한 평가 메트릭(비선형)에 의한 인공적 현상 주장 (실제로는 연속적 향상). Wei et al.: 규모에서 갑자기 나타나는 능력들 실재 (CoT, BIG-Bench 태스크). 현재: 대부분 연속적이지만 일부 능력(특히 Reasoning)은 갑작스러운 전환점 존재 가능. 확실히 더 큰 모델이 더 많은 능력을 가짐.
