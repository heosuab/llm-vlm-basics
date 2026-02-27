# LLM / VLM / VLA — Further Questions

> 난이도: `[Easy]` 기본 개념 / `[Medium]` 메커니즘·비교 / `[Hard]` 수학적 유도·연구 수준

---

## 목차

- [0. Fundamentals](#0-fundamentals)
- [I. LLM](#i-llm)
  - [1. Transformer Architecture](#1-transformer-architecture)
  - [2. Positional Encoding & Tokenization](#2-positional-encoding--tokenization)
  - [3. Pretraining](#3-pretraining)
  - [4. Post-Training & SFT](#4-post-training--sft)
  - [5. Alignment](#5-alignment)
  - [6. PEFT](#6-peft)
  - [7. Training Stability & Optimization](#7-training-stability--optimization)
  - [8. Distributed Training](#8-distributed-training)
  - [9. Inference Efficiency](#9-inference-efficiency)
  - [10. Quantization](#10-quantization)
  - [11. Decoding Strategies](#11-decoding-strategies)
  - [12. Long Context & RAG](#12-long-context--rag)
  - [13. Reasoning & Test-Time Compute](#13-reasoning--test-time-compute)
  - [14. Safety](#14-safety)
  - [15. Benchmarks & Evaluation](#15-benchmarks--evaluation)
- [II. VLM](#ii-vlm)
  - [16. Vision Encoder](#16-vision-encoder)
  - [17. Architecture & Projector](#17-architecture--projector)
  - [18. Training Pipeline](#18-training-pipeline)
  - [19. Resolution Handling & Token Efficiency](#19-resolution-handling--token-efficiency)
  - [20. Hallucination & Reliability](#20-hallucination--reliability)
  - [21. Video Understanding](#21-video-understanding)
- [III. VLA](#iii-vla)
  - [22. Foundations](#22-foundations)
  - [23. Action Representation](#23-action-representation)
  - [24. Representative Models](#24-representative-models)
  - [25. Embodied AI Topics](#25-embodied-ai-topics)

---

# 0. Fundamentals

---

**Q1. `[Easy]` Language model이란 무엇인가? Autoregressive generation이란?**

Language model: 이전 token들이 주어졌을 때 다음 token의 확률 분포를 모델링하는 것. `P(x_t | x_1, ..., x_{t-1})`.

Autoregressive generation: 이전에 생성한 token들을 다시 입력으로 받아 순차적으로 다음 token을 생성하는 방식. 각 step의 출력이 다음 step의 입력이 됨. 생성이 완료될 때까지 반복함.

---

**Q2. `[Easy]` Token이란 무엇이고, vocabulary란 무엇인가?**

Token: LLM이 처리하는 기본 단위. 단어 전체, 단어의 일부(subword), 또는 단일 문자가 될 수 있음. "unhappiness" → ["un", "happi", "ness"] 처럼 분리됨.

Vocabulary: 모델이 인식할 수 있는 모든 token의 집합. 보통 32K~152K 크기. Vocabulary 외의 text는 unknown token 또는 여러 token 조합으로 처리됨.

---

**Q3. `[Easy]` Embedding이란 무엇인가?**

Token을 고차원 실수 벡터로 변환하는 것. "cat"이라는 token을 [0.2, -0.5, 0.8, ...] 같은 벡터로 표현함. 의미적으로 유사한 token은 embedding space에서 가까운 위치를 가지도록 학습됨. LLM의 입력 단계에서 token ID를 벡터로 변환하는 embedding layer가 담당함.

---

**Q4. `[Easy]` Training / Fine-tuning / Inference의 차이는?**

- **Training (Pretraining)**: 대규모 데이터로 모델 파라미터를 처음부터 학습. Gradient 계산 및 업데이트가 수행됨.
- **Fine-tuning**: 이미 학습된 pretrained model을 특정 task/domain 데이터로 추가 학습. 이미 학습된 표현을 출발점으로 삼아 효율적임.
- **Inference**: 학습 완료된 모델로 예측을 수행. Gradient 계산 없음. `torch.no_grad()` 상태로 실행됨.

---

**Q5. `[Easy]` LLM 학습에서 cross-entropy loss가 어떻게 쓰이는가?**

다음 token이 정답 token일 확률의 negative log: `L = -log P(token_t | x_1, ..., x_{t-1})`

이를 전체 시퀀스에 평균냄: `L = -1/T · Σᵢ log P(xᵢ | x_{<i})`

loss를 최소화하면 모델이 정답 다음 token에 높은 확률을 부여하도록 학습됨. SFT에서는 assistant 응답 token에만 loss를 계산함(loss masking).

---

**Q6. `[Easy]` Perplexity란 무엇이고 어떻게 해석하는가?**

언어 모델이 데이터를 얼마나 잘 예측하는지 나타내는 지표. `PPL = exp(average cross-entropy loss)`

낮을수록 좋음. PPL=10은 모델이 각 position에서 평균 10개의 equally likely한 선택지가 있다는 의미. 모델 비교에 널리 쓰이나 downstream task 성능과 항상 일치하지는 않음.

---

**Q7. `[Easy]` Softmax 함수란 무엇이고 LLM에서 어디에 쓰이는가?**

벡터를 확률 분포로 변환: `softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)`

모든 값이 0~1 사이, 합이 1. LLM에서:
1. Attention score 계산 시 QK^T 결과를 확률로 변환.
2. 최종 vocabulary distribution 생성 시 logit을 확률로 변환.
Temperature τ로 나누면 분포를 조절 가능함.

---

**Q8. `[Easy]` Gradient descent와 backpropagation이란?**

**Gradient descent**: loss를 최소화하기 위해 파라미터를 gradient 반대 방향으로 업데이트: `θ ← θ - α · ∇_θ L`

**Backpropagation**: chain rule로 loss에 대한 각 파라미터의 gradient를 출력에서 입력 방향으로 전파하는 알고리즘. `∂L/∂W = (∂L/∂y)(∂y/∂W)`. PyTorch의 `loss.backward()`가 자동으로 수행함.

---

**Q9. `[Easy]` Overfitting과 Underfitting이란?**

- **Overfitting**: 학습 데이터에 과하게 맞춰져 새 데이터에서 성능이 낮아지는 현상. Training loss↓, Validation loss↑. LLM에서 SFT 시 특정 도메인에만 overfitting하면 다른 task 성능이 저하됨(catastrophic forgetting과 연관).
- **Underfitting**: 모델이 학습 데이터조차 잘 설명하지 못하는 현상. Training loss도 높음. 모델 크기가 너무 작거나 학습이 부족할 때 발생함.

---

**Q10. `[Easy]` Transfer learning과 fine-tuning이란?**

**Transfer learning**: 대규모 데이터로 학습한 모델의 지식을 다른 task에 전이하는 패러다임.

**Fine-tuning**: pretrained model의 weight를 출발점으로 삼아 task-specific 데이터로 추가 학습. LLM pipeline에서: pretraining(일반 언어 능력) → SFT(instruction following) → alignment(human preference) 순으로 진행됨.

---

**Q11. `[Easy]` Zero-shot / Few-shot / In-context learning의 차이는?**

- **Zero-shot**: 예시 없이 task description만으로 task 수행. "다음 문장을 프랑스어로 번역해줘: Hello"
- **Few-shot**: 몇 개의 예시(demonstration)를 context에 포함하여 task 수행.
- **In-context learning (ICL)**: model weight를 업데이트하지 않고 context의 examples만으로 task를 학습하는 것처럼 보이는 현상. Few-shot의 일반화 개념임.

---

**Q12. `[Easy]` Context window란 무엇이고, 왜 제한이 있는가?**

LLM이 한 번에 처리할 수 있는 최대 token 수. Input prompt + generated output의 합이 이를 초과할 수 없음.

제한이 있는 이유: self-attention의 O(n²) 메모리·연산 복잡도. n이 커지면 메모리와 계산이 폭발적으로 증가함. GPT-4: 128K, Gemini 1.5: 1M, LLaMA-3 8B: 8K(기본).

---

**Q13. `[Easy]` System prompt란 무엇이고 user prompt와 어떻게 다른가?**

System prompt: LLM에게 역할, 행동 방식, 제약을 지정하는 초기 instruction. "You are a helpful assistant..."처럼 서비스 제공자가 설정함.

User prompt: 사용자가 실시간으로 입력하는 메시지. System prompt보다 낮은 우선순위를 가짐(instruction hierarchy). SFT 학습 시 chat template에서 `<|system|>`, `<|user|>`, `<|assistant|>` 로 구분됨.

---

**Q14. `[Easy]` Parameter와 Hyperparameter의 차이는?**

- **Parameter**: 모델이 학습을 통해 최적화하는 값. Weight 행렬과 bias 벡터. "7B 모델"은 70억 개의 learnable parameter를 의미함.
- **Hyperparameter**: 학습 전 사람이 설정하는 값. Learning rate, batch size, layer 수, hidden dimension 등. 학습으로 최적화되지 않음. 잘못 설정하면 학습이 불안정하거나 성능이 낮음.

---

**Q15. `[Easy]` Base model / Instruction model / Chat model의 차이는?**

- **Base model**: pretraining만 된 모델. 텍스트를 이어 쓰는 next-token prediction만 수행. GPT-3, LLaMA base.
- **Instruction model**: base model + SFT. Instruction을 따를 수 있음. LLaMA-3-8B-Instruct.
- **Chat model**: instruction model + alignment(RLHF/DPO). 안전하고 helpful하며 harmless함. ChatGPT, Claude, Gemini 등.

---

**Q16. `[Medium]` Batch size가 학습에 미치는 영향은?**

- **큰 batch**: gradient estimate의 분산이 낮아 안정적, GPU utilization 높음. 하지만 sharp minima에 수렴하는 경향, generalization이 낮아질 수 있음. 더 큰 learning rate와 함께 사용함(linear scaling rule).
- **작은 batch**: noisy gradient로 regularization 효과, 넓은 minima에 수렴 가능. 하지만 GPU 효율 낮고 학습 불안정.

LLM 학습에서는 gradient accumulation으로 effective batch를 키움: `effective_batch = micro_batch × accum_steps × n_GPUs`.

---

**Q17. `[Medium]` Learning rate warmup이 왜 필요하고, cosine decay와 어떻게 결합하는가?**

Warmup 필요성: 학습 초기 파라미터가 random하거나 불안정한 상태에서 갑자기 큰 LR을 적용하면 gradient가 폭발하거나 학습이 발산함. 초반에 LR을 0에서 목표값까지 점진적으로 높여 안정적인 출발을 보장함.

Cosine decay: warmup 이후 LR을 cosine 함수로 부드럽게 감소시킴. Pretraining 표준. `LR(t) = LR_max · 0.5 · (1 + cos(π · t/T))`

---

**Q18. `[Easy]` VLM이란 무엇이고, 어떤 task를 수행하는가?**

VLM(Vision-Language Model): 이미지(시각)와 텍스트(언어)를 함께 처리하는 모델. 구성: Vision Encoder + Projector + LLM.

주요 task: VQA(이미지 질문 답변), image captioning, OCR(텍스트 인식), 차트/문서 이해, 시각적 추론, 멀티모달 대화. LLaVA, Qwen2-VL, InternVL, GPT-4V 등이 대표적임.

---

**Q19. `[Easy]` VLA란 무엇이고, VLM과 어떻게 다른가?**

VLA(Vision-Language-Action model): 시각 입력과 언어 명령을 받아 로봇 action을 출력하는 모델. "테이블 위의 빨간 컵을 집어" → joint angle sequence 출력.

VLM과의 차이: VLM은 텍스트를 출력, VLA는 연속적인 robot action(joint angles, end-effector position 등)을 출력. VLM 위에 action generation module을 추가한 구조임. RT-2, π0, OpenVLA 등이 대표적임.

---

**Q20. `[Easy]` Reinforcement learning의 기본 개념은? LLM alignment에서 어떻게 쓰이는가?**

RL 기본 요소:
- **State (s)**: 환경의 현재 상태
- **Action (a)**: agent가 취할 수 있는 행동
- **Reward (r)**: action 후 환경으로부터 받는 신호
- **Policy (π)**: state에서 action을 선택하는 전략

LLM alignment에서: "응답 생성" = action, "human preference 점수" = reward, "응답 품질 개선" = policy 최적화. RLHF의 PPO가 이 framework를 직접 적용함.

---

# I. LLM

## 1. Transformer Architecture

---

**Q21. `[Medium]` Transformer의 self-attention 메커니즘을 수식과 함께 설명하고, Q/K/V 각각의 역할을 설명해줘.**

`Attention(Q, K, V) = softmax(QK^T / √d_k) · V`

- **Q (Query)**: 현재 token이 "무엇을 찾고 있는가"를 표현하는 벡터.
- **K (Key)**: 각 token이 "나는 이런 정보를 가지고 있다"를 표현하는 벡터.
- **V (Value)**: 실제로 전달할 정보를 담은 벡터.

QK^T로 모든 token 쌍의 유사도를 계산하고, √d_k로 나눠 softmax가 한 token에 쏠리는 것(gradient vanishing)을 방지함. softmax 후 V를 weighted sum하여 context-aware한 표현을 만듦.

---

**Q22. `[Easy]` Multi-Head Attention(MHA)에서 여러 head를 쓰는 이유는?**

단일 attention head는 하나의 "관점"(예: 의미적 유사성)만 학습함. 여러 head를 사용하면 문법적 의존성, 의미적 유사성, 위치적 근접성 등 다양한 관점을 병렬로 학습 가능함. 각 head의 `d_k = d_model / h`로 줄여 전체 연산량은 single-head와 동일하게 유지함.

---

**Q23. `[Medium]` GQA / MQA / MHA를 비교하고, GQA가 현대 LLM 표준이 된 이유는?**

- **MHA**: 모든 head가 독립적인 K, V 보유. KV-cache가 크고 serving 시 memory-bandwidth 병목.
- **MQA**: 모든 head가 단 1쌍의 K, V 공유. KV-cache를 h배 줄이나 성능 약간 저하.
- **GQA**: head를 G개 그룹으로 나눠 그룹 내에서 K, V 공유. MQA만큼 효율적이면서 MHA에 훨씬 가까운 성능 유지.

GQA 표준 이유: decode 단계가 memory-bandwidth-bound이므로 KV-cache 크기가 throughput의 핵심 병목. GQA는 성능 손실 없이 이를 대폭 줄임. LLaMA-2/3, Mistral, Gemma, Qwen2 채택.

---

**Q24. `[Hard]` FlashAttention의 핵심 아이디어와 왜 수치 결과가 정확히 동일한가?**

기존 attention은 n×n attention matrix를 GPU HBM에 저장해야 함. FlashAttention은 Q, K, V를 SRAM에 맞는 tile로 분할하여 계산하고, 중간 n×n 행렬을 HBM에 쓰지 않음. Online softmax trick으로 tile 단위 결과를 누적하면서 최종값을 정확히 재현함. Backward pass에 필요한 activation은 저장 대신 재계산함.

수치 동일성: online softmax가 sequential softmax와 수학적으로 동치임. 메모리 O(n²) → O(n), 속도 2~4배 향상.

---

**Q25. `[Medium]` Pre-LN과 Post-LN의 차이점과 Pre-LN이 더 안정적인 이유는?**

- **Post-LN**: `Output = LayerNorm(x + Sublayer(x))`. Residual path에 LayerNorm 포함.
- **Pre-LN**: `Output = x + Sublayer(LayerNorm(x))`. Residual path가 LayerNorm을 거치지 않음.

Pre-LN이 안정적인 이유: residual path `x`가 직접 이전 layer까지 gradient를 전달함. Post-LN에서는 깊은 네트워크에서 gradient가 각 layer의 LayerNorm을 거쳐 불안정해짐. GPT-3 이후 Pre-LN이 표준.

---

**Q26. `[Easy]` RMSNorm이 LayerNorm보다 빠른 이유는?**

LayerNorm: 평균 계산 및 빼기(centering) + RMS 정규화.
RMSNorm: centering 없이 RMS로만 정규화. `RMSNorm(x) = x / RMS(x) · γ`

평균 계산·빼기 연산을 생략하여 ~10-15% 빠름. 성능은 동등함(centering이 표현력에 크게 기여하지 않음). LLaMA, Mistral, Gemma 등 현대 LLM의 표준임.

---

**Q27. `[Medium]` SwiGLU가 기존 FFN activation(ReLU/GeLU)보다 표현력이 높은 이유는?**

기존 FFN: `FFN(x) = Activation(xW₁) · W₂`

SwiGLU: `FFN(x) = (Swish(xW₁) ⊙ (xW₃)) · W₂`

두 번째 linear projection이 gate 역할을 함. `Swish(xW₁)` 값이 작으면 `xW₃`을 억제하여 불필요한 정보를 차단함. 이 gating 메커니즘이 표현력을 높임. LLaMA, PaLM, Gemma 등이 채택.

---

**Q28. `[Easy]` Causal mask가 왜 필요한가?**

Decoder-only LLM은 "이전 token으로 다음 token을 예측"하는 CLM으로 학습함. Causal mask 없이는 position 5의 token이 position 6, 7의 미래 token을 볼 수 있어 학습 목표가 trivial해짐(답을 미리 보고 예측). `QK^T` 계산 후 softmax 이전에 upper triangular를 `-inf`로 마스킹하여 attention score를 0으로 만듦.

---

**Q29. `[Easy]` Residual connection이 없으면 어떤 문제가 생기는가?**

각 layer를 통과할 때 gradient가 layer 수에 따라 지수적으로 감소(vanishing) 또는 증가(explosion)함. 100+ layer LLM에서는 사실상 학습 불가능함. Residual connection `x + f(x)`는 backward pass에서 gradient가 직접 이전 layer로 흐르는 "highway"를 제공하여 이를 해결함.

---

**Q30. `[Medium]` Decoder-only 아키텍처가 대규모 LLM의 표준이 된 이유는?**

1. **Data efficiency**: 하나의 시퀀스에서 n-1개의 학습 예시를 한 번의 forward pass로 동시 획득(CLM).
2. **ICL**: 모든 context가 하나의 시퀀스로 이어져 in-context learning이 자연스러움.
3. **Scalability**: 단순한 구조로 대규모 확장이 용이하고 scaling law가 잘 적용됨.
4. **Flexibility**: generation, classification, embedding 등 다양한 task에 통일된 방식 적용 가능.

---

**Q31. `[Medium]` MoE에서 expert collapse란 무엇이며, load balancing loss로 어떻게 해결하는가?**

Expert collapse: routing이 특정 expert에만 token을 집중시켜 나머지 expert는 학습이 되지 않는 현상. 더 많이 사용된 expert가 더 잘 학습 → 더 많이 선택되는 positive feedback loop 형성.

Load balancing loss: `L_lb = α · Σᵢ fᵢ · pᵢ` (fᵢ: 실제 routing 비율, pᵢ: 평균 routing probability). 이를 메인 loss에 더해 모든 expert가 균등하게 사용되도록 유도함.

---

**Q32. `[Hard]` Mamba(Selective SSM)의 핵심 아이디어와 Transformer 대비 추론 시 장점은?**

기본 SSM: `h_t = A·h_{t-1} + B·x_t`, `y_t = C·h_t` (A, B, C 고정).

Mamba: A, B, C를 입력 x_t에 따라 동적으로 결정. 관련 있는 정보는 기억하고 없는 것은 잊는 selective 처리.

추론 시 장점: O(n) KV-cache 대신 O(d_state) 고정 크기 hidden state만 유지. 학습 시 parallel scan으로 병렬, 추론 시 recurrent 방식으로 O(1) 메모리. 매우 긴 시퀀스에서 메모리 병목 없음.

---

**Q33. `[Medium]` Scaling law와 emergent ability의 관계는?**

Scaling law: model size N, data D, compute C 증가 시 loss가 멱법칙으로 예측 가능하게 감소함. `L ∝ N^(-α)` 형태.

Emergent ability: scaling law로 예측되지 않다가 특정 임계값을 넘으면 갑자기 나타나는 능력. 예: few-shot learning, chain-of-thought reasoning.

관계: loss는 연속적으로 감소하지만, downstream task 성능은 불연속적으로 급증함. 다수의 sub-skill이 모두 학습되어야 작동하는 composition 효과로 해석됨.

---

**Q34. `[Hard]` In-context learning(ICL)이 왜 작동하는지에 대한 주요 가설들은?**

1. **Bayesian inference**: demonstration이 task prior를 업데이트하고 모델이 posterior로 추론.
2. **Implicit gradient descent**: forward pass 중 attention이 암묵적으로 in-context gradient descent를 수행한다는 해석.
3. **Task retrieval**: pretraining에서 비슷한 패턴을 학습했고, demonstration이 적절한 "mode"를 activate.

실용적 insight: label의 정확성보다 format과 input-output 관계 패턴이 더 중요함. Random label로도 일부 성능 유지됨.

---

**Q35. `[Hard]` Mixture of Depths(MoD)란 무엇이고, MoE와 어떻게 다른가?**

MoD: 각 layer를 모든 token이 통과하는 것이 아니라, router가 일부 token만 layer를 통과하게 하는 방법. 통과하지 않는 token은 residual bypass로 직접 다음 layer로 전달됨.

MoE와 차이: MoE는 FFN expert 중 일부만 활성화(depth 고정, width sparse). MoD는 layer 자체를 일부 token에 대해 skip(depth sparse). 결합 시(MoE + MoD) 더 큰 compute 절약 가능. 계산 비용을 동적으로 token 중요도에 따라 배분함.

---

**Q36. `[Medium]` Hybrid architecture(Jamba, Zamba)의 아이디어와 장점은?**

Mamba layer와 Transformer attention layer를 교차 배치하는 방식. 예: [Mamba × 3, Attention × 1] 패턴으로 반복.

장점:
- SSM의 효율(O(1) recurrent 추론, 긴 시퀀스 메모리 절약)
- Attention의 정확한 information retrieval 능력

Mamba만으로는 long-range exact recall이 어렵고, Attention만으로는 긴 시퀀스에서 메모리 병목. 두 방식을 결합하여 각각의 단점을 보완함.

---

## 2. Positional Encoding & Tokenization

---

**Q37. `[Medium]` RoPE의 수학적 아이디어와 extrapolation이 좋은 이유는?**

Q, K 벡터를 위치 m에 따라 회전 행렬 R_m으로 변환:
핵심 성질: `(R_m·q)^T · (R_n·k) = q^T · R_{m-n} · k`
→ attention score가 절대 위치(m, n)가 아닌 **상대 위치(m-n)에만 의존**함.

Extrapolation이 좋은 이유: 학습에 없던 긴 position에서도 상대적 거리 관계가 유지됨. 추가 파라미터 없음. LLaMA, Mistral, Qwen, Gemma의 표준.

---

**Q38. `[Hard]` RoPE context length 확장 기법들(Position Interpolation, NTK Scaling, YaRN)의 차이는?**

- **Position Interpolation**: 긴 position index를 학습 범위 내로 선형 압축. 단순하지만 짧은 context 성능 약간 저하.
- **NTK Scaling**: RoPE의 base 주파수를 조정(θ → θ × scale). Fine-tuning 없이도 어느 정도 extrapolation 가능.
- **YaRN**: 고주파 성분(짧은 거리 패턴)은 보간 없이 유지, 저주파 성분만 압축. LLaMA-3 128K context 확장에 사용됨.

---

**Q39. `[Easy]` BPE tokenizer의 동작 원리는?**

1. 모든 문자를 개별 symbol로 시작.
2. 가장 빈번한 인접 symbol 쌍을 찾아 새 symbol로 merge.
3. 원하는 vocabulary 크기에 도달할 때까지 반복.

장점: 빈도 높은 단어는 하나의 token, 희귀 단어는 subword로 분리. OOV(Out-of-Vocabulary) 문제를 자연스럽게 해결함. GPT 계열, LLaMA 사용.

---

**Q40. `[Medium]` Vocabulary 크기를 키울 때의 trade-off는?**

- **크면**: 같은 텍스트를 더 적은 token으로 표현 → context window 효율적, 다국어 지원 우수. 단, embedding table 파라미터 증가, 희귀 token 학습 어려움.
- **작으면**: 더 많은 token 필요 → context window 낭비. 학습은 안정적.

트렌드: 32K(LLaMA-2) → 128K(LLaMA-3) → 152K(Qwen2). 다국어·코드 표현 효율 향상 목적.

---

**Q41. `[Medium]` SentencePiece가 BPE와 다른 점과 다국어에 유리한 이유는?**

BPE: 공백으로 단어를 먼저 분리 후 subword merge → language-specific pre-tokenization 필요.

SentencePiece: 공백을 일반 문자로 취급하여 raw text를 직접 처리. 한국어, 중국어, 일본어처럼 공백이 없는 언어에서도 동일하게 작동함. 언어별로 별도 tokenizer를 만들 필요 없이 다국어 데이터를 한꺼번에 학습 가능. T5, LLaMA, Gemma 사용.

---

**Q42. `[Easy]` SFT 학습에서 loss masking이 반드시 필요한 이유는?**

(user: "서울의 수도는?" / assistant: "서울은 한국의 수도입니다.") 에서 loss masking 없이 학습하면 user turn에도 loss가 계산됨 → 모델이 "user처럼 질문하는 방식"도 학습.

원하는 것: "assistant처럼 응답하는 방식"만 학습. 따라서 assistant 응답 부분에만 loss 계산, user turn은 masking(-100으로 처리)해야 함.

---

**Q43. `[Medium]` Chat template이 SFT 학습에서 왜 중요한가?**

Chat template: 대화 구조(system/user/assistant 역할)를 구분하는 special token 형식. LLaMA-3: `<|start_header_id|>user<|end_header_id|>`, ChatML: `<|im_start|>user`.

중요한 이유: 모델이 어디서 응답해야 하는지(assistant turn 시작)를 학습하려면 학습 데이터와 inference 시 동일한 template을 사용해야 함. Template 불일치 시 모델이 올바르게 응답하지 못함.

---

## 3. Pretraining

---

**Q44. `[Medium]` Chinchilla scaling law의 핵심과 LLaMA가 이를 의도적으로 어긴 이유는?**

Chinchilla: compute C 고정 시 모델 크기 N과 데이터 D를 균등하게 늘리는 것이 compute-optimal. `D_opt ≈ 20 × N_opt`.

LLaMA의 의도적 이탈: training compute보다 inference 비용이 중요하다고 판단. 더 작은 모델에 더 많은 데이터(LLaMA-1 7B: 1T tokens)를 학습하면 inference 비용이 낮음. 수십억 회 inference를 고려하면 총비용 면에서 유리함.

---

**Q45. `[Medium]` Pretraining 데이터 deduplication이 왜 중요하고, MinHash+LSH는 어떻게 동작하는가?**

중요성: 중복 데이터가 많으면 특정 텍스트가 memorization되고, perplexity는 낮아도 실제 generalization 능력이 향상되지 않음.

MinHash+LSH: 각 문서를 n-gram 집합으로 표현 → MinHash로 compact signature 생성(Jaccard similarity 근사) → LSH로 유사 문서를 같은 bucket에 배치 → bucket 내에서만 정밀 비교. FineWeb, RedPajama 등에 사용됨.

---

**Q46. `[Medium]` Data mixture에서 코드 비율이 reasoning 능력에 영향을 미치는 이유는?**

코드는 구조적이고 논리적인 텍스트임. 코드 학습 시: 변수 추적, 함수 호출 순서, 조건 분기 등 명시적 논리 구조를 학습. 단계별 추론의 자연스러운 학습 데이터가 됨. 실험적으로 코드 비율을 높이면 순수 언어 추론 task에서도 성능이 향상됨. LLaMA-2, Mixtral 모두 코드 데이터에서 강점을 보임.

---

**Q47. `[Easy]` CLM(Causal Language Modeling)이 data efficient한 이유는?**

"The cat sat on the mat" 하나의 시퀀스에서 동시에 학습:
- P(sat | The cat)
- P(on | The cat sat)
- P(the | The cat sat on)

→ n-1개의 학습 예시를 하나의 forward pass에서 병렬로 얻음. Teacher forcing으로 모든 position의 loss를 동시에 계산하기 때문에 데이터 효율이 매우 높음.

---

**Q48. `[Medium]` Pretraining 데이터 quality filtering의 주요 방법들은?**

1. **Heuristic filtering**: 텍스트 길이, 특수문자 비율, 반복 비율 등 규칙 기반 필터링.
2. **Perplexity filtering**: KenLM 같은 작은 언어 모델로 perplexity를 계산하여 너무 높거나(random text) 낮은(boilerplate) 데이터를 제거.
3. **Classifier-based**: 고품질 데이터(Wikipedia, 책 등)와 저품질 데이터를 구분하는 classifier로 점수 부여.
4. **DSIR(Data Selection via Importance Resampling)**: 목표 분포(고품질)에 맞도록 데이터를 importance sampling으로 선택.

---

**Q49. `[Hard]` Scaling hypothesis와 the bitter lesson이 LLM 발전에 미친 영향은?**

**The bitter lesson(Sutton)**: 인간의 domain knowledge를 모델에 넣으려는 시도보다 scale과 general-purpose computation이 장기적으로 항상 이겼음. 결론: 학습과 탐색에 더 많은 compute를 투자하는 것이 최선.

**Scaling hypothesis**: 충분히 큰 모델에 충분한 데이터와 compute를 주면 intelligence가 창발한다는 주장. GPT-3의 성공으로 검증됨.

영향: inductive bias 줄이기(CNN 대신 ViT, RNN 대신 Transformer), 모델 크기와 데이터를 지속 확장하는 전략의 정당화.

---

**Q50. `[Medium]` Curriculum learning이 pretraining에서 어떻게 활용되는가?**

학습 초기에 더 쉬운 고품질 데이터를 제공하고 후기에 어렵거나 특수한 데이터를 제공하는 방법. LLaMA-3: 일반 학습 후 마지막 단계에 long context 데이터를 집중 투입하여 128K context를 지원. Codestral: 일반 pretraining 후 code-heavy 데이터를 집중 학습. 데이터 분포를 학습 단계에 맞게 조절하는 것이 핵심.

---

## 4. Post-Training & SFT

---

**Q51. `[Easy]` Instruction tuning에서 task 다양성이 데이터 크기보다 중요한 이유는?**

FLAN 실험에서 발견: task 수가 많을수록 unseen task generalization이 더 크게 향상됨. 동일 task 데이터를 늘리는 것보다 새로운 task를 추가하는 것이 효과적임.

이유: 다양한 task는 "instruction을 읽고 그에 맞게 응답하는" 범용 능력(meta-skill)을 학습하게 함. 특정 task에 overfitting되지 않고 instruction following 자체를 일반화함.

---

**Q52. `[Medium]` Self-Instruct와 Evol-Instruct의 아이디어 차이는?**

**Self-Instruct**: seed instruction에서 LLM이 새 instruction을 생성. Alpaca(GPT-3로 52K 생성)가 대표. 생성된 instruction이 비슷한 패턴으로 집중되고 복잡도 증가가 없음.

**Evol-Instruct(WizardLM)**: 기존 instruction을 더 복잡하게 진화시킴. In-depth evolution(더 복잡한 제약 추가), In-breadth evolution(새 유형 생성). 반복적으로 진화시켜 다양한 난이도 분포를 만듦.

---

**Q53. `[Medium]` Rejection Sampling Fine-tuning(RSF)이 단순 SFT보다 나은 이유는?**

단순 SFT는 고정 annotation 데이터에만 의존. RSF:
1. 현재 policy에서 N개 응답 생성.
2. Reward model/verifier로 best 선택.
3. Best만 SFT 데이터로 사용.

장점: 모델 자신이 생성한 데이터를 사용하여 distribution이 학습 데이터와 일치함. Iterative하게 반복하면 점진적으로 품질 향상. LLaMA-2-chat에 extensively 사용됨.

---

**Q54. `[Medium]` Knowledge distillation에서 soft label이 hard label보다 나은 이유를 "dark knowledge" 개념으로 설명해줘.**

Hard label: [0, 0, 1, 0, ...] — 정답만 1.
Soft label: teacher 확률 분포, 예: [0.03, 0.01, 0.85, 0.08, ...]

Dark knowledge: soft label에는 클래스 간 관계가 인코딩됨. "cat(85%)"을 예측할 때 "kitten(8%)"도 높다면 이 둘이 의미적으로 유사함을 student에 전달. DeepSeek-R1이 긴 thinking trace를 작은 모델에 distillation하여 reasoning 능력 전이한 것이 대표 사례.

---

**Q55. `[Medium]` RAG와 Fine-tuning 중 어떤 상황에 무엇을 선택해야 하는가?**

| 상황 | 선택 |
|------|------|
| 최신 정보가 자주 바뀜 | RAG |
| 도메인 지식이 방대하고 정확성 중요 | RAG |
| 특정 응답 style/format 학습 | Fine-tuning |
| 새로운 task 능력 학습 | Fine-tuning |
| 개인화 | Fine-tuning |

RAG: "무엇을 알고 있는가"를 동적으로 변경. Fine-tuning: "어떻게 행동하는가"를 변경. 둘을 결합하는 것이 가장 강력함.

---

**Q56. `[Medium]` Knowledge cutoff 이후 정보를 다루는 방법들은?**

1. **RAG**: 최신 문서를 검색하여 context에 포함. Knowledge cutoff에 영향 없음.
2. **Continual pretraining**: 새 데이터로 추가 pretraining. 비용이 높고 catastrophic forgetting 위험.
3. **In-context injection**: system prompt에 최신 정보를 직접 포함.
4. **Tool use**: web search tool로 실시간 정보 검색 후 LLM에 전달.

현재 일반적: RAG + tool use 결합. Continual pretraining은 분기별 주요 업데이트에만 사용.

---

## 5. Alignment

---

**Q57. `[Medium]` RLHF 전체 파이프라인을 단계별로 설명하고, 4개 모델의 역할은?**

**Step 1**: SFT로 `π_SFT` 학습.
**Step 2**: (prompt, chosen, rejected) 쌍으로 Reward Model `r_φ` 학습. `P(y_w > y_l) = σ(r_φ(x,y_w) - r_φ(x,y_l))`.
**Step 3**: PPO로 `π_θ` 최적화.
- `r_total = r_φ(x, y) - β · KL(π_θ || π_ref)`
- `π_ref`: frozen `π_SFT`, KL penalty 계산용.
- `V_ψ`: value/critic model, advantage 추정용.

4개 모델: policy, reference, reward, value. 복잡하지만 높은 성능.

---

**Q58. `[Hard]` DPO의 수식을 유도하고, RLHF 대비 핵심 장점을 설명해줘.**

RLHF optimal policy의 closed-form: `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)`

역으로 풀면: `r(x,y) = β · log(π*(y|x)/π_ref(y|x)) + const`

이를 Bradley-Terry model에 대입:
```
L_DPO = -E[log σ(β·(log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l)))]
```

장점: reward model 불필요, 2개 모델만 필요, offline 학습 가능, 구현 단순. Mistral, Zephyr, Gemma 채택.

---

**Q59. `[Medium]` GRPO가 PPO보다 메모리를 절약하는 이유와 DeepSeek-R1에서의 역할은?**

PPO는 policy, reference, reward, value model 4개가 동시에 필요함. GRPO: 각 prompt에 G개 샘플을 생성하고 그룹 내 상대적 reward로 advantage 추정. `A_i = (r_i - mean(r)) / std(r)`. Value model이 불필요하여 policy + reference 2개 모델만 필요. 메모리 ~25-50% 절약.

DeepSeek-R1: RLVR(verifiable reward)와 GRPO를 결합하여 긴 thinking trace를 생성하는 policy를 학습함.

---

**Q60. `[Medium]` RLVR이란 무엇이고, 기존 RLHF와 다른 점은?**

RLHF: reward model이 human preference를 근사 → reward hacking, Goodhart's Law 위험.

RLVR: 수학 정답 여부, 코드 실행 결과 등 자동 검증 가능한 binary reward를 직접 사용. Reward model 학습 불필요. Reward hacking이 어려움.

한계: verifiable reward가 있는 domain에서만 작동함. 수학, 코딩, 형식 검증은 가능. 글쓰기, 창작 등 주관적 task에는 여전히 RLHF/DPO 필요.

---

**Q61. `[Hard]` Reward overoptimization(Goodhart's Law)이 LLM에서 어떻게 나타나고, 어떻게 완화하는가?**

Goodhart's Law: "measure가 target이 되면 좋은 measure이기를 멈춤."

LLM 현상: PPO 학습이 오래될수록 reward score는 계속 오르지만 실제 quality는 저하. 모델이 reward model의 취약점을 학습함: 응답을 과도하게 길게, 긍정적 표현 반복, repetitive high-scoring patterns.

완화:
1. KL penalty β 조정.
2. Reward model ensemble(여러 모델의 평균).
3. 학습 중 실제 human evaluation으로 early stopping.
4. Reward model을 주기적으로 업데이트.

---

**Q62. `[Medium]` Constitutional AI의 두 단계 과정을 설명해줘.**

**Stage 1 — SL-CAI**: 모델이 harmful request에 응답 생성 → 헌법 원칙에 따라 자기 비판 → 응답 자체 수정 → (원본, 수정) 쌍으로 SFT.

**Stage 2 — RL-CAI**: AI가 두 응답 중 헌법 원칙에 더 부합하는 것 판단(RLAIF) → reward model 학습 → RLHF 진행.

장점: 사람 annotation 없이 harmless model 학습 가능. Anthropic Claude의 핵심 방법론.

---

**Q63. `[Hard]` RLAIF(AI feedback)이 RLHF보다 scalable한 이유와 한계는?**

RLHF: 인간이 preference를 직접 제공 → 비용 높고 scale 어려움. 대량 응답 품질 비교에 전문 annotator가 필요함.

RLAIF: AI(LLM)가 preference를 제공. 훨씬 저렴하고 무한 확장 가능. Gemini, Claude 등이 대규모로 사용함.

한계:
1. AI feedback의 quality가 ceiling이 됨 — teacher보다 뛰어난 student를 만들 수 없음.
2. AI annotator의 편향이 그대로 학습됨.
3. AI feedback이 인간 가치와 얼마나 일치하는지 검증 어려움.

---

**Q64. `[Hard]` Model merging(SLERP, TIES, DARE)의 아이디어와 장단점은?**

여러 fine-tuned 모델의 weight를 combining하여 각 모델의 능력을 결합하는 방법. Fine-tuning 데이터 없이 능력 조합 가능.

- **SLERP**: 두 모델을 spherical linear interpolation으로 부드럽게 interpolate. 두 모델 간의 "shortest path"로 이동함.
- **TIES**: sign conflict를 해결(다수결)하고 task vector를 merge. 여러 모델 결합에 적합.
- **DARE**: delta weight(fine-tuned - base)를 sparse random masking으로 prune 후 rescale하여 merge. 덜 중요한 delta를 제거하여 충돌 감소.

장점: 학습 비용 없이 능력 조합. 단점: 모델 간 능력 충돌 시 성능 저하, 예측이 어려움.

---

## 6. PEFT

---

**Q65. `[Medium]` LoRA의 수식과 핵심 직관을 설명하고, inference 시 latency가 없는 이유는?**

`W_new = W₀ + ΔW = W₀ + B·A`
- `B ∈ ℝ^(d×r)`, `A ∈ ℝ^(r×k)`, `r ≪ min(d, k)`
- 초기화: B=0, A~N(0,σ²) → 초기 ΔW=0 보장.

직관: fine-tuning 시 weight 변화가 low intrinsic rank를 가진다는 관찰. Task-specific adaptation은 낮은 rank의 변화만 필요함.

Inference merge: `W_final = W₀ + B·A`. 동일한 shape이므로 별도 adapter 없이 동일한 forward pass로 추론 가능. 추가 latency 없음.

---

**Q66. `[Medium]` QLoRA의 세 가지 핵심 구성 요소를 설명해줘.**

1. **NF4**: 정규분포를 따르는 weight에 최적화된 4-bit quantization. 정규분포 quantile을 균등 간격으로 커버하여 단순 uniform quantization보다 오차 작음.
2. **Double Quantization**: quantization constants 자체를 8-bit으로 재quantization. 파라미터당 ~0.37 bit 추가 절약.
3. **Paged Optimizers**: gradient checkpointing 시 메모리 spike 발생 시 optimizer state를 CPU RAM으로 page-out. GPU OOM 방지.

결과: 65B 모델 full fine-tuning ~780GB → QLoRA ~48GB. 단일 GPU에서 가능해짐.

---

**Q67. `[Medium]` LoRA rank r을 어떻게 결정하고, 크다고 항상 좋지 않은 이유는?**

선택 기준: 단순 task(분류, 도메인 adaptation) r=4~8, 복잡한 instruction following r=16~64, full fine-tuning 수준 필요 시 r=128~256.

크다고 좋지 않은 이유:
1. r이 클수록 파라미터 수 증가 → PEFT 효율성 감소.
2. 학습 데이터가 적은 경우 overfitting 위험.
3. r이 너무 크면 low-rank constraint가 없어져 catastrophic forgetting 위험이 full fine-tuning 수준으로 증가.

---

**Q68. `[Medium]` Full fine-tuning vs PEFT: 언제 무엇을 선택하는가?**

| 상황 | 선택 |
|------|------|
| GPU 메모리 제한 | PEFT (LoRA/QLoRA) |
| 빠른 실험 · 프로토타이핑 | PEFT |
| 여러 task를 하나의 base에서 serving | PEFT (task별 adapter) |
| 매우 도메인 특화 · 최대 성능 필요 | Full fine-tuning |
| 아키텍처 변경 필요 | Full fine-tuning |

실용적으로: PEFT가 대부분의 경우 full fine-tuning과 유사한 성능을 달성하면서 비용이 훨씬 낮아 기본 선택지가 됨.

---

**Q69. `[Hard]` Continual learning에서 catastrophic forgetting이 왜 발생하고, 주요 완화 방법들은?**

원인: neural network의 gradient update는 새 task에 맞게 weight를 변경하는데, 이것이 이전 task를 위한 weight 설정을 덮어씀. 새 task의 loss가 이전 task 성능에 무관하게 최적화되기 때문.

완화:
1. **EWC(Elastic Weight Consolidation)**: Fisher information matrix로 이전 task에 중요한 파라미터를 식별하고 변화에 penalty 부여.
2. **Replay buffer**: 이전 task 데이터 일부를 새 task 학습에 혼합.
3. **LoRA**: weight 변화를 low-rank space로 제한하여 암묵적 regularization.
4. **PackedBERT/LLaMA approach**: 새 task 데이터를 이전 데이터와 항상 혼합.

---

## 7. Training Stability & Optimization

---

**Q70. `[Medium]` Gradient clipping이 왜 direction은 유지하고 magnitude만 줄이는가?**

`g ← g × τ / ||g||` (||g|| > τ인 경우)

Direction을 유지하는 이유: gradient direction은 loss가 감소하는 방향이므로 변경하면 최적화 방향이 틀려짐. Magnitude만 줄이면 같은 방향으로 더 작은 step을 밟아 폭발적 업데이트를 방지하면서 학습 지속. 일부 gradient를 0으로 clamp하면 해당 파라미터가 전혀 업데이트되지 않아 불균일한 학습이 발생함.

---

**Q71. `[Medium]` BF16이 FP16보다 LLM 학습에 적합한 이유와, FP8의 장단점은?**

| Format | Exponent | Mantissa | Range |
|--------|---------|----------|-------|
| FP32 | 8 | 23 | ±3.4×10^38 |
| BF16 | 8 | 7 | FP32와 동일 |
| FP16 | 5 | 10 | ±65504 |

BF16: FP32와 동일한 exponent range → overflow 없음, dynamic loss scaling 불필요.
FP16: exponent가 작아 overflow 위험 → dynamic loss scaling 필요.

FP8: H100 네이티브 지원, BF16 대비 2배 속도. DeepSeek-V3가 FP8 학습으로 비용 절감. 단, activation outlier에 민감하고 학습 안정성 관리 더 복잡함.

---

**Q72. `[Medium]` Gradient checkpointing의 memory-compute trade-off를 설명해줘.**

일반 학습: backward pass를 위해 모든 intermediate activation 저장. 메모리 O(n_layers).
Gradient checkpointing: activation을 저장하지 않고 backward 시 필요할 때 재계산. 저장되는 activation이 O(√n_layers)로 줄어듦.

Trade-off: 메모리를 약 절반으로 줄이는 대신 forward pass를 ~33% 더 수행. 메모리 제약이 병목일 때(긴 시퀀스, 큰 batch) 매우 유용함.

---

## 8. Distributed Training

---

**Q73. `[Hard]` Ring-AllReduce가 GPU 수에 무관하게 통신 효율이 유지되는 이유는?**

각 GPU는 이웃 GPU하고만 통신함. 두 단계:
1. **Scatter-Reduce**: gradient를 N chunk로 분할. Ring 방향으로 N-1번 전송. 각 GPU가 자신의 chunk에 대한 전체 합을 가짐.
2. **AllGather**: 완성된 chunk를 ring 방향으로 N-1번 전송. 모든 GPU가 averaged gradient를 가짐.

각 GPU의 총 전송량 ≈ `2 × gradient_size`. GPU 수 N이 아무리 커도 각 GPU의 통신량은 거의 일정함. DDP는 backward와 통신을 overlap하여 추가 효율화함.

---

**Q74. `[Hard]` Tensor Parallel, Pipeline Parallel, Sequence Parallel의 차이점은?**

- **TP**: weight matrix를 column/row로 분할. 레이어 내 분산. 매 레이어마다 AllReduce 필요 → NVLink가 있는 같은 node 내 GPU에서만 효율적. 보통 node 내 8 GPU에 적용.
- **PP**: 레이어를 순서대로 여러 GPU에 분산. Stage 간 activation만 전송 → 느린 inter-node도 허용. Bubble 문제 존재. 보통 node 간 적용.
- **SP**: 매우 긴 시퀀스를 여러 GPU에 분산. Ring Attention으로 각 GPU가 local window를 처리.

대규모 학습: TP × PP × DP의 3D parallelism 결합.

---

**Q75. `[Hard]` ZeRO Stage 1/2/3의 차이와 Stage 3의 단점은?**

| Stage | 분산 대상 | 메모리 절약 |
|-------|----------|----------|
| 1 | Optimizer state | ~4배 |
| 2 | + Gradient | ~8배 |
| 3 | + Parameters | ~GPU수 배 |

Stage 3 단점: forward/backward 시 파라미터를 AllGather로 모아야 함. 통신 overhead가 크고 구현이 복잡함. 작은 모델에는 비효율적. ZeRO-2가 대부분의 경우 좋은 balance를 제공함.

---

**Q76. `[Hard]` Pipeline bubble의 정의와 1F1B schedule이 이를 어떻게 개선하는가?**

Pipeline bubble: pipeline이 채워지거나 비워지는 동안 일부 GPU stage가 idle인 시간.

Naive schedule bubble ratio: `(p-1) / m` (p: stage 수, m: micro-batch 수)

**1F1B schedule**: 각 micro-batch에 대해 forward 1번, backward 1번을 번갈아 실행. Steady-state에서 idle이 없어짐. Bubble ratio: `(p-1) / (m + p - 1)` → m이 크면 bubble 무시할 수준으로 줄어듦.

---

**Q77. `[Medium]` FSDP와 ZeRO-3의 차이점은?**

FSDP는 ZeRO-3의 PyTorch 공식 구현. 개념적으로 동일: 파라미터, gradient, optimizer state를 모두 GPU 간 분산.

FSDP 장점: PyTorch native이므로 추가 dependency 없음. `torch.distributed`와 자연스럽게 통합. HuggingFace Trainer, torchtitan이 기본 지원. DeepSpeed ZeRO보다 설정이 단순하고 생태계 통합이 좋음.

---

## 9. Inference Efficiency

---

**Q78. `[Medium]` KV-cache의 메모리 크기를 계산하는 방법은?**

`KV-cache = 2 × n_layers × n_kv_heads × d_head × seq_len × bytes_per_element`

LLaMA-3 8B 예시 (32 layers, 8 KV heads, 128 d_head, BF16=2bytes):
- seq_len=8K: `2 × 32 × 8 × 128 × 8192 × 2 ≈ 0.54 GB` per request
- seq_len=128K: ≈ 8.4 GB per request

GQA로 n_kv_heads를 줄이면 비례하게 줄어듦. 많은 concurrent request 처리 시 KV-cache가 핵심 병목임.

---

**Q79. `[Hard]` Speculative decoding의 원리를 설명하고, target model과 동일한 분포를 보장하는 이유는?**

원리:
1. Draft model이 K개 token을 빠르게 생성.
2. Target model이 K+1 token을 한 번의 forward pass로 병렬 검증.
3. Accept/reject: `p_target ≥ p_draft` → accept. 아니면 `p_target/p_draft` 확률로 accept.
4. Reject 시: `max(0, p_target - p_draft)`를 normalize하여 수정된 token 샘플링.

분포 보장: accept/reject + 수정 샘플링의 결합 분포가 정확히 `p_target`과 동일함(rejection sampling의 수학적 성질). 2~3배 throughput 향상.

---

**Q80. `[Medium]` PagedAttention이 기존 방식 대비 메모리 효율이 높은 이유는?**

기존: 각 request에 최대 context length만큼 KV-cache를 연속 메모리 블록으로 pre-allocate. 실제 생성이 짧으면 대부분 낭비됨. Fragmentation 심각.

PagedAttention: KV-cache를 고정 크기 block으로 관리. Logical block → Physical block 매핑 테이블 유지. 필요한 만큼만 할당, non-contiguous 배치 가능. Fragmentation < 4%로 줄어들어 동일 메모리에서 2-4배 많은 concurrent request 처리 가능.

---

**Q81. `[Medium]` Continuous batching이 static batching보다 GPU utilization이 높은 이유는?**

Static batching: batch 내 모든 request가 완료될 때까지 새 request 추가 불가. 짧은 request가 완료되어도 긴 request를 기다리며 GPU slot 낭비.

Continuous batching: 매 token generation step마다 completed request 제거, waiting request 추가. GPU slot이 항상 채워진 상태 유지. Throughput 3-5배 향상. vLLM, TGI가 구현.

---

**Q82. `[Medium]` Chunked prefill이 TTFT 개선에 어떻게 도움이 되는가?**

문제: 긴 prompt의 prefill은 compute-intensive하여 다른 request의 decode를 blocking함.

Chunked Prefill: 긴 prompt를 청크(예: 512 tokens)로 분할. 각 iteration에서 prefill chunk와 decode batch를 함께 처리. 긴 prompt가 다른 request를 blocking하지 않음.

TTFT 개선: 새 request가 청크 단위로 빨리 prefill 시작 가능. 대신 TPOT은 약간 증가할 수 있는 trade-off 존재.

---

**Q83. `[Medium]` Prefix caching(RadixAttention)이란 무엇이고, 어떤 서비스에서 특히 효과적인가?**

아이디어: 동일한 prefix(예: system prompt)의 KV-cache를 request 간에 재사용. Radix tree 자료구조로 공통 prefix를 효율적으로 관리함.

효과적인 서비스:
- 동일한 긴 system prompt를 가진 서비스(예: RAG with fixed context, 코드 리뷰 서비스).
- Multi-turn 대화에서 이전 대화 내용의 KV-cache 재사용.
- Few-shot examples가 동일한 경우.

SGLang의 RadixAttention이 구현함. 반복적인 공통 prefix가 많은 서비스에서 TTFT를 크게 줄임.

---

## 10. Quantization

---

**Q84. `[Hard]` AWQ와 GPTQ의 차이점과 어떤 상황에 무엇을 사용하는가?**

**AWQ**: activation 크기가 큰 salient weight channel을 per-channel scale로 보호. 빠른 calibration(수 분). llama.cpp, Ollama와 잘 호환됨.

**GPTQ**: Hessian 기반으로 weight를 순서대로 양자화하면서 오차를 나머지 weight에 분산 보상(OBQ). Calibration 시간이 AWQ보다 길지만 더 정밀한 오차 보정.

일반적으로: 빠른 배포에는 AWQ, 최대 quality가 필요하면 GPTQ. 단, 최신 benchmark에서 두 방법의 차이가 줄어드는 추세임.

---

**Q85. `[Hard]` KV-cache quantization에서 Key가 Value보다 quantization에 더 민감한 이유는?**

Key: `Q·K^T` 계산에서 query와 내적됨. Quantization error가 attention score 전체에 영향을 미쳐 attention pattern이 변함. Outlier가 있는 key dimension이 quantize되면 softmax 분포가 크게 왜곡됨.

Value: attention probability로 weighted sum될 때 사용. Attention pattern이 이미 결정된 후이므로 quantization error가 상대적으로 덜 영향을 미침.

실용적 접근: Key는 INT8 유지, Value만 INT4로 quantize하는 asymmetric 방식이 효과적임.

---

**Q86. `[Medium]` PTQ(Post-Training Quantization)와 QAT(Quantization-Aware Training)의 차이와 LLM에서 PTQ가 주류인 이유는?**

- **PTQ**: 학습 완료된 모델을 재학습 없이 양자화. Calibration dataset으로 scale factor 결정. 빠르지만 정밀도 손실이 있음.
- **QAT**: quantization을 시뮬레이션하면서 처음부터 학습. 더 높은 품질이지만 전체 학습 비용이 필요함.

LLM에서 PTQ 주류인 이유: 수백~수천 GPU-day의 학습 비용을 QAT를 위해 다시 쓰기 어려움. AWQ, GPTQ 같은 advanced PTQ 방법들이 QAT에 근접한 품질을 달성함. 새 모델이 지속 출시되는 환경에서 빠른 quantization이 필수.

---

**Q87. `[Medium]` Weight-only quantization이 특히 decode 속도를 높이는 이유는?**

Decode 단계는 memory-bandwidth-bound: 매 step마다 KV-cache와 weight matrix를 HBM에서 읽는 것이 병목. Arithmetic 연산은 GPU core가 충분히 빠름.

Weight-only INT4/INT8: weight를 낮은 precision으로 저장 → HBM에서 읽는 양이 줄어듦 → bandwidth 병목 완화. 추론 시 FP16으로 dequantize 후 연산하므로 precision 손실 최소화. Prefill(compute-bound)에서는 효과가 적고, decode(memory-bound)에서 효과가 큼.

---

## 11. Decoding Strategies

---

**Q88. `[Easy]` Temperature가 LLM generation에서 어떤 역할을 하는가?**

Softmax 전에 logits를 T로 나눔: `softmax(logits / T)`

- **T < 1**: 분포가 뾰족해짐 → 높은 확률 token이 더 강조, 결정론적 경향. 코딩·수학에 적합.
- **T = 1**: 원래 분포 그대로.
- **T > 1**: 분포가 평탄해짐 → 다양성 증가, 창의적이지만 coherence 낮아질 수 있음. 창작 글쓰기에 적합.

T=0은 greedy decoding(argmax)과 동일함.

---

**Q89. `[Medium]` Top-p(nucleus) sampling이 top-k보다 adaptive한 이유는?**

Top-k: 항상 상위 k개 token에서 샘플링. 분포 형태 무관.

Top-p: 누적 확률이 p에 도달할 때까지 token을 추가.
- 분포가 뾰족하면 → 2~3개 token만으로 p 달성 → 보수적 선택.
- 분포가 평탄하면 → 많은 token이 필요 → 다양성 허용.

자동으로 분포 형태에 adaptive하게 조정됨. Temperature + top-p 조합(top-p=0.9, temp=0.7)이 일반적임.

---

**Q90. `[Medium]` Constrained decoding이 항상 valid JSON을 보장하는 메커니즘은?**

JSON 문법을 FSM으로 컴파일. 각 step:
1. 현재 FSM 상태 추적(object key 기대 중, value 기대 중 등).
2. 현재 상태에서 valid한 다음 token 집합 계산.
3. Invalid token의 logit을 `-∞`로 설정.
4. 남은 valid token에서 정상적으로 샘플링.

LLM의 generation 능력을 유지하면서 구조적 제약을 강제함. Outlines, SGLang, Guidance가 구현.

---

**Q91. `[Medium]` Beam search가 open-ended generation에 부적합한 이유는?**

Beam search는 높은 log probability 시퀀스를 탐색함:
1. **Repetition**: 반복적인 구문이 높은 확률을 가지는 경향.
2. **Generic**: 가장 "안전한" 평균적 표현이 선택됨.
3. **Low diversity**: 상위 k개 beam이 유사한 시퀀스로 수렴.

번역·요약(정확성 중요)에는 유리하나, 대화·창작에서는 단조롭고 boring한 응답 생성. Sampling-based 방법이 더 자연스럽고 다양한 응답을 생성함.

---

## 12. Long Context & RAG

---

**Q92. `[Medium]` "Lost in the Middle" 현상이란 무엇이고, RAG에서 어떻게 완화하는가?**

발견(Liu et al., 2023): 여러 문서를 context로 제공할 때 context의 앞부분과 뒷부분 정보는 잘 활용하지만 중간은 잘 활용하지 못하는 현상.

RAG 완화:
1. 가장 relevant한 chunk를 context 맨 앞이나 맨 뒤에 배치.
2. Reranking으로 top-k만 선택하여 총 context 최소화.
3. Recursive retrieval로 핵심 정보를 반복 배치.

---

**Q93. `[Medium]` RAG pipeline에서 bi-encoder와 cross-encoder를 두 단계로 사용하는 이유는?**

**Bi-encoder**: query와 document를 각각 독립 인코딩 → dot product로 점수. Document embedding을 미리 계산하여 vector DB에 저장. 수백만 문서에서 수ms 내 top-100 검색. 단, query-document 간 deep interaction 없음.

**Cross-encoder**: query + document를 함께 입력하여 interaction 포착. 더 정확하지만 모든 쌍을 실시간 계산해야 해서 느림.

두 단계: bi-encoder로 빠르게 후보 좁히기 → cross-encoder로 정밀 reranking. 속도와 정확도를 모두 확보.

---

**Q94. `[Easy]` Chunking 전략들(fixed-size, semantic, recursive)의 차이는?**

- **Fixed-size**: 고정 token 수로 분할(예: 512 tokens, 50 overlap). 단순하지만 문장 중간에 잘릴 수 있음.
- **Sentence/Paragraph-aware**: 문장·단락 경계를 존중. 의미적으로 완전한 chunk를 만들지만 크기 불균일.
- **Semantic**: 인접 문장의 embedding 유사도로 주제가 바뀌는 지점에서 분할. 의미적으로 응집되지만 computation 비용 높음.
- **Recursive**: 큰 단위로 나누고 너무 크면 재귀적으로 분할. LangChain의 RecursiveCharacterTextSplitter가 대표.

---

**Q95. `[Medium]` HyDE(Hypothetical Document Embedding)의 아이디어와 한계는?**

문제: 사용자 query와 답변 document의 embedding space 표현이 다를 수 있음. 짧은 query와 긴 document는 서로 멀리 있을 수 있음.

HyDE: LLM이 query에 대한 가상의 답변을 먼저 생성 → 가상 답변을 embedding → 이 embedding으로 vector DB 검색. 실제 document와 유사한 형태의 text를 query로 사용하므로 embedding space에서 더 가까워짐.

한계: LLM이 hallucination이 포함된 가상 답변을 생성할 수 있음. Hallucination이 포함된 query로 잘못된 document를 검색할 위험.

---

**Q96. `[Medium]` RAG에서 reranking이 왜 중요하고, 어떤 경우에 reranking이 도움이 되지 않는가?**

Reranking이 중요한 이유: bi-encoder는 query-document interaction을 독립적으로 처리하여 fine-grained relevance를 놓칠 수 있음. Cross-encoder reranking으로 정밀한 재정렬.

Reranking이 도움 안 되는 경우:
1. Top-1 문서가 이미 정확히 올바른 경우 — reranking overhead만 추가.
2. Retrieval 자체가 너무 부정확하여 top-100에도 정답이 없는 경우.
3. Reranker의 도메인 expertise가 retrieval과 동일 수준인 경우.

일반적으로: 높은 정확도 요구 시 reranking은 거의 항상 도움이 됨.

---

## 13. Reasoning & Test-Time Compute

---

**Q97. `[Medium]` Chain-of-Thought(CoT)가 왜 큰 모델에서만 효과적인가?**

Transformer는 constant-depth 계산 그래프 — 한 번의 forward pass로 할 수 있는 계산 깊이가 제한됨. CoT는 각 생성 step을 추가 계산 step으로 활용함.

작은 모델에서 효과 없는 이유: 중간 추론 단계 자체를 올바르게 생성하려면 기반 능력이 필요함. 기반 능력이 부족하면 중간 단계가 틀려 최종 답도 틀림. ~100B+ 파라미터에서 emergent ability로 나타남.

---

**Q98. `[Medium]` Self-consistency가 단일 CoT보다 성능이 좋은 이유와 Best-of-N과의 차이는?**

**Self-Consistency**: 동일 질문에 여러 CoT 경로를 샘플링하고 최종 답에 대해 majority vote. Reward model 없이 voting으로 선택. 수학/코드처럼 답이 명확한 경우 적합.

**Best-of-N**: N개 응답 생성 후 reward model로 best 선택. 답 외에도 응답 quality를 종합 평가. Reward model이 필요하지만 더 유연한 quality 기준 적용 가능.

---

**Q99. `[Hard]` PRM과 ORM의 차이점과 PRM 데이터 수집이 어려운 이유는?**

**ORM**: 최종 답만 정답 여부로 평가. 수집 쉬움. 단점: 틀린 추론 과정으로 우연히 맞은 경우를 positive로 처리.

**PRM**: 각 추론 step마다 점수 부여. 오류 위치 정확 파악, 잘못된 경로 조기 차단. Tree search와 결합 시 효과적.

어려운 이유: 수학 문제 각 step에 전문가 annotation 필요. 100 step 풀이면 100번의 annotation. ORM은 최종 답 정답 여부만 확인. 데이터 수집 비용이 ORM의 수십~수백 배.

---

**Q100. `[Hard]` Test-time compute scaling의 핵심 메커니즘과 practical implication은?**

핵심: "작은 모델 + 많은 inference compute ≈ 큰 모델 + 적은 compute"

메커니즘:
- Best-of-N: N이 클수록 성공 확률 `P = 1 - (1-p)^N` 증가.
- MCTS: 더 많은 rollout으로 더 좋은 경로 탐색.
- Long CoT: 더 긴 thinking으로 복잡한 문제 해결.

Practical implication:
1. 문제 난이도에 따라 inference compute를 동적 조절 가능.
2. 같은 모델을 low-latency mode / high-accuracy mode로 dual-mode 운영 가능.
3. Compute를 training time에서 inference time으로 이동하는 패러다임 전환.

---

**Q101. `[Hard]` Long CoT 학습에서 나타난 "aha moment"란 무엇이고 왜 중요한가?**

DeepSeek-R1 학습 중 관찰: RLVR 학습 과정에서 아무도 명시적으로 학습시키지 않았음에도 모델이 "Wait, let me reconsider..." 같은 자기 수정(self-verification) 행동을 spontaneously 학습함.

중요성:
1. Reasoning 능력이 단순 패턴 모방이 아닌 RL을 통해 genuine하게 창발될 수 있음을 보여줌.
2. 학습 데이터에 없던 meta-reasoning 전략이 reward signal만으로 발생함.
3. o1/o3 계열 모델의 핵심 능력의 근거가 됨.

---

**Q102. `[Hard]` Long CoT 학습에서 length hacking 문제란 무엇이고, 어떻게 완화하는가?**

문제: RLVR + Long CoT 학습에서 모델이 reward와 무관하게 응답을 과도하게 길게 만드는 reward hacking 현상. "더 길게 생각하면 reward가 높다"는 패턴을 학습하여 불필요한 반복, 무의미한 중간 단계를 삽입함.

완화:
1. **Length penalty**: 응답 길이에 비례한 penalty를 reward에 추가.
2. **Efficiency reward**: 짧고 정확한 해법에 더 높은 reward.
3. **Thinking budget**: 최대 thinking token 수를 제한.
4. **Correlation 모니터링**: thinking 길이와 reward 간 correlation이 높아지면 학습 중단.

---

## 14. Safety

---

**Q103. `[Medium]` LLM hallucination의 원인과 완화 방법을 설명해줘.**

원인:
1. Training data의 상관관계를 인과관계로 오인(통계적 패턴 학습).
2. Confidence calibration 부재.
3. LLM의 "가장 그럴듯한 다음 token" 생성 경향.
4. Knowledge cutoff 이후 정보 요구 시 추정으로 생성.

완화: RAG(검색된 사실 정보로 grounding), RLHF(사실적 정확성 reward 포함), calibration training(불확실한 경우 "모른다"고 표현), citation 강제.

---

**Q104. `[Medium]` Jailbreak 공격의 주요 유형들과 방어 방법은?**

| 유형 | 설명 | 방어 |
|------|------|------|
| Role-playing | "DAN" 등 제한 없는 AI 역할 강제 | Adversarial training |
| Obfuscation | Base64, 역방향으로 유해 내용 숨김 | Input/output decode 후 검사 |
| Multi-turn | 점진적으로 가이드라인 우회 | Conversation-level safety classifier |
| Translation | 안전 가드가 약한 언어로 질문 | 다국어 safety training |
| Adversarial suffix | 최적화된 suffix로 harmful 응답 유도 | Adversarial training, perturb detection |

---

**Q105. `[Medium]` Prompt injection이 RAG와 agent 시스템에서 특히 위험한 이유는?**

Prompt injection: 외부 입력(웹페이지, 문서)에 숨겨진 명령으로 LLM 행동을 조작하는 공격.

RAG 위험: 검색된 외부 문서가 context에 그대로 들어감. 악성 웹페이지가 "Ignore previous instructions..." 명령을 숨겨두면 모델이 이를 따를 수 있음.

Agent 위험: agent는 실제 액션(파일 읽기, API 호출, 이메일 전송)을 수행함. 악성 사이트가 agent를 조작하면 실제 피해 발생 가능.

방어: instruction hierarchy 명확화(system > user > external data), delimiter로 외부 데이터 구분, output validation.

---

**Q106. `[Medium]` Calibration이란 무엇이고, ECE를 어떻게 해석하는가?**

Calibration: 모델의 confidence가 실제 정확도와 일치하는 정도. 잘 calibrated된 모델에서 "90% 확신"이라면 실제로 90%의 경우 맞아야 함.

`ECE = Σ_bins |acc(B_m) - conf(B_m)| × |B_m| / n`

0에 가까울수록 좋음. LLM 문제: RLHF 후 모델이 자신의 답에 과도하게 confident해지는 경향(overconfidence). Temperature scaling, label smoothing으로 완화함.

---

**Q107. `[Hard]` Red teaming의 목적과 Automated red teaming이 동작하는 방식은?**

Red teaming: 모델의 취약점을 찾기 위해 의도적으로 공격적 입력을 시도하는 체계적 평가. 모델 출시 전 수행됨.

Automated red teaming 루프:
1. **Attacker LLM**: harmful 응답을 유도하는 adversarial prompt를 생성.
2. **Target LLM**: 해당 prompt에 응답.
3. **Judge LLM**: 응답이 유해한지 평가.
4. Attacker가 성공한 prompt를 기반으로 더 효과적인 새 prompt를 생성.
5. 반복하여 취약점 탐색.

발견된 취약점은 adversarial training 데이터로 사용하여 모델을 강화함.

---

## 15. Benchmarks & Evaluation

---

**Q108. `[Medium]` Benchmark contamination을 탐지하는 방법들과 contamination 방지 전략은?**

탐지:
1. **n-gram overlap**: test set의 n-gram(7~13-gram)이 training data에 존재하는지 확인.
2. **Min-k% Prob**: 모델이 test example에서 가장 낮은 확률 k% token의 평균 log prob 측정. Memorized text는 이 값이 높음.
3. **Canary string**: 학습 전 특수 문자열을 데이터에 삽입.

방지 전략:
- Time-based filtering: 모델 학습 cutoff 이후 문제만 사용(LiveCodeBench).
- Procedurally generated: 수학 문제를 템플릿으로 생성.
- Private test set: benchmark를 공개하지 않고 제출 기반 평가.

---

**Q109. `[Medium]` LLM-as-a-Judge의 주요 편향들과 완화 방법은?**

| 편향 | 설명 | 완화 |
|------|------|------|
| Position bias | 먼저 나온 응답 선호 | A>B, B>A 순서 모두 평가 |
| Verbosity bias | 더 긴 응답 선호 | 길이 무관 rubric 제공 |
| Self-enhancement | 자신과 유사한 스타일 선호 | 여러 모델 judge ensemble |
| Sycophancy | 사용자 의견 추종 | 의견 없이 중립적 판단 요청 |

---

**Q110. `[Hard]` Arena-style evaluation(Chatbot Arena)이 static benchmark보다 좋은 점과 한계는?**

장점:
1. 실제 사용자 질문으로 평가 → contamination 없음.
2. 다양하고 예측 불가능한 질문 분포.
3. 비교 방식(A vs B)으로 절대 점수의 calibration 문제 없음.
4. Elo rating으로 지속 업데이트 가능.

한계:
1. 느림 — 수천 번의 human comparison 필요.
2. 비용이 높음.
3. 특정 task(코딩, 수학)에서의 강점/약점을 구분 못함.
4. 인기 있는 모델이 더 많이 평가받는 선택 편향.
5. 모델 출시 직후 성능이 과대/과소 평가될 수 있음.

---


---

# II. VLM

## 16. Vision Encoder

---

**Q111. `[Easy]` ViT의 patch embedding 과정을 설명하고, patch size가 성능에 미치는 영향은?**

과정:
1. `H×W×C` 이미지를 `P×P` patch로 분할 → `N = (H/P)×(W/P)` 개.
2. 각 patch를 flatten: `P²×C` 벡터.
3. Linear projection: `P²×C → D_model`.
4. Learnable positional embedding 추가.
5. Transformer에 token sequence로 입력.

Patch size 영향:
- **작은 P(14)**: 더 많은 token(256+), 세밀한 표현. OCR, fine-grained에 유리. 연산 비용 높음.
- **큰 P(32)**: 적은 token(49), 빠르지만 세밀한 패턴 손실.

---

**Q112. `[Medium]` CLIP contrastive learning의 핵심과 왜 일반적 시각 표현을 만드는가?**

CLIP: 수억 개 이미지-텍스트 쌍에서 InfoNCE loss로 학습.
`L = -log(exp(sim(vᵢ, tᵢ)/τ) / Σⱼ exp(sim(vᵢ, tⱼ)/τ))`

같은 쌍은 embedding space에서 가깝게, 다른 쌍은 멀게. 배치 내 N²-N개 negative 쌍을 동시 활용. 큰 batch size(32K+)가 필요.

일반적 표현 이유: ImageNet label(1000개)보다 훨씬 풍부한 텍스트 설명이 supervision으로 작용. 다양한 개념과 관계를 semantic space에서 학습하여 zero-shot transfer가 잘 됨.

---

**Q113. `[Medium]` SigLIP이 CLIP보다 나은 점과 분산 학습에 유리한 이유는?**

CLIP 문제: softmax 기반 InfoNCE는 배치 내 모든 negative 쌍이 denominator에 기여 → 큰 batch(32K+) 필요 → 분산 학습 시 전체 batch를 gather해야 함.

SigLIP: sigmoid loss `L = -Σᵢⱼ log(σ(zᵢⱼ · yᵢⱼ))`. 각 쌍을 독립적으로 binary classification으로 처리. 큰 batch 불필요, GPU 간 gather 없이 local batch로 학습 가능. 성능도 CLIP보다 우수. LLaVA-NeXT, InternVL, PaLiGemma 채택.

---

**Q114. `[Medium]` Vision encoder를 frozen할 때와 fine-tuning할 때의 차이와 적합한 상황은?**

**Frozen encoder**: CLIP/SigLIP의 일반적 시각 표현 보존. Catastrophic forgetting 없음. Stage 1 alignment에서 표준. 단, task-specific visual feature 포착 한계.

**Fine-tuning**: 특정 VLM task에 맞는 visual feature 학습. OCR, 의료 이미지 등 domain-specific task에서 성능 향상. 단, catastrophic forgetting 위험, 높은 학습 비용.

실용적 접근: Stage 2에서 매우 낮은 LR(1e-6)로 fine-tuning하거나 LoRA 적용. 대규모 multimodal 데이터가 충분할 때만 전체 fine-tuning 권장.

---

**Q115. `[Medium]` CNN과 ViT를 vision encoder로 비교하고, VLM에서 ViT가 표준이 된 이유는?**

**CNN**: locality bias 덕분에 소규모 데이터에 강함. 장거리 의존성 포착이 약하고 대규모 확장성 제한.

**ViT**: 이미지를 patch sequence로 처리. Global self-attention으로 어떤 두 patch 간 관계도 첫 layer에서 계산 가능. 대규모 데이터에서 CNN을 능가.

VLM에서 ViT 표준 이유:
1. CLIP/SigLIP이 ViT 기반으로 학습되어 고품질 pretrained weight 활용 가능.
2. Transformer 구조가 LLM과 통일되어 연구·구현이 용이.
3. 대규모 데이터에서 scalability가 우수함.

---

## 17. Architecture & Projector

---

**Q116. `[Medium]` VLM의 다양한 projector 유형들(Linear, MLP, Q-Former, Perceiver)을 비교해줘.**

| Projector | 입력 tokens | 출력 tokens | 특징 |
|-----------|-----------|-----------|------|
| Linear | N_patches | N_patches | 단순, 비선형성 없음 |
| MLP (LLaVA-1.5) | N_patches | N_patches | 비선형성 추가, 현재 주류 |
| Q-Former (BLIP-2) | N_patches | 32 고정 | 강력한 압축, 세밀 정보 손실 가능 |
| Perceiver Resampler | N_patches | M 고정 | 유연한 pooling, multi-image 유리 |

현재 트렌드: MLP projector + AnyRes 조합이 성능·효율 균형으로 LLaVA-NeXT, InternVL 등 주류.

---

**Q117. `[Hard]` Unified Multimodal Transformer란 무엇이고, 기존 VLM과 어떻게 다른가?**

기존 VLM: Vision Encoder → Projector → LLM. 시각과 언어 처리가 분리됨.

Unified Multimodal Transformer: 별도 vision encoder 없이 하나의 Transformer가 이미지 patch와 텍스트 token을 처음부터 함께 처리. 예: Fuyu, Chameleon.

장점:
- 더 깊은 cross-modal interaction.
- 구조가 단순하고 pipeline 없음.
- 이미지 생성과 이해를 동일 모델에서 처리 가능.

단점:
- 대규모 paired 데이터 없이 CLIP 같은 강력한 pretrained encoder를 활용 불가.
- 학습이 더 어렵고 불안정할 수 있음.

---

## 18. Training Pipeline

---

**Q118. `[Easy]` VLM 2-stage training에서 Stage 1과 Stage 2에 사용하는 데이터의 차이는?**

**Stage 1 (Alignment pretraining)**:
- 목적: visual feature를 language space에 align.
- 설정: Vision Encoder + LLM frozen, projector만 학습.
- 데이터: 대규모 이미지-캡션 쌍(CC3M, LAION, ShareGPT4V). 수백만 샘플. 간단한 1문장 캡션.

**Stage 2 (Instruction fine-tuning)**:
- 목적: diverse multimodal instruction following.
- 설정: Projector + LLM 함께 학습(또는 LLaMA LoRA).
- 데이터: VQA, OCR, 차트, 멀티턴 대화 등 다양한 task. 수십만~수백만 샘플.

---

**Q119. `[Medium]` LLaVA 계열 모델의 발전 과정(1.0→1.5→NeXT)에서 각 버전의 핵심 개선점은?**

**LLaVA 1.0**: CLIP ViT-L + Vicuna + Linear projector. Simple projector로도 효과적임을 입증. 2-stage training 방법론 확립.

**LLaVA-1.5**: Linear → MLP projector(비선형성). CLIP ViT-L/336으로 해상도 업그레이드. Instruction data 품질 향상. 단순 변경으로 SOTA 달성.

**LLaVA-NeXT(1.6)**: AnyRes 도입으로 고해상도 처리(최대 1344×1344). GPT-4V 생성 고품질 instruction data. OCR, 문서, 수학 이미지 이해에서 큰 향상.

---

**Q120. `[Medium]` VLM instruction tuning 데이터 구성 시 중요한 고려사항들은?**

1. **Task diversity**: VQA, OCR, captioning, reasoning 등 다양한 task를 포함하여 generalization 향상.
2. **Data quality**: GPT-4V로 생성한 데이터 vs 자동 template 데이터의 품질 차이가 큼.
3. **Negative examples**: "이 이미지에 X가 없다"는 부정 응답을 학습하지 않으면 hallucination 증가.
4. **Text data mixing**: pure text instruction data를 혼합하여 LLM의 언어 능력 유지(catastrophic forgetting 방지).
5. **Task balance**: 특정 task가 너무 많으면 다른 task에 overfit.

---

## 19. Resolution Handling & Token Efficiency

---

**Q121. `[Medium]` AnyRes(Dynamic Tiling)의 동작 방식과 token 수를 controllable하게 유지하는 방법은?**

동작:
1. 이미지 aspect ratio를 고려하여 미리 정의된 grid 세트(1×1, 2×1, 1×2, 2×2, ...)에서 적합한 것 선택.
2. 이미지를 grid에 맞게 tile로 분할. 각 tile을 ViT 입력 크기로 리사이즈 후 독립 인코딩.
3. Global thumbnail(전체 이미지 리사이즈)과 함께 LLM에 전달.
4. Tile 간 구분을 `\n` token으로 표시하여 spatial 구조 전달.

Token 수 관리: max grid를 제한하여 최대 token 수 controllable. 2×2 grid: 4 tile × 256 tokens + 256 global = 1280 tokens.

---

**Q122. `[Medium]` Visual token pruning이 왜 필요하고, 주요 방법들은 무엇인가?**

필요성: 고해상도 이미지 처리 시 수천 개 visual token이 생성되어 LLM context window를 빠르게 소진하고 attention 연산 비용이 O(n²)로 증가함.

주요 방법:
1. **Attention score 기반**: LLM attention score가 낮은 visual token 제거. 모델이 집중하지 않는 token 제거.
2. **Similarity 기반**: 인접하거나 유사한 visual token을 병합. 배경처럼 uniform한 영역에서 효과적.
3. **Learned compression**: projector 단에서 learned pooling으로 token 수 압축. Q-Former 방식이 극단적 예.
4. **FastV**: early layer의 attention score로 redundant token을 early exit.

---

**Q123. `[Medium]` Qwen2-VL의 Dynamic Resolution 방식이 AnyRes와 어떻게 다른가?**

AnyRes: 미리 정의된 grid preset에서 선택 → aspect ratio가 약간 왜곡될 수 있고 grid 크기에 제한.

Qwen2-VL Dynamic Resolution:
- 이미지 원본 해상도를 거의 그대로 입력(최대 해상도까지).
- Aspect ratio 왜곡 없이 처리.
- M-RoPE로 각 patch에 (height, width) 2D position을 직접 인코딩.
- Token 수는 이미지 해상도에 비례하여 동적으로 결정.

장점: 어떤 aspect ratio든 자연스럽게 처리, preset grid 제한 없음. 단점: 고해상도에서 token 수가 매우 많아질 수 있음.

---

**Q124. `[Hard]` VLM에서 OCR과 Document Understanding이 어려운 이유와 해결 방법은?**

어려운 이유:
1. 텍스트가 매우 작아 고해상도 처리 없이는 판독 불가(영수증, 표, 소형 텍스트).
2. 텍스트의 spatial layout이 의미를 가짐(표, 수식, 코드 블록).
3. 다양한 폰트, 방향, 언어, 손글씨 처리 필요.
4. 기존 ViT의 입력 해상도(224~336px)로는 작은 텍스트 판독 불가.

해결:
1. **AnyRes/Dynamic Resolution**: 고해상도 tile로 분할하여 인코딩.
2. **OCR-specialized pretraining**: OCR 데이터를 대규모로 포함한 pretraining.
3. **Layout encoding**: 텍스트의 2D position 정보를 별도 인코딩(LayoutLM 방식).
4. **Qwen2-VL style**: 원본 해상도를 거의 그대로 처리.

---

## 20. Hallucination & Reliability

---

**Q125. `[Medium]` VLM hallucination의 주요 유형과 POPE 벤치마크가 어떻게 평가하는가?**

유형:
- **Object hallucination**: 이미지에 없는 물체를 있다고 설명(가장 흔함).
- **Attribute hallucination**: 물체의 색상, 크기, 위치를 잘못 설명.
- **Relationship hallucination**: 물체 간 관계를 잘못 설명.

원인: 강한 언어 prior(LLM이 "식탁 이미지 → 음식이 있다"를 통계적으로 학습)가 시각 정보보다 강하게 작동함.

POPE: "이미지에 [물체]가 있나요?" yes/no로 묻는 방식. Random / Popular / Adversarial 세 가지 난이도. Accuracy, Precision, Recall, F1로 평가.

---

**Q126. `[Hard]` VLM의 sycophancy 문제란 무엇이고, VLM에서 특히 문제가 되는 상황은?**

Sycophancy: 모델이 사용자의 의견이나 기대에 맞게 틀린 답변도 긍정하는 현상.

VLM에서: "이 이미지에 고양이가 있죠?" 라고 물으면 실제 이미지에 없어도 "네, 고양이가 있습니다"라고 답하는 경향.

특히 문제가 되는 상황:
1. 의료 이미지 분석에서 의사의 의견을 과도하게 따름.
2. 문서 검토에서 사용자가 "이 조항은 문제없죠?"라고 물을 때.
3. 이미지 기반 fact-checking에서 사용자의 기대에 부합하는 방향으로 왜곡.

완화: factual accuracy를 RLHF reward로 명시적 포함, contrastive decoding(이미지 유무에 따른 응답 차이 amplify).

---

**Q127. `[Hard]` VLM의 visual grounding과 spatial reasoning이 어떻게 다르고, 현재 VLM의 약점은?**

**Visual grounding**: 텍스트로 설명된 물체/영역을 이미지에서 localize하는 task. "빨간 컵" → bounding box 출력. 모델이 언어와 이미지 위치를 연결해야 함.

**Spatial reasoning**: 물체 간의 공간적 관계를 추론하는 task. "컵이 접시 위에 있는가?", "두 물체 중 더 왼쪽에 있는 것은?" 등.

현재 VLM 약점:
1. CLIP 표현이 high-level semantic에 강하지만 fine-grained pixel-level 위치 이해가 약함.
2. Left/right, above/below 같은 상대적 위치 개념의 정확도가 인간보다 낮음.
3. 물체 counting에서 오류가 빈번함.

해결: Grounding-specific pretraining(GLIP, Grounding DINO), SAM과 결합, bounding box prediction을 token으로 학습.

---

## 21. Video Understanding

---

**Q128. `[Medium]` Video LLM에서 "token explosion" 문제를 어떻게 해결하는가?**

문제: 30fps, 1분 = 1800 frames × 256 tokens/frame = 460K tokens. LLM context window 순식간에 초과.

해결:
1. **Frame subsampling**: 1fps, 0.5fps으로 frame 수 축소. 단순하지만 빠른 동작 누락.
2. **Temporal pooling**: 인접 frame feature를 평균 또는 aggregation. LLaVA-Video 사용.
3. **Token compression**: frame당 visual token pruning.
4. **Q-Former on video**: 전체 video feature를 고정 수의 token으로 압축.
5. **Selective sampling**: 장면 변화 감지로 중요한 frame만 선택.

---

**Q129. `[Hard]` M-RoPE(Multimodal RoPE)란 무엇이고, 비디오 이해에서 왜 중요한가?**

기존 RoPE: 텍스트는 1D position만 인코딩. 비디오 frame의 patch들이 1D로 나열되면 시간적·공간적 위치가 뒤섞임.

M-RoPE: 각 token에 3개의 독립적 position encoding 적용: temporal(T), height(H), width(W).
- 텍스트 token: T=H=W=같은 값.
- 이미지 patch: T=고정, H/W=grid position.
- 비디오 patch: T=frame index, H/W=patch position.

중요성: 시간적 순서를 position encoding에 자연스럽게 반영. 다양한 해상도·프레임 수를 동일 framework로 처리. "무엇이 먼저 일어났는가" 같은 temporal reasoning 능력 향상. Qwen2-VL이 도입하여 video 이해에서 오픈소스 SOTA 달성.

---

# III. VLA

## 22. Foundations

---

**Q130. `[Medium]` Behavior Cloning의 covariate shift 문제를 수학적으로 설명해줘.**

BC는 전문가 policy의 state 분포 `d^{π_expert}(s)`에서 (s, a) 쌍을 수집하여 학습함. 배포 시 student policy는 자신의 분포 `d^{π_student}(s)`에서 실행됨.

Covariate shift: `d^{π_student}(s) ≠ d^{π_expert}(s)`. T step 후 오류가 `O(T² × ε)`으로 누적됨. 작은 초기 오류 ε이 긴 task에서 치명적으로 증폭됨. 이것이 long-horizon robot task에서 BC가 실패하는 근본 원인임.

---

**Q131. `[Medium]` DAgger가 covariate shift를 이론적으로 해결하는 방법과 현실적 한계는?**

아이디어: student policy가 실제로 방문하는 state에서 올바른 action을 학습.

알고리즘:
1. πᵢ으로 환경 rollout.
2. 방문한 모든 state에서 전문가가 correct action 제공.
3. 데이터셋 D에 추가. D 전체로 πᵢ₊₁ 학습.
4. 반복. → `d^{πᵢ}(s)` → `d^{π_expert}(s)` 수렴.

현실적 한계: 매 iteration마다 전문가가 실제 로봇 실행 중 개입 필요 → 비용 높음. 위험한 state에서 전문가 개입이 필요한 안전 문제.

---

**Q132. `[Hard]` Offline RL과 Online RL의 차이점, CQL이 distribution shift를 어떻게 해결하는가?**

**Online RL**: 환경과 상호작용하며 실시간 데이터 수집. 항상 최신 policy 분포를 따름.
**Offline RL**: 고정 데이터셋으로만 학습. 환경 상호작용 없음.

Offline RL 핵심 문제: 데이터셋에 없는 (s, a)에 대해 Q-value를 과대 추정. 이를 따르면 나쁜 action을 취함.

**CQL**: `L_CQL = L_Bellman + α · (E_{μ}[Q(s,a)] - E_{D}[Q(s,a)])`

데이터셋에 없는 action의 Q-value를 낮추고, 데이터셋 action의 Q-value를 높임. 보수적인 Q-function 추정으로 distribution shift로 인한 과대 추정을 방지함.

---

**Q133. `[Medium]` Sim-to-Real gap의 원인과 domain randomization이 왜 효과적인가?**

Sim-to-Real gap 원인:
1. **Visual gap**: 시뮬레이터 rendering vs 실제 카메라 이미지.
2. **Physics gap**: 마찰, 탄성, 중력 모델의 부정확성.
3. **Dynamics gap**: 실제 로봇의 모터 특성, 백래쉬, 진동.

Domain Randomization: 물체 색상/질감, 조명, 마찰 계수(±50%), 카메라 위치, 배경 등을 학습 시 랜덤하게 변화. 충분히 넓은 sim 분포를 커버하면 실제 환경은 그 분포 내의 한 점. Policy가 특정 sim 파라미터에 overfitting되지 않고 robust한 feature를 학습하도록 강제함.

---

**Q134. `[Easy]` Imitation learning이란 무엇인가? Behavior Cloning과의 관계는?**

Imitation learning: 전문가의 시연(demonstration)을 관찰하여 policy를 학습하는 방법. Reward function을 명시적으로 설계하지 않아도 됨.

Behavior Cloning(BC): imitation learning의 가장 단순한 형태. 전문가 시연에서 (observation, action) 쌍을 수집하여 지도학습으로 `π(action|observation)`을 학습. 간단하지만 covariate shift 문제가 있음.

---

## 23. Action Representation

---

**Q135. `[Medium]` Continuous Regression과 Discrete Tokenization의 trade-off는?**

**Discrete Tokenization**: 연속 action을 N개 bin으로 양자화, LLM vocabulary에 추가. Autoregressive generation. 기존 LLM 아키텍처 재사용 가능, language와 action이 동일 space. 단, 정밀도 손실, 느린 생성, 고차원 action space에서 비효율.

**Continuous Regression**: LLM의 마지막 hidden state에 regression head 추가. 연속 action vector 직접 예측. 빠르고 정밀함. 단, MSE loss로는 unimodal 분포만 표현 가능. Multimodal action distribution에서 실패.

---

**Q136. `[Hard]` Diffusion Policy가 multimodal action distribution을 포착할 수 있는 이유는?**

문제 상황: 장애물 양쪽으로 돌아갈 수 있을 때, 왼쪽(-30°)과 오른쪽(+30°) 모두 valid.

Continuous Regression(MSE): "(-30°) + (+30°) = 평균(0°)" 예측 → invalid trajectory.

Diffusion Policy: Gaussian noise `x_T`에서 시작하여 반복적 denoising으로 action `x_0` 생성. 초기 noise에 따라 다른 mode로 수렴 가능. Score function `∇ log p(x_t|condition)`이 분포의 여러 mode를 모두 표현함. Noise sample A → 왼쪽 trajectory, Noise sample B → 오른쪽 trajectory.

---

**Q137. `[Hard]` Flow Matching이 Diffusion Policy보다 빠른 이유와 π0에서의 역할은?**

Diffusion: Gaussian → data를 구불구불한 경로로 이동. 보통 100~1000 denoising steps 필요.

Flow Matching: `dx/dt = v_θ(x_t, t, condition)`. 직선 경로(straight-path ODE)로 이동. 학습 목표: `v_θ(x_t, t) = x_1 - x_0` (직선 속도 벡터). 직선 경로이므로 10~50 steps로 충분.

π0에서의 역할: PaliGemma VLM(언어 이해·계획) + Flow matching action expert(고주파 정밀 제어). VLM token과 action expert가 cross-attention으로 연결됨. Dexterous bimanual manipulation에서 최고 성능 달성.

---

**Q138. `[Medium]` Chunked action prediction이 high-frequency control에서 유리한 이유는?**

기존: 매 step마다 LLM forward pass 필요. LLM inference latency로 25~100Hz 제어에 부적합.

Chunked prediction: 한 번의 LLM forward pass로 H개(예: 10~16개) future action을 동시 예측. Receding horizon: H개 중 일부를 실행하고 새로운 observation으로 다시 H개 예측.

장점:
1. Latency: 10 action씩 예측하면 10배 낮은 LLM 호출 빈도.
2. Temporal consistency: 연속된 action이 함께 예측되어 부드러운 trajectory.
3. Background에서 다음 chunk를 계산하면서 현재 chunk를 실행하는 overlap 가능.

---

## 24. Representative Models

---

**Q139. `[Medium]` RT-1과 RT-2의 핵심 차이점과 RT-2의 "semantic generalization"이란?**

**RT-1 (2022)**: EfficientNet + TokenLearner + causal Transformer. 처음부터 로봇 데이터만으로 학습. 700개 이상 task에서 97% 성공률. 단, 웹 지식과 단절. 새 물체·명령 조합에 generalization 어려움.

**RT-2 (2023)**: 기존 VLM(PaLI-X 55B, PaLM-E 562B) backbone 재사용. Robot action을 텍스트 token으로 vocabulary에 추가. 웹 + 로봇 데이터 co-finetuning.

Semantic generalization: "음료를 아픈 사람 옆에 가져다 줘" → 의학 상식 활용. 학습에 없던 물체·명령 조합에 generalization. Web-scale pretraining 지식이 robot으로 전이됨.

---

**Q140. `[Medium]` Octo가 "generalist policy"로 설계된 방식과 open-source 기여점은?**

Octo 설계: Modular architecture (각 modality별 tokenizer + Transformer backbone + task-specific head). Action head로 diffusion을 사용하여 multimodal distribution 포착. Open X-Embodiment 데이터(다양한 로봇, 다양한 task)로 학습. 새 로봇에 fine-tuning 가능하게 설계.

Open-source 기여: model, code, training recipe, data pipeline 모두 공개. 연구자들이 자체 로봇에 fine-tuning할 수 있는 foundation 제공. Cross-embodiment learning 연구를 커뮤니티에 개방.

---

**Q141. `[Medium]` OpenVLA가 오픈소스 VLA로 중요한 이유는?**

구조: SigLIP + LLaMA-2 7B. Action tokenization: RT-2와 동일한 discrete autoregressive 방식.

중요성:
1. RT-2(55B+)의 성능을 7B 모델로 달성. 접근 가능한 크기.
2. Model weight, training code, data 모두 공개.
3. 연구자들이 48GB GPU에서 학습·fine-tuning 가능.
4. VLA 연구의 democratization.
5. 특정 로봇 설정에 LoRA fine-tuning으로 specialization 가능.

---

**Q142. `[Hard]` π0의 dual-network 구조가 왜 효과적인가?**

구조:
1. **PaliGemma VLM**: 언어 이해, 계획, 상황 판단 담당.
2. **Flow matching action expert**: 고주파(25Hz+) 정밀 제어 신호 생성.

왜 분리하는가:
- VLM은 언어 이해·계획에 특화되어 있고 고주파 제어에 부적합.
- 정밀한 manipulation은 물리적 dynamics에 대한 연속 action이 필요.
- Flow matching은 직선 경로로 빠른 action 생성 가능.
- VLM token과 action expert가 cross-attention으로 연결되어 language conditioning 유지.

결과: dexterous bimanual manipulation(두 손 협력 조작)에서 RT-2, OpenVLA 대비 크게 향상.

---

## 25. Embodied AI Topics

---

**Q143. `[Medium]` Hierarchical planning이 왜 long-horizon robot task에 필요한가?**

"커피를 만들어라" 같은 task는 수십~수백 개 primitive action으로 구성됨. End-to-end로 학습하면: credit assignment 문제 심각, 오류 누적, 데이터 수집 극도로 어려움.

계층 구조:
- **High-level**: LLM이 "커피 만들기" → [컵 집기, 커피 머신 이동, 버튼 누르기, ...] 분해.
- **Mid-level**: 각 primitive skill 실행.
- **Low-level**: joint angle level 제어(10~100Hz).

SayCan: LLM plan + affordance model(현재 환경에서 실행 가능한지 판단)을 결합하여 현실적인 계획을 생성함.

---

**Q144. `[Medium]` Open X-Embodiment의 의의와 cross-embodiment learning의 도전은?**

의의: 30+ 기관이 다양한 로봇 데이터를 RLDS 표준 형식으로 통합. RT-X: 이 데이터로 학습한 모델이 특정 로봇 전용 모델보다 strong baseline 제공. 로봇 학습의 데이터 부족 문제를 collaboration으로 해결하는 시도.

Cross-embodiment 도전:
1. Action space 불일치(6-DOF arm vs 7-DOF arm vs bipedal robot).
2. Observation 불일치(다른 카메라 각도, 해상도).
3. Task 불일치(로봇마다 다른 skill set).

해결 방향: language로 task 명세, action을 relative end-effector space로 정규화, embodiment token으로 로봇 유형 조건화.

---

**Q145. `[Medium]` Language-conditioned manipulation에서 generalization 수준들은?**

1. **Object-level**: 같은 task를 다른 물체(다른 색상, 크기)에 적용.
2. **Compositional**: 각각 학습된 concept들을 새롭게 조합.
3. **Zero-shot**: 학습에 없던 물체를 VLM 지식으로 이해.
4. **Analogical**: 유추로 새 task 이해. "붓으로 하는 것처럼 솔로 청소해."

RT-2, π0 등 VLM 기반 VLA가 zero-shot과 compositional generalization에서 특히 강함.

---

**Q146. `[Hard]` VLA에서 world model이란 무엇이고, 왜 중요한가?**

World model: 환경의 내부 모델을 학습하여 미래 state/observation을 예측하는 모델. Model-based RL의 핵심.

VLA에서 중요성:
1. 실제 로봇 실행 없이 policy를 simulation에서 평가 가능.
2. Planning 시 "이 action을 하면 어떻게 될까"를 상상하여 최적 action 선택.
3. 데이터 효율성 향상: real robot 데이터 없이도 world model에서 simulated experience 생성.

현재 연구: UniSim, GROOT 등이 video prediction 기반 world model로 robot policy 학습. LLM 자체를 "텍스트 세계에 대한 world model"로 해석하는 관점도 있음.

---

**Q147. `[Hard]` Dexterous manipulation의 핵심 도전들과 현재 접근 방법들은?**

핵심 도전:
1. **High DoF control**: 손가락 제어는 20+ DoF. 일반 robot arm(6-7 DoF)보다 훨씬 복잡.
2. **Contact-rich dynamics**: 물체와의 접촉, 마찰, deformation 모델링 어려움.
3. **Perception**: 손가락과 물체의 정밀한 위치 추정(mm 수준).
4. **데이터 수집**: 고품질 dexterous demonstration 수집이 매우 어려움.

현재 접근:
1. **ALOHA/ACT**: bimanual arm으로 high-quality demonstration 수집, chunked action prediction.
2. **Diffusion Policy**: multimodal action distribution 포착으로 다양한 grasp 전략 처리.
3. **π0 flow matching**: 연속 action을 빠른 ODE solver로 생성.
4. **Tactile sensing**: 손끝에 tactile sensor 추가하여 contact 정보 활용.

---

**Q148. `[Hard]` Neurosymbolic approach와 end-to-end VLA의 차이점과 장단점은?**

**End-to-end VLA**: 입력에서 action까지 하나의 neural network.
- 장점: 복잡한 feature를 자동 학습, human engineering 불필요.
- 단점: interpretability 낮음, failure mode 불명확, 디버깅 어려움.

**Neurosymbolic**: perception(neural) + planning(symbolic AI)의 조합.
- 장점: interpretable, 각 module 독립 수정 가능, symbolic planning의 논리적 보장.
- 단점: perception-planning 인터페이스 설계 어려움, symbolic planning이 연속 action space에 부적합.

**현재 절충안**: LLM-based hierarchical planner(symbolic-like) + neural low-level controller. "LLM이 high-level plan을 natural language로 생성 → VLA가 각 step을 실행"하는 구조가 두 방식의 장점을 결합함.

---

**Q149. `[Medium]` VLA 데이터 수집 방법들과 각각의 trade-off는?**

| 방법 | 설명 | 장점 | 단점 |
|------|------|------|------|
| Kinesthetic teaching | 사람이 로봇 팔을 직접 움직여 시연 | 자연스럽고 정밀 | 느리고 비용 높음 |
| VR teleoperation | VR controller로 원격 제어 | 비교적 빠름 | 장비 비용, latency |
| Scripted primitives | 규칙 기반 자동 시연 | 완전 자동화 | 복잡한 task 불가 |
| RL in sim | 시뮬레이터에서 RL로 생성 | 무한 데이터 | Sim-to-real gap |
| Human video | 인터넷 human video 활용 | 대규모 | Action label 없음, embodiment gap |

최근 트렌드: ALOHA 같은 leader-follower teleoperation과 VR 조합이 π0 등에 사용됨.

---

**Q150. `[Hard]` 로봇 학습에서 reward function 설계의 어려움과 imitation learning이 대안이 되는 이유는?**

Reward function 설계 어려움:
1. **Reward shaping**: sparse reward("task 완료")만으로는 학습이 매우 느림. Dense reward는 설계가 어렵고 reward hacking 위험.
2. **Unintended behavior**: reward를 최대화하는 예상치 못한 방법을 학습(예: 컵을 들지 않고 컵을 밀어 목표 위치로 이동).
3. **Task 다양성**: 수백 가지 task마다 reward function을 설계하는 것은 현실적으로 불가능.
4. **Contact-rich manipulation**: 성공/실패가 명확하지 않은 grasp 품질 평가 어려움.

Imitation learning 대안인 이유:
1. 전문가 시연만 있으면 reward function 불필요.
2. 인간의 직관적인 시연으로 복잡한 behavior를 자연스럽게 전달.
3. RT-2, π0 등이 대규모 human demonstration으로 성공한 것이 이를 입증.

