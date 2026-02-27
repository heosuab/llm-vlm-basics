# LLM / VLM / VLA Topics Summary

> 특정 세부 주제들에 대한 간결한 정리.

---

## 목차

### LLM
1. [Transformer Block 기본 구성](#1-transformer-block-기본-구성)
2. [Attention 효율화 (MQA / GQA / FlashAttention)](#2-attention-효율화-mqa--gqa--flashattention)
3. [Positional Encoding & Context 확장](#3-positional-encoding--context-확장)
4. [Tokenization](#4-tokenization)
5. [Model Architecture 유형 (Decoder-only / MoE)](#5-model-architecture-유형-decoder-only--moe)
6. [Alternative Architecture (Mamba / RWKV)](#6-alternative-architecture-mamba--rwkv)
7. [Pretraining: Objective & Data Pipeline](#7-pretraining-objective--data-pipeline)
8. [Post-Training & SFT](#8-post-training--sft)
9. [Alignment (RLHF / DPO / GRPO)](#9-alignment-rlhf--dpo--grpo)
10. [PEFT (Parameter-Efficient Fine-Tuning)](#10-peft-parameter-efficient-fine-tuning)
11. [Training Stability & Optimization Tricks](#11-training-stability--optimization-tricks)
12. [Distributed Training](#12-distributed-training)
13. [Decoding Strategies](#13-decoding-strategies)
14. [Inference Efficiency (KV-Cache / PagedAttention / Speculative Decoding)](#14-inference-efficiency-kv-cache--pagedattention--speculative-decoding)
15. [Quantization](#15-quantization)
16. [Serving Frameworks & Metrics](#16-serving-frameworks--metrics)
17. [Long Context & RAG](#17-long-context--rag)
18. [Reasoning & Test-Time Compute Scaling](#18-reasoning--test-time-compute-scaling)
19. [LLM Safety](#19-llm-safety)
20. [Benchmarks & Evaluation Issues](#20-benchmarks--evaluation-issues)

### VLM
21. [Vision Encoder (CNN vs ViT, CLIP, SigLIP)](#21-vision-encoder-cnn-vs-vit-clip-siglip)
22. [VLM Architecture & Projector 설계](#22-vlm-architecture--projector-설계)
23. [Image Resolution Handling](#23-image-resolution-handling)
24. [VLM Training Pipeline (2-Stage)](#24-vlm-training-pipeline-2-stage)
25. [VLM Hallucination](#25-vlm-hallucination)
26. [Catastrophic Forgetting in VLM](#26-catastrophic-forgetting-in-vlm)
27. [Video Understanding in VLM](#27-video-understanding-in-vlm)

### VLA
28. [VLA 기초 (BC, Covariate Shift, Offline RL)](#28-vla-기초-bc-covariate-shift-offline-rl)
29. [Action Representation](#29-action-representation)
30. [대표 VLA 모델들](#30-대표-vla-모델들)
31. [Embodied AI 핵심 주제들](#31-embodied-ai-핵심-주제들)

---

# LLM

## 1. Transformer Block 기본 구성

**왜 중요한가**: LLM의 모든 능력은 Transformer block이 쌓인 구조에서 나옴. 각 컴포넌트가 왜 있는지 이해해야 architecture 선택과 최적화를 제대로 할 수 있음.

**핵심 구조 흐름**: `입력 → (LayerNorm → Self-Attention → Residual) → (LayerNorm → FFN → Residual) → 출력`

**주요 컴포넌트**:

- **Self-Attention**: 각 token이 다른 token과의 관련도를 `QK^T / √d_k`로 계산하고 softmax로 weighted sum. "은행(bank)" 같은 단어의 의미를 문맥(river 등)에서 파악하는 핵심 메커니즘임.
- **Multi-Head Attention (MHA)**: 하나의 관점 대신 h개의 head가 서로 다른 관점(문법·의미·위치 등)을 병렬 학습하는 방법. 각 head가 `d_k = d_model / h`로 독립 학습하고 마지막에 concat 후 W_O로 투영함.
- **Causal Mask**: Decoder-only LM에서 future token을 못 보게 `QK^T` 이후 softmax 이전에 upper triangular를 -inf로 처리하는 방법. 이게 있어야 "다음 token 예측" 학습이 가능함.
- **Residual Connection**: 각 sublayer의 입출력을 `x + Sublayer(x)`로 더하는 shortcut. 없으면 100+ layer에서 gradient가 소실되어 학습 불가능함.
- **Pre-LN**: LayerNorm을 sublayer 이전에 적용(`x + Sublayer(LN(x))`). 기존 Post-LN보다 학습이 훨씬 안정적이어서 GPT-3 이후 표준이 됨.
- **RMSNorm**: 평균 제거 없이 RMS로만 정규화하는 방법. LayerNorm보다 ~10-15% 빠르고 성능은 동등하여 LLaMA, Mistral 등 현대 LLM 표준임.
- **FFN & SwiGLU**: `d_ff = 4 × d_model`로 확장하는 MLP. 현대 LLM은 단순 activation 대신 SwiGLU(`(W₁x) ⊙ Swish(W₃x)`)로 gating하여 불필요한 정보를 억제함.

---

## 2. Attention 효율화 (MQA / GQA / FlashAttention)

**왜 필요한가**: MHA의 O(n²) 메모리·연산 복잡도와 큰 KV-cache가 긴 시퀀스·대규모 serving의 핵심 병목임.

**두 가지 문제**:
1. **메모리 (KV-cache)**: 모든 head가 독립 K, V를 가지면 KV-cache가 너무 큼 → MQA/GQA로 해결
2. **속도 (IO 병목)**: n×n attention matrix를 GPU HBM에 반복 쓰고 읽어서 느림 → FlashAttention으로 해결

**주요 방법**:

- **MQA (Multi-Query Attention)**: 모든 head가 단 1개의 K, V를 공유. KV-cache를 h배 줄이지만 성능이 약간 저하됨. PaLM, Falcon 사용함.
- **GQA (Grouped Query Attention)**: head를 G개 그룹으로 묶고 그룹 내에서 K, V 공유. MQA만큼 빠르면서 MHA에 가까운 품질을 달성하여 현재 decoder-only LLM의 표준임. LLaMA-2/3, Mistral, Gemma, Qwen2 등이 사용함.
- **FlashAttention v1/v2/v3**: attention 결과는 동일하게 유지하면서 GPU IO를 최적화하는 방법. Q, K, V를 SRAM에 맞는 tile로 나눠서 n×n 중간 행렬을 HBM에 저장하지 않고 재계산(recomputation)함. 메모리 O(n²) → O(n), 속도 2~4배 향상. 현재 LLM 학습·추론의 표준 구현임.
- **Sliding Window Attention**: 각 token이 주변 w개 token과만 attend. O(n·w) 복잡도로 긴 시퀀스 처리. Mistral 7B 사용함.

---

## 3. Positional Encoding & Context 확장

**왜 필요한가**: Transformer의 self-attention은 position을 모르므로 "The dog bit the man"과 "The man bit the dog"을 구분 못 함. 위치 정보를 별도로 주입해야 함.

**추가 문제**: 학습 길이 이상의 긴 context에서도 잘 동작해야 함(extrapolation). 이게 어렵기 때문에 다양한 기법이 필요함.

**주요 방법**:

- **Sinusoidal PE**: sin/cos 파형으로 위치 인코딩. 학습 없이 계산 가능하나 extrapolation이 나쁨.
- **Learned PE**: 각 위치 임베딩을 학습. BERT, GPT-2 사용. 최대 길이 이상으로는 extrapolation 불가함.
- **RoPE (Rotary Position Embedding)**: Q, K를 위치에 따라 회전시켜 attention score가 상대 위치(m-n)에만 의존하게 만드는 방법. Extrapolation이 좋고 추가 파라미터 없어 LLaMA, Mistral, Qwen 등 현대 LLM 표준임.
- **ALiBi**: attention score에 거리에 비례한 선형 페널티를 더하는 방법. 파라미터 없이 구현이 단순하고 extrapolation이 강함. MPT, BLOOM 사용함.
- **Context Length 확장 기법들**: 학습 후에도 context window를 늘리는 방법들. NTK Scaling(RoPE base 주파수 조정), Position Interpolation(긴 위치를 학습 범위 내로 압축), YaRN(고주파/저주파 성분을 다르게 처리)이 대표적임.
- **M-RoPE (Multimodal RoPE)**: 텍스트는 1D position, 이미지 patch는 (time, height, width) 3D position을 독립 인코딩하는 방법. Qwen2-VL이 도입하여 비디오 이해에도 활용함.

---

## 4. Tokenization

**왜 중요한가**: 같은 텍스트라도 tokenization에 따라 token 수가 달라지고, 이게 context window 효율·다국어 지원·학습 품질에 직접 영향을 줌.

**핵심 tradeoff**: vocab이 크면 적은 token으로 표현 가능(context 효율적)하지만 embedding 파라미터가 커지고 희귀어 처리가 복잡해짐.

**주요 방법**:

- **BPE (Byte Pair Encoding)**: 가장 빈번한 인접 symbol 쌍을 반복 merge하여 vocabulary 구성. GPT 계열, LLaMA 사용함.
- **WordPiece**: likelihood를 가장 높이는 쌍을 merge. BERT 사용하며 subword 앞에 `##`를 붙임.
- **SentencePiece**: 공백 없이 raw text를 직접 처리하여 언어에 독립적으로 동작. 한국어·중국어 등 공백 없는 언어에 특히 유리함. T5, LLaMA, Gemma 사용함.
- **Vocabulary 크기 추세**: 32K(LLaMA-2) → 128K(LLaMA-3) → 152K(Qwen2)로 확대. 큰 vocab은 같은 텍스트를 더 적은 token으로 표현하여 context window를 더 효율적으로 활용함.
- **Chat Template**: SFT 학습 시 대화 구조를 구분하는 special token 형식. LLaMA-3의 `<|start_header_id|>`, ChatML의 `<|im_start|>` 등이 있음. SFT 데이터는 반드시 이 형식을 따라야 함.
- **SFT Loss Masking**: user turn에는 loss를 주지 않고 assistant 응답에만 loss를 계산하는 방법. 이래야 "user 질문 형태 모방"이 아닌 "assistant 응답 방식"을 학습함.

---

## 5. Model Architecture 유형 (Decoder-only / MoE)

**왜 중요한가**: 어떤 아키텍처를 선택하느냐에 따라 pretraining objective, 적합한 task, 효율성이 달라짐.

**주요 유형**:

- **Encoder-only (BERT 계열)**: 양방향 attention + MLM으로 사전학습. 풍부한 표현 학습에 유리하지만 autoregressive generation에 부적합함. 분류·임베딩 task에 사용함.
- **Decoder-only (GPT/LLaMA 계열)**: Causal attention + CLM으로 사전학습. 생성에 자연스럽고 ICL이 용이하며 scalability가 우수하여 LLM의 현재 표준임. 하나의 시퀀스에서 n개의 학습 예시를 동시에 얻을 수 있어 데이터 효율적임.
- **Encoder-Decoder (T5/BART 계열)**: Encoder가 양방향으로 입력을 처리하고 Decoder가 cross-attention으로 참조하며 생성. 번역·요약 같은 seq2seq task에 강함.
- **MoE (Mixture-of-Experts)**: FFN layer를 N개의 expert FFN으로 교체하고 각 token마다 top-k(보통 k=2)만 활성화. 같은 추론 비용(active params)으로 total params를 크게 늘려 성능을 높임. Mixtral, DeepSeek-V2/V3, Qwen2-MoE 등이 사용함.
- **MoE 핵심 문제 - Expert Collapse**: 특정 expert에만 token이 몰리는 현상. Load balancing loss(`α · Σ f_i · p_i`)를 메인 loss에 더해 모든 expert가 균등하게 사용되도록 유도함.

---

## 6. Alternative Architecture (Mamba / RWKV)

**왜 필요한가**: Transformer의 근본 문제인 O(n²) attention 복잡도와 KV-cache의 O(n) 선형 메모리 증가를 해결하기 위함. 매우 긴 시퀀스·edge device 추론에서 Transformer가 현실적으로 한계가 있음.

**핵심 아이디어**: 학습 시에는 parallel하게, 추론 시에는 O(1) 고정 메모리로 recurrent하게 동작하는 구조.

**주요 방법**:

- **SSM (State Space Model)**: `h_t = A·h_{t-1} + B·x_t`, `y_t = C·h_t` 형태의 linear recurrence. 학습 시 parallel convolution으로, 추론 시 O(1) 메모리 recurrent 계산으로 동작함.
- **Mamba (Selective SSM)**: 기존 SSM의 A, B, C 행렬을 입력에 따라 동적으로 결정하는 방법. 관련 있는 정보는 기억하고 없는 정보는 잊는 selective 처리가 핵심임. 추론 메모리가 O(n) KV-cache → O(d_state) 고정으로 줄어드는 것이 핵심 장점임.
- **RWKV**: Time-mixing(attention 역할)과 Channel-mixing(FFN 역할)을 RNN 방식으로 결합. 학습 시 행렬 곱으로 병렬, 추론 시 완전한 RNN으로 변환됨.
- **Hybrid (Jamba, Zamba)**: Mamba layer와 Attention layer를 교차 배치하여 SSM의 효율 + Attention의 정확한 recall을 결합한 구조.
- **현재 위치**: Transformer가 여전히 지배적이나, 긴 시퀀스와 edge 추론 중요성이 커지면서 hybrid 구조 채택이 증가하는 추세임.

---

## 7. Pretraining: Objective & Data Pipeline

**왜 중요한가**: LLM의 모든 capability는 pretraining에서 형성됨. 어떤 objective로, 어떤 데이터로 학습하느냐가 모델의 기반 능력을 결정함.

**Chinchilla Scaling**: Compute C가 고정될 때 모델 크기 N과 데이터 D를 균등하게 늘리는 것이 compute-optimal임 (`D_opt ≈ 20 × N_opt`). LLaMA는 inference-optimal을 위해 의도적으로 smaller model + more data 전략을 택함.

**Pretraining Objective**:

- **CLM (Causal LM)**: 이전 token들로 다음 token 예측. Teacher forcing으로 전체 시퀀스를 한 번에 병렬 학습 가능. Decoder-only 표준임.
- **MLM (Masked LM)**: token 15%를 [MASK]로 가리고 양방향 context로 예측. BERT 계열 Encoder-only 표준임.
- **Span Corruption**: 연속 span을 sentinel token으로 대체하고 decoder가 복원. T5 사용함.

**Data Pipeline**:

- **Deduplication**: 중복 문서 제거. MinHash+LSH로 near-duplicate 탐지하고 Jaccard similarity 기반 필터링. 미처리 시 특정 텍스트 memorization 증가 및 perplexity 저하됨.
- **Quality Filtering**: heuristic(길이·특수문자 비율), perplexity filtering(KenLM으로 저품질 제거), classifier-based(고품질 분류기 점수) 방법으로 웹 데이터 품질을 높임. FineWeb, CCNet이 대표적 결과물임.
- **Data Mixture**: 웹(50-80%), 책(10-20%), 코드(5-20%), Wikipedia(2-5%) 등의 혼합 비율을 조정. 코드 비율을 높이면 reasoning 능력이 향상됨. DoReMi는 proxy model로 최적 비율을 자동 추정함.
- **Curriculum Learning**: 쉬운 데이터에서 어려운 데이터 순으로 학습. LLaMA-3는 마지막 단계에 long context 데이터를 집중적으로 넣어 128K context를 지원함.

---

## 8. Post-Training & SFT

**왜 필요한가**: Pretraining으로 언어 능력을 갖춘 base model은 instruction을 따르지 않고 단순히 텍스트를 이어서 쓰는 경향이 있음. SFT로 "사용자 지시에 응답하는 assistant"로 만들어야 함.

**핵심 요소**:

- **SFT (Supervised Fine-Tuning)**: (instruction, response) 쌍으로 fine-tuning. **response 부분에만 loss를 계산**하는 loss masking이 핵심임. user turn에 loss를 주면 "user처럼 말하는 것"을 학습하게 됨.
- **Instruction Tuning**: 다양한 NLP task를 instruction 형식으로 변환해 multi-task 학습. FLAN이 대표적이며, 다양한 task 수가 데이터 개수보다 더 중요함.
- **Self-Instruct / Synthetic Data**: LLM 자신으로 instruction 데이터를 자동 생성하는 방법. Alpaca(GPT-3로 52K 생성), WizardLM(Evol-Instruct로 점진적 복잡화)가 대표적임. Teacher 모델 품질이 ceiling이 됨.
- **Rejection Sampling Fine-tuning**: 모델에서 N개 샘플을 생성하고 reward model/verifier로 필터링하여 좋은 것만 SFT 데이터로 사용. 반복하면 iterative rejection sampling이 되며 LLaMA-2 chat에 extensively 사용됨.
- **Knowledge Distillation**: Teacher의 soft label(전체 확률 분포)을 target으로 student를 학습. Hard label보다 "cat 60%, dog 30%" 같은 관계 정보를 더 많이 전달함. DeepSeek-R1이 긴 CoT를 소형 모델에 distillation하여 reasoning 모델을 만든 것이 대표 사례임.

---

## 9. Alignment (RLHF / DPO / GRPO)

**왜 필요한가**: SFT만으로는 모델이 helpful하고 harmless하고 honest(HHH)하게 행동하도록 보장하기 어려움. 사람의 선호도를 학습에 반영하는 alignment 단계가 필요함.

**핵심 challenge**: 사람의 선호도를 정량적으로 측정하기 어렵고, reward를 과도하게 최적화하면 reward model을 "속이는" 행동이 학습됨 (Goodhart's Law).

**주요 방법**:

- **RLHF (PPO)**: ① Reward model 학습(chosen/rejected 쌍에서 Bradley-Terry model로 학습) → ② PPO로 policy 최적화 + KL penalty(`reward = r - β·KL(π‖π_ref)`)로 reference 이탈 방지. 4개 모델(policy, reference, reward, value)이 필요해 복잡하지만 성능이 높음.
- **DPO (Direct Preference Optimization)**: Reward model 없이 preference data에서 직접 policy를 학습. RLHF의 closed-form solution을 유도하여 `L_DPO = -E[log σ(β·(log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l)))]`로 2개 모델(policy, reference)만 필요함. 구현이 단순하고 offline 학습 가능함. Mistral, Zephyr, Gemma 사용함.
- **GRPO (Group Relative Policy Optimization)**: 각 prompt에 G개 샘플 생성 후 상대적 reward로 advantage 추정. Value model(critic)이 불필요하여 PPO보다 메모리 절약. DeepSeek-R1의 핵심 학습 방식임.
- **RLVR (RL with Verifiable Rewards)**: 수학 정답 여부·코드 실행 결과 등 자동 검증 가능한 binary reward로 policy를 최적화하는 방법. GRPO와 결합하여 o1/DeepSeek-R1 계열의 Long CoT 학습에 사용됨.
- **KTO**: 쌍(pair) 없이 (prompt, response, good/bad) 단일 피드백으로 학습. Annotation 비용이 DPO보다 낮음.
- **Constitutional AI (Anthropic)**: 모델이 헌법 원칙에 따라 자기 응답을 비판·수정하고, 이를 preference data로 사용하는 방법. RLAIF(AI feedback)와 결합하여 사람 annotation 없이 harmless model을 학습함.
- **Reward Overoptimization**: PPO가 너무 진행되면 reward score는 오르지만 실제 quality는 떨어지는 현상. 응답이 비정상적으로 길어지거나 과장된 표현이 늘어남. KL penalty + reward model ensemble로 완화함.

---

## 10. PEFT (Parameter-Efficient Fine-Tuning)

**왜 필요한가**: 7B 모델 full fine-tuning에도 60GB+ GPU 메모리가 필요함. 전체 파라미터의 0.1~1%만 학습하면서도 full fine-tuning에 근접한 성능을 내는 방법이 필요함.

**핵심 아이디어**: 원래 weight는 frozen하고, 작은 추가 모듈이나 low-rank 근사만 학습함.

**주요 방법**:

- **LoRA (Low-Rank Adaptation)**: `ΔW = B·A` (B∈R^(d×r), A∈R^(r×k), r≪min(d,k))로 weight 변화량을 low-rank 분해로 근사. B는 0으로, A는 random으로 초기화하여 학습 시작 시 ΔW=0 보장. Inference 시 `W = W₀ + B·A`로 merge하여 추가 비용 없음. 현재 PEFT의 표준임.
- **QLoRA**: NF4(4-bit NormalFloat) + double quantization + paged optimizer 결합. 65B 모델을 단일 48GB GPU에서 fine-tuning 가능하게 함.
- **Adapters**: Transformer layer 사이에 bottleneck 모듈(`d → r → d`)을 삽입. Inference 시 추가 레이어를 통과해야 하므로 latency가 약간 증가함.
- **Prefix Tuning / Prompt Tuning**: 각 layer의 K, V 앞에 학습 가능한 virtual token을 삽입(Prefix Tuning), 또는 input embedding에만 learnable token을 추가(Prompt Tuning). 파라미터 수가 가장 적음.
- **Catastrophic Forgetting**: Fine-tuning 후 사전학습 지식이 손상되는 문제. LoRA는 weight 변화를 low-rank space로 제한하여 암묵적 regularization 효과가 있음. EWC(중요 파라미터 보호)나 replay buffer(사전학습 데이터 일부 혼합)로도 완화 가능함.

---

## 11. Training Stability & Optimization Tricks

**왜 중요한가**: 대규모 모델 학습에서 gradient explosion, overflow, 메모리 부족이 발생하면 수백 GPU-hour의 낭비가 생김. 안정적인 학습을 위한 기법들이 필수임.

**주요 기법**:

- **Mixed Precision (BF16)**: BF16은 FP32와 같은 exponent 범위(overflow 없음)를 가져 LLM 학습의 표준임. FP16은 overflow 위험으로 dynamic loss scaling이 필요함. H100에서는 FP8 학습으로 비용을 추가 절감함(DeepSeek-V3 사례).
- **Gradient Clipping**: 모든 파라미터의 gradient norm을 계산하고 threshold(보통 1.0)를 초과하면 scale down. 방향은 유지하고 크기만 제한하여 gradient explosion을 방지함.
- **Gradient Checkpointing**: Backward pass를 위한 중간 activation을 저장하지 않고 재계산하는 방법. 메모리를 O(n_layers) → O(√n_layers)로 줄이는 대신 forward pass를 ~1.33배 더 수행함.
- **Gradient Accumulation**: N번의 small batch에 걸쳐 gradient를 누적하고 N번마다 한 번 update. `effective batch = physical_batch × accum_steps × n_GPUs`로 메모리 제약을 극복함.
- **AdamW**: Adam에 decoupled weight decay를 결합한 표준 optimizer. β₁=0.9, β₂=0.95, weight_decay=0.1이 LLM 학습의 일반적 설정임.
- **LR Schedule**: Warmup(초반 gradient 불안정성 극복) + Cosine Decay(이후 부드럽게 감소)가 pretraining 표준임.
- **Label Smoothing**: One-hot 대신 `(1-ε)·y + ε/K` softened label 사용. 모델의 overconfidence를 방지하고 calibration을 개선함.

---

## 12. Distributed Training

**왜 필요한가**: LLaMA-3 405B 같은 모델은 단일 GPU에 들어가지도 않고, 수조 개의 token 학습에는 수천 GPU가 필요함. 이를 효율적으로 병렬화하는 방법이 필수임.

**메모리 분석 (16B 모델 기준)**: 파라미터 32GB + gradient 64GB + optimizer state 128GB = 총 ~224GB. 8xA100(80GB)에 들어가려면 sharding이 필수임.

**주요 병렬화 전략**:

- **Data Parallel (DDP)**: 모델 전체를 각 GPU에 복제하고 데이터를 나눠 처리. Ring-AllReduce로 gradient를 평균내고 backward와 통신을 overlap하여 효율적임. 단, 모델이 단일 GPU에 들어가야 함.
- **Tensor Parallel (TP)**: 하나의 weight matrix 자체를 column/row 방향으로 여러 GPU에 분할. 레이어 내 AllReduce 통신이 필요하므로 같은 node의 NVLink GPU에서만 효율적. Megatron-LM이 표준 구현임.
- **Pipeline Parallel (PP)**: 모델 레이어를 순서대로 여러 GPU에 분산. GPU 대기(bubble) 발생이 문제이며 micro-batch 수를 늘릴수록 bubble 비율(`(p-1)/(m+p-1)`)이 감소. 1F1B schedule로 메모리 절약함.
- **Sequence / Context Parallel**: 매우 긴 시퀀스를 여러 GPU에 분산. Ring Attention으로 각 GPU가 local window를 처리하며 K, V를 순차 전달함.
- **ZeRO (DeepSpeed)**: optimizer state만(Stage 1), +gradient(Stage 2), +파라미터(Stage 3) 순으로 분산하여 메모리를 단계적으로 줄임. ZeRO-Infinity는 CPU RAM과 NVMe도 활용함.
- **FSDP (PyTorch)**: ZeRO Stage 3의 PyTorch 공식 구현. Forward/backward 시 필요한 파라미터를 AllGather로 모아서 연산 후 즉시 해제함.
- **3D Parallelism**: TP × PP × DP를 결합한 전략. Megatron-LM이 구현하며 LLaMA-3 405B처럼 수천 GPU를 효율적으로 활용함.

---

## 13. Decoding Strategies

**왜 중요한가**: 동일한 모델이라도 decoding 방식에 따라 생성 품질, 다양성, 속도가 크게 달라짐. Task에 맞는 decoding 선택이 중요함.

**핵심 tradeoff**: 결정론적(정확하지만 단조로움) vs 확률적(다양하지만 품질 불안정)

**주요 방법**:

- **Greedy Decoding**: 매 step 최고 확률 token 선택. 결정론적이고 빠르나 반복 생성에 취약하고 창의적 생성에 부적합함.
- **Beam Search**: 상위 k개 가설을 동시에 유지하며 탐색. 번역·요약에는 유리하나 open-ended 생성에서는 다양성이 부족함. 길이 페널티(`score = log_prob / len^α`)로 짧은 시퀀스 편향을 보정함.
- **Top-k Sampling**: 상위 k개 token에서만 샘플링. 분포가 뾰족할 때와 넓을 때를 구분 못하는 단점이 있음.
- **Top-p (Nucleus) Sampling**: 누적 확률이 p를 넘을 때까지의 token에서 샘플링. 분포에 따라 후보 수가 자동 조정되어 top-k보다 robust함. 현재 가장 널리 사용됨.
- **Temperature**: Softmax 전에 logits를 T로 나눔. T<1이면 분포가 뾰족해지고(확실한 선택), T>1이면 평탄해짐(다양성 증가). 코딩/수학은 T≈0, 창작은 T≈1.1 사용함.
- **Constrained Decoding (JSON Schema)**: JSON 문법을 FSM으로 표현하고 각 step에서 invalid token의 logit을 -inf로 설정하여 항상 valid JSON을 보장하는 방법. Outlines, SGLang이 구현함.

---

## 14. Inference Efficiency (KV-Cache / PagedAttention / Speculative Decoding)

**왜 필요한가**: LLM 추론은 decode 단계가 memory-bandwidth-bound라 느리고, 많은 사용자를 동시에 처리하려면 메모리 관리와 throughput 최적화가 필수임.

**두 단계의 특성 차이**:
- **Prefill**: 전체 prompt를 한 번에 처리. Compute-bound (GPU core가 병목).
- **Decode**: 매 step 1 token 생성. Memory-bandwidth-bound (KV-cache 읽기가 병목).

**주요 기법**:

- **KV-Cache**: Decode 시 이전 K, V를 재계산하지 않고 저장해 재사용. 메모리는 `2 × n_layers × n_kv_heads × d_head × seq_len × dtype`에 비례함. GQA로 kv_heads를 줄이면 메모리가 크게 줄어듦.
- **PagedAttention (vLLM)**: OS의 virtual memory에서 영감을 받아 KV-cache를 고정 크기 block으로 관리. 기존 방식의 pre-allocation에 의한 낭비를 없애 fragmentation < 4%로 줄이고, 더 많은 request를 동시 처리 가능하게 함.
- **Continuous Batching**: 각 iteration마다 완료된 request를 제거하고 새 request를 추가하는 방법. Static batching의 "짧은 request가 끝나도 긴 request 완료까지 대기"하는 GPU 낭비를 없애 throughput을 크게 향상시킴.
- **Speculative Decoding**: 작은 draft model이 여러 token을 제안하고 큰 target model이 한 번의 forward pass로 accept/reject. Target model의 분포와 동일한 출력을 유지하면서 2~3배 throughput 향상. EAGLE/Medusa는 target model 자체에 draft head를 학습함.
- **Chunked Prefill**: 긴 prompt를 청크로 나눠 prefill과 decode를 번갈아 처리. 긴 prompt가 다른 request의 decode를 blocking하지 않아 TTFT(Time To First Token)를 개선함.
- **Prefix Caching (RadixAttention)**: 공통 prefix(system prompt 등)의 KV-cache를 재사용하는 방법. 반복적인 동일 system prompt가 있는 서비스에서 매우 효율적임.

---

## 15. Quantization

**왜 필요한가**: 70B 모델은 BF16으로도 140GB가 필요하지만 INT4로는 35GB로 줄어듦. Quantization은 모델을 더 작은 device에서 실행하고 decode의 memory-bandwidth 병목을 완화하는 핵심 기술임.

**핵심 tradeoff**: 낮은 precision일수록 메모리·속도는 좋아지지만 정확도가 저하됨.

**주요 방법**:

- **PTQ (Post-Training Quantization)**: 학습 완료된 모델을 재학습 없이 양자화. Calibration dataset으로 activation 분포를 측정하고 scale factor를 결정함.
- **Weight-Only Quantization (INT4/INT8)**: Weight만 낮은 precision으로 저장하고 추론 시 FP16으로 dequantize 후 연산. Decode의 memory-bandwidth 병목 완화에 특히 효과적임.
- **W8A8 (Activation Quantization)**: Weight와 Activation 모두 INT8로 양자화하여 INT8 tensor core를 완전 활용. LLM activation의 outlier 문제로 LLM.int8()처럼 outlier dimension은 FP16으로 따로 처리함.
- **AWQ**: Activation 크기가 큰 salient weight를 per-channel scale로 보호하여 INT4에서도 FP16에 가까운 성능을 유지하는 방법.
- **GPTQ**: Hessian 기반으로 weight를 순서대로 양자화하면서 오차를 나머지 weight에 분산 보상. 단순 라운딩보다 훨씬 낮은 양자화 오차를 달성함.
- **FP8**: H100에서 네이티브 지원하는 8-bit float. BF16 대비 2배 빠르고 DeepSeek-V3가 FP8 학습으로 학습 비용을 크게 줄임.
- **KV Quantization**: KV-cache를 INT8/INT4로 양자화하여 메모리를 2~4배 절약. Key가 Value보다 quantization에 더 민감함.

---

## 16. Serving Frameworks & Metrics

**왜 중요한가**: 같은 모델이라도 serving 인프라에 따라 throughput이 수십 배 차이나고, 비용과 latency SLA가 달라짐.

**핵심 Metrics**:

- **Throughput (tokens/sec)**: GPU 전체 효율의 지표. Batch size, KV-cache 용량, quantization에 의해 결정됨.
- **TTFT (Time To First Token)**: 첫 token이 반환될 때까지의 시간. Streaming 서비스의 사용자 경험과 직결됨. Prompt 길이에 비례함.
- **Tail Latency (P95/P99)**: SLA의 기준. Median보다 중요한 서비스 품질 지표임. Traffic spike 시 급증하는 경향이 있음.
- **Cost per Token**: `GPU 시간당 비용 / (throughput × 3600)`. Quantization, speculative decoding, continuous batching으로 최적화 가능함.

**주요 Frameworks**:

- **vLLM**: PagedAttention + continuous batching이 핵심. Prefix caching, LoRA serving, speculative decoding을 지원하며 OpenAI-compatible API를 제공하는 오픈소스 표준임.
- **TGI (HuggingFace)**: Docker 기반 serving. HuggingFace 생태계와 통합되어 token streaming, GPTQ/AWQ quantization을 지원함.
- **TensorRT-LLM (NVIDIA)**: Kernel fusion(여러 연산을 하나의 kernel로 통합하여 HBM 접근 최소화)으로 H100에서 최고 성능을 냄.
- **SGLang**: Radix tree 기반 prefix caching(RadixAttention)과 structured generation DSL이 특징. 긴 공통 system prompt가 있는 서비스에 특히 효율적임.

---

## 17. Long Context & RAG

**왜 필요한가**: 긴 문서·코드베이스·대화를 처리하거나, 최신 정보·도메인 지식을 모델에 주입하려면 긴 context 처리 능력 또는 외부 지식 검색이 필요함.

**Long Context의 핵심 문제**:

- **O(n²) Memory**: n=128K tokens이면 attention matrix가 수십 GB. FlashAttention + Ring Attention으로 해결함.
- **RoPE Extrapolation**: 학습 길이 이상에서 positional encoding이 작동하지 않음. NTK Scaling, YaRN, Position Interpolation으로 완화함.
- **Lost in the Middle**: 긴 context의 중간 정보를 잘 활용하지 못하는 현상. RAG에서 중요한 chunk를 context 앞/뒤에 배치하는 전략이 필요함.

**RAG Pipeline**:

- **Chunking**: 긴 문서를 검색 가능한 단위로 분할. Fixed-size(단순하나 의미 파괴 위험), Sentence/Paragraph-aware(자연스러운 경계 존중), Semantic(의미 유사도 기반) 방식이 있음.
- **Embedding & Vector DB**: Bi-encoder로 텍스트를 벡터화하고 HNSW/IVF 알고리즘으로 근사 최근접 이웃 검색. Faiss, Pinecone, Weaviate 등이 대표적임.
- **Reranking (Cross-Encoder)**: 빠른 bi-encoder 검색 후 query-document 쌍을 함께 입력받는 cross-encoder로 정밀 re-scoring하는 2단계 검색 방법. 정확도를 크게 향상시킴.
- **HyDE**: 실제 문서 대신 LLM이 생성한 가상의 답변을 query로 사용하여 embedding 검색 품질을 높이는 방법.

---

## 18. Reasoning & Test-Time Compute Scaling

**왜 중요한가**: 같은 모델도 얼마나 "생각하게 하느냐"에 따라 수학·코딩·논리 성능이 크게 달라짐. 추론 시간의 compute를 늘리는 것이 모델을 더 크게 만드는 것과 경쟁할 수 있음.

**핵심 직관**: Transformer는 constant-depth 계산 그래프라 복잡한 추론을 한 번에 할 수 없음. CoT는 여러 forward pass에 해당하는 계산을 sequential한 token으로 펼쳐주는 것임.

**Prompting 기법**:

- **Chain-of-Thought (CoT)**: 단계별 풀이 과정을 생성하게 하는 방법. Few-shot CoT(예시 제공)와 zero-shot CoT("Let's think step by step")가 있음. 충분히 큰 모델(~100B+)에서만 효과적임.
- **Self-Consistency**: 같은 질문에 여러 CoT 경로를 샘플링하고 majority vote로 결정. 단일 CoT보다 일관적으로 더 높은 성능을 달성함.
- **Tree-of-Thought (ToT)**: 추론을 tree로 구성하고 BFS/DFS/MCTS로 탐색. 중간 단계를 평가하여 유망한 경로를 선택적으로 확장함.

**Reward Model 기반 Search**:

- **ORM (Outcome Reward Model)**: 최종 답만 평가. 레이블 수집이 쉽지만 틀린 추론 과정도 행운으로 맞을 수 있다는 문제가 있음.
- **PRM (Process Reward Model)**: 각 추론 step마다 점수를 매김. 오류 위치를 정확히 파악하고 잘못된 경로를 조기 차단 가능. 단, step-level annotation 비용이 매우 높음.
- **Best-of-N (BoN)**: N개 샘플을 생성하고 reward model로 최선을 선택. `P(success) = 1 - (1-p)^N`으로 N을 늘릴수록 성능이 향상됨.
- **Test-Time Scaling Law**: 작은 모델 + 많은 inference compute가 큰 모델 + 적은 compute를 이길 수 있음 (3B+256x ≈ 34B+1x). Compute를 "학습 시"에서 "추론 시"로 이동하는 패러다임 전환임.

**Training 기법**:

- **Long CoT (RLVR + GRPO)**: 모델이 verifiable reward(수학 정답, 코드 실행 결과)를 기반으로 긴 내부 "thinking" 과정을 생성하도록 GRPO로 학습하는 방법. DeepSeek-R1, o1/o3의 핵심 학습 방식임.

---

## 19. LLM Safety

**왜 중요한가**: LLM이 실제 서비스에 배포되면 유해 콘텐츠 생성, 사실 왜곡, 악의적 사용 등의 문제가 발생함. 모델 개발 단계에서부터 안전성을 고려해야 함.

**주요 위험 및 대응**:

- **Hallucination**: 사실이 아닌 내용을 자신있게 생성하는 현상. Factual(잘못된 사실), Intrinsic(소스와 모순), Extrinsic(소스에 없는 내용 추가) 유형이 있음. RAG(사실 정보 주입), RLHF(사실적 정확성 보상), uncertainty quantification으로 완화함.
- **Jailbreak**: 모델의 안전 가이드라인을 우회하는 공격. Role-playing("DAN 역할"), obfuscation(Base64·역방향), multi-turn 점진적 유도, translation attack 방법이 있음. Adversarial training과 Constitutional AI로 방어함.
- **Prompt Injection**: RAG·agent 시스템에서 외부 문서·웹페이지의 숨겨진 명령으로 모델 행동을 조작하는 공격. Instruction hierarchy(시스템 > 사용자 > 외부 데이터 우선순위) 명확화와 delimiter로 방어함.
- **Bias**: 학습 데이터의 사회적 편향(성별, 인종 등)이 그대로 모델에 반영됨. BBQ, WinoBias 벤치마크로 평가하고 RLHF로 완화함.
- **Calibration & Abstention**: 모델의 confidence가 실제 정확도와 일치하는지(ECE로 측정), 모르는 것을 "모른다"고 할 수 있는지의 능력. SFT 데이터에 abstention 예시를 포함하고 RLHF에서 높은 reward를 부여하여 학습함.
- **Red Teaming**: 모델 취약점을 찾기 위해 의도적으로 공격적 입력을 시도하는 체계적 평가. Manual(전문가 직접)과 Automated(Attacker LLM → Target LLM → Judge LLM 루프) 방식이 있음.

---

## 20. Benchmarks & Evaluation Issues

**왜 중요한가**: 벤치마크 점수가 실제 능력을 정확히 반영하지 못하는 경우가 많음. 올바른 평가 설계와 결과 해석이 연구의 신뢰성을 결정함.

**주요 벤치마크**:

- **MMLU**: 57개 분야 4지선다. 2025년 기준 frontier model들이 87~88%로 수렴하여 saturation됨.
- **GPQA (Diamond)**: 전문가 수준 과학 문제. 인간 전문가도 ~65%인데 o3가 초과하여 새 기준이 됨.
- **MATH / AIME**: 고등~올림피아드 수준 수학. o1/o3가 AIME에서 전문가 수준에 도달함.
- **HumanEval / LiveCodeBench**: 코딩 능력 평가. LiveCodeBench는 최근 문제를 사용하여 contamination 방지함.

**핵심 평가 이슈**:

- **Benchmark Contamination**: 학습 데이터에 test set이 포함되어 암기인지 이해인지 구분 불가한 문제. n-gram overlap, min-k% prob으로 탐지함. Time-based filtering(최근 문제만 사용)이나 procedurally generated 문제로 완화함.
- **Benchmark Saturation**: 모델들이 벤치마크 ceiling에 근접하여 성능 차이가 구분 안 되는 현상. MMLU → MMLU-Pro, GSM8K → MATH → AIME로 점점 어려운 벤치마크가 계속 필요해짐.
- **LLM-as-a-Judge Biases**: LLM 평가자의 편향들. Position bias(순서 편향), verbosity bias(긴 답 선호), self-enhancement bias(자기 스타일 선호), sycophancy(사용자 의견 추종)가 대표적임. 순서를 바꿔서 두 번 평가하거나 여러 모델 ensemble로 완화함.
- **Evaluation Harness**: lm-evaluation-harness(EleutherAI)가 표준 프레임워크. 같은 모델도 shot 수와 prompt 형식에 따라 결과가 달라지므로 표준 harness로 동일 조건 비교가 중요함.

---

# VLM

## 21. Vision Encoder (CNN vs ViT, CLIP, SigLIP)

**왜 필요한가**: LLM은 텍스트 token만 처리할 수 있으므로, 이미지를 LLM이 이해할 수 있는 token sequence로 변환하는 컴포넌트가 필요함.

**CNN vs ViT 선택의 배경**: CNN은 locality bias 덕분에 소규모 데이터에 강하지만 장거리 의존성 포착이 약하고 대규모 확장성이 제한적임. ViT는 이미지를 patch sequence로 처리하고 global self-attention으로 어떤 두 patch 간 관계도 첫 layer에서 계산 가능함. 대규모 데이터에서 CNN을 능가하여 VLM의 표준이 됨.

**Patch Embedding 과정**: 이미지를 `P×P` patch로 분할 → flatten → linear projection → positional embedding → token sequence. 224×224 이미지를 16×16 patch로 나누면 196개 token이 생성됨.

**사전학습 방법**:

- **CLIP (Contrastive Language-Image Pretraining)**: 수백만 이미지-텍스트 쌍에서 InfoNCE loss로 contrastive 학습. 같은 쌍은 embedding space에서 가깝게, 다른 쌍은 멀게 학습함. 대규모 batch가 필요하고 VLM vision encoder로 가장 많이 활용됨.
- **SigLIP**: CLIP의 softmax 기반 InfoNCE 대신 sigmoid loss를 사용. 대규모 batch 없이도 효과적으로 학습 가능하고 성능이 더 좋아 최신 VLM들(LLaVA-NeXT, InternVL 등)이 채택함.
- **ViT-L/14 vs ViT-H/14**: patch size 14인 ViT-Large와 Huge. 작은 patch가 더 세밀한 표현을 제공하여 OCR·fine-grained 시각 분석에 유리함.

---

## 22. VLM Architecture & Projector 설계

**왜 중요한가**: Vision encoder와 LLM은 완전히 다른 공간에서 사전학습됨. 두 공간을 어떻게 연결하느냐가 멀티모달 이해 품질을 결정함.

**전체 구조**: `[Vision Encoder] → [Projector/Connector] → [LLM]`. Projector가 "시각 언어를 텍스트 언어로 번역하는 역할"을 담당함.

**Projector 유형**:

- **Linear Projector**: `(N_patches, D_vision) → Linear → (N_patches, D_llm)`. 극도로 단순하나 비선형 변환이 없어 표현력 제한. LLaVA 1.0이 사용하여 단순함도 충분히 효과적임을 증명함.
- **MLP Projector**: Linear → GELU → Linear 구조로 비선형성 추가. LLaVA-1.5가 사용하여 성능 향상. 현재 가장 많이 쓰이는 구조임.
- **Q-Former (BLIP-2)**: 고정된 수의 learnable query token이 cross-attention으로 image feature를 압축. 이미지 token 수를 32개처럼 고정으로 줄여 LLM의 context 부담을 크게 낮춤.
- **Perceiver Resampler (Flamingo)**: Q-Former와 유사하나 더 유연한 pooling. 다양한 해상도·수의 이미지에서 고정 수의 visual token을 출력함.
- **Unified Multimodal Transformer**: 별도 vision encoder 없이 하나의 Transformer가 이미지 patch와 텍스트 token을 처음부터 함께 처리. 더 깊은 cross-modal interaction이 가능함.

**Vision Encoder & LLM frozen/finetuning 결정**:

- Stage 1에서는 vision encoder와 LLM을 frozen하고 projector만 학습하는 것이 표준임. 이미 검증된 각 모델의 표현을 보존하면서 빠르게 alignment를 학습함.
- Stage 2에서 LLM을 함께 fine-tuning하여 복잡한 멀티모달 reasoning을 학습함. LoRA로 효율적으로 fine-tuning하는 것이 일반적임.

---

## 23. Image Resolution Handling

**왜 필요한가**: 기본 vision encoder(ViT-L/14)의 입력은 224×224 또는 336×336임. 실제 이미지(영수증, 문서, 인포그래픽)를 강제로 리사이즈하면 텍스트가 뭉개져서 판독 불가 수준이 됨. OCR·문서 이해에서 고해상도 처리는 필수적임.

**핵심 문제**: 고해상도를 그대로 처리하면 token 수가 폭발적으로 늘어나 LLM context window를 빠르게 소진함. 정보 손실 없이 token 수를 적절히 유지하는 것이 핵심임.

**주요 방법**:

- **Fixed Resolution (Baseline)**: 모든 이미지를 고정 크기로 리사이즈. 구현이 단순하고 token 수가 일정하나 고해상도 이미지에서 정보 손실이 큼. 고해상도 처리가 불필요한 task에서 사용함.
- **AnyRes / Dynamic Tiling (LLaVA-NeXT, InternVL)**: 이미지를 미리 정의된 grid(1×1, 2×1, 1×2, 2×2 등)에서 가장 가까운 것을 선택하여 여러 tile로 분할. 각 tile을 독립적으로 인코딩 후 global thumbnail과 함께 LLM에 전달. 해상도를 높이면서 token 수를 controllable하게 유지함.
- **Dynamic Resolution (Qwen2-VL)**: 이미지 원래 해상도를 거의 그대로 처리하고 M-RoPE로 spatial position을 인코딩하는 방법. Aspect ratio 왜곡 없이 처리 가능함.
- **Token Pruning / Compression**: 시각 token이 너무 많을 때 redundant token을 제거하거나 병합하는 방법. Attention score 기반, 유사도 기반, learned compression 방식이 있음.

---

## 24. VLM Training Pipeline (2-Stage)

**왜 필요한가**: Vision encoder와 LLM은 완전히 다른 목표로 사전학습되어 직접 연결하면 training이 불안정함. 단계적으로 alignment한 뒤 instruction following을 학습하는 2-stage 전략이 필요함.

**Stage 1 - Alignment Pre-training**:
- **목적**: Vision encoder 출력을 LLM 토큰 공간에 정렬.
- **설정**: Vision encoder + LLM 모두 frozen, projector만 학습.
- **데이터**: 대규모 이미지-캡션 쌍(CC3M, LAION, ShareGPT4V 등). 600K~수백만 샘플.
- **효과**: Projector가 "시각 언어 → 텍스트 언어 번역" 능력을 빠르게 학습함.

**Stage 2 - Instruction Fine-tuning**:
- **목적**: 다양한 멀티모달 instruction을 따르는 능력 학습.
- **설정**: Projector + LLM 함께 학습(또는 LLM에 LoRA 적용). Vision encoder는 frozen하거나 낮은 LR로 fine-tuning.
- **데이터**: VQA, 이미지 설명, OCR, 차트 이해, 추론 등 멀티모달 instruction 데이터.

**왜 2단계인가**: Stage 1에서 LLM을 frozen하는 이유는 이미 검증된 LLM의 언어 능력을 보존하면서 projector가 빠르게 alignment를 학습하게 하기 위함임. LLM을 처음부터 함께 학습하면 비효율적이고 불안정함.

---

## 25. VLM Hallucination

**왜 문제인가**: 이미지라는 명확한 grounding 소스가 있음에도 이미지에 없는 물체나 틀린 정보를 사실처럼 생성함. 의료 이미지 분석, 문서 처리 등 정확성이 중요한 분야에서 심각한 문제가 됨.

**유형**:
- **Object Hallucination**: 이미지에 없는 물체를 있다고 설명하는 가장 흔한 유형.
- **Attribute Hallucination**: 물체의 색상, 크기, 위치 등을 잘못 설명.
- **Relationship Hallucination**: 물체 간의 관계를 잘못 설명.

**원인**: 강한 언어 prior(LLM이 "이런 상황에서는 이런 물체가 있다"를 통계적으로 학습)가 시각 정보보다 과도하게 반영됨. Vision-text alignment가 불완전하면 LLM이 이미지를 제대로 참조하지 않고 언어 패턴으로만 응답함.

**평가 - POPE (Polling-based Object Probing Evaluation)**: 이미지에 특정 물체가 있는지 yes/no로 묻는 방법. Random(랜덤 물체), Popular(자주 등장하는 물체), Adversarial(이미지에 없지만 공존 가능성 높은 물체) 세 가지 난이도로 평가함.

**완화 방법**: Grounding 데이터(물체 위치 정보) 추가, instruction tuning에서 부정 예시 강화, RLHF에서 hallucination에 낮은 reward 부여, 답변 전 이미지를 다시 참조하게 하는 prompt engineering.

---

## 26. Catastrophic Forgetting in VLM

**왜 문제인가**: VLM 학습 중 vision encoder나 LLM을 fine-tuning하면 사전학습에서 얻은 표현이 손상됨. Vision encoder의 CLIP 표현이 무너지거나 LLM의 언어 능력이 저하되면 전체 VLM 성능이 떨어짐.

**발생 지점**:
- **Vision Encoder 손상**: High LR로 vision encoder를 fine-tuning하면 CLIP의 일반적인 시각 표현이 멀티모달 task에 과적합됨.
- **LLM Catastrophic Forgetting**: VLM instruction data에만 fine-tuning하면 LLM의 일반 언어 능력(추론, 지식 등)이 저하됨.

**완화 방법**:
- **Stage 1에서 Vision Encoder + LLM Frozen**: 사전학습 표현을 보존하면서 projector만 alignment 학습함.
- **낮은 LR로 Vision Encoder Fine-tuning**: Stage 2에서 vision encoder를 fine-tuning할 경우 매우 낮은 LR(1e-6 수준)을 사용함.
- **LoRA 적용**: LLM 전체를 fine-tuning하는 대신 LoRA로 변화량을 low-rank space로 제한하여 원래 표현을 최대한 보존함.
- **Mixed Data (Replay)**: VLM instruction 데이터에 pure text instruction 데이터를 혼합하여 LLM 언어 능력을 유지함.

---

## 27. Video Understanding in VLM

**왜 어려운가**: 비디오는 시간 차원이 추가된 이미지 시퀀스임. 30fps, 1분 = 1800 frame × frame당 수백 token = 수십만 token이 필요하여 LLM context window를 순식간에 초과함. Token 수를 줄이면서 시간적 정보를 보존하는 것이 핵심 과제임.

**주요 방법**:

- **Frame Subsampling**: 1fps, 0.5fps 등으로 frame 수를 줄임. 단순하지만 빠른 동작이나 짧은 이벤트를 놓칠 수 있음.
- **Temporal Pooling**: 인접 frame의 feature를 평균하거나 aggregation하여 token 수를 줄임. LLaVA-Video가 사용함.
- **M-RoPE (Qwen2-VL)**: 각 frame의 patch token에 (시간 t, 행 h, 열 w) 3D position을 독립 인코딩하여 시간적 위치 정보를 position encoding 자체에 포함시킴. 비디오 이해에서 오픈소스 최강 성능을 달성함.
- **Multi-Image Position IDs**: 여러 frame을 처리할 때 frame 간 position ID가 겹치지 않도록 부여하여 시간 순서를 모델이 구분할 수 있게 함.
- **핵심 능력들**: 시간적 순서 이해("무엇이 먼저 일어났는가"), 이벤트 위치 파악(특정 장면의 timestamp), fine-grained 시간 이해가 필요함.

---

# VLA

## 28. VLA 기초 (BC, Covariate Shift, Offline RL)

**VLA란**: Vision + Language 입력을 받아 robot action을 출력하는 모델. VLM의 능력을 로봇 제어로 확장한 것임.

**Behavior Cloning (BC)**:

- 전문가 시연 데이터로 `π_θ(action|observation)`을 지도학습으로 모방하는 가장 단순한 방법.
- **Covariate Shift 문제**: 작은 실행 오류 → 학습 분포를 벗어남 → 이후 상태에서 어떻게 해야 할지 모름 → 오류 누적. 로봇 학습의 핵심 난제임.
- **DAgger**: Interactive하게 전문가 피드백을 받아 오류가 발생한 상태에서의 correct action을 학습 데이터에 추가하는 방법. Covariate shift를 직접 해결함.

**Offline RL**:

- 환경 상호작용 없이 미리 수집된 데이터셋만으로 policy를 학습하는 방법. 데이터셋 외 (s,a) 쌍에 대한 Q-value 과대 추정이 핵심 문제임.
- **CQL (Conservative Q-Learning)**: 데이터셋에 없는 action에 대해 Q-value를 의도적으로 낮게 추정하여 distribution shift를 방지함.
- **IQL (Implicit Q-Learning)**: Q-value를 직접 최대화하지 않고 expectile regression으로 안정적으로 학습함.

**Sim-to-Real Transfer**: 시뮬레이터에서 학습한 policy를 실제 로봇에 전이하는 방법. Domain randomization(물체 색상·질감·조명·물리 파라미터를 다양화)으로 sim-real gap을 줄임.

---

## 29. Action Representation

**왜 중요한가**: 연속적인 로봇 action(joint angles, end-effector position 등)을 LLM이 처리하고 생성할 수 있는 형태로 표현하는 방법을 잘 선택해야 함. 각 방법이 trade-off가 있음.

**주요 방법**:

- **Discrete Tokenization**: 연속 action을 256개 bin으로 양자화하여 LLM vocabulary에 추가. 기존 LLM 아키텍처를 그대로 재사용 가능하나 정밀도 손실과 느린 autoregressive 생성이 단점임.
- **Continuous Regression**: LLM의 마지막 hidden state에 regression head를 붙여 action을 연속 벡터로 직접 예측. 빠르고 정밀하나 multimodal action distribution(같은 상황의 여러 valid action) 처리가 어려움.
- **Diffusion Policy**: Random noise에서 시작하여 DDPM/DDIM으로 반복적 denoising하여 action sequence를 생성. Multimodal action distribution을 포착 가능한 것이 핵심 장점임. "왼쪽에서 접근"과 "오른쪽에서 접근" 둘 다 valid할 때, regression은 두 방법의 평균(invalid trajectory)을 출력하지만 diffusion은 하나를 일관되게 선택함.
- **Flow Matching**: Gaussian noise에서 data까지 직선 경로(straight-path ODE)로 이동. Diffusion보다 샘플링 step이 적고 빠름. π0가 사용함.
- **Chunked Action Prediction**: 한 번에 H개의 미래 action을 예측하고 receding horizon 방식으로 실행. Temporal consistency를 높이고 고주파 제어에서 latency를 줄임.

---

## 30. 대표 VLA 모델들

**발전 흐름**: task-specific → multitask(RT-1) → VLM 기반(RT-2) → 오픈소스(Octo, OpenVLA) → dexterous manipulation(π0)

**주요 모델**:

- **RT-1 (Google 2022)**: EfficientNet + TokenLearner(시각 token 512→8 압축) + causal Transformer로 130K real robot 에피소드를 학습. 700개 이상 task에서 97% 성공률을 달성하여 대규모 multitask robot learning 가능성을 입증함.
- **RT-2 (Google 2023)**: PaLI-X(55B)/PaLM-E(562B) VLM에 robot action token을 vocabulary에 추가하고 co-finetuning. 웹 지식을 로봇으로 전이하여 학습에 없던 물체·명령 조합에도 generalization하는 것이 핵심 기여임.
- **Octo (UC Berkeley 2023)**: Modular 설계의 오픈소스 generalist robot policy. Open X-Embodiment 다양한 로봇 데이터로 학습하고 action을 diffusion head로 예측함. 누구나 새 로봇에 fine-tuning 가능함.
- **OpenVLA**: SigLIP + LLaMA-2 7B 기반 오픈소스 VLA. Discrete autoregressive action tokenization으로 RT-2급 성능을 오픈소스로 제공함.
- **π0 (Physical Intelligence 2024)**: PaliGemma VLM + flow matching action expert module 구조. 언어 이해(VLM 담당)와 정밀 manipulation(action expert 담당)을 분리하여 dexterous manipulation에서 최고 성능을 달성함.

---

## 31. Embodied AI 핵심 주제들

### Hierarchical Planning

**왜 필요한가**: "Make coffee" 같은 long-horizon task를 저수준 joint control로 직접 학습하기 어려움. 고수준 계획 → 중간 step → 저수준 제어로 분리하면 각 레벨에서 더 다루기 쉬운 문제가 됨.

- **계층 구조**: Goal(LLM이 "커피 만들기"를 step으로 분해) → Sub-goals(각 물체별 manipulation) → Actions(joint angle 제어, 10Hz).
- **LLM as Planner**: LLM이 자연어로 high-level plan을 생성하고, 하위 policy(VLA)가 각 step을 실행함. SayCan은 affordance model(현재 환경에서 실행 가능한지)과 결합하여 현실적인 계획을 생성함.

### Language-Conditioned Manipulation

**왜 중요한가**: 자연어 명령으로 로봇을 제어하면 특정 물체·위치·동작에 대한 사전 프로그래밍 없이 flexible한 사용이 가능함.

**Generalization 수준**: Zero-shot generalization(학습에 없던 새 물체), compositional generalization(새 조합), analogical generalization(유추), language-only generalization(텍스트 설명만으로 이해) 4단계가 있음.

### Tool Use (Embodied)

- LLM의 API tool use를 물리적 도구로 확장. pick-and-place, screwdriver 조작, 버튼 클릭 등을 언어로 명세하여 실행함.
- Multi-robot collaboration: 로봇이 다른 로봇에게 작업을 위임하는 방식으로 확장 가능함.

### Data Collection & Open X-Embodiment

- 로봇 학습 데이터 수집의 어려움: kinesthetic teaching, VR teleoperation, human video 매핑 등을 사용함.
- **Open X-Embodiment**: Google 등이 여러 기관의 로봇 데이터를 통합한 대규모 데이터셋. 다양한 로봇 형태와 task에서 generalist policy 학습을 가능하게 함.
