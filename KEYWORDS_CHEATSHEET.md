# LLM / VLM / VLA Keywords Summary

> 각 키워드에 대한 핵심만 간결하게 정리한 문서.

---

## 목차

- [LLM](#llm)
  - [기초 수학 & 정보이론](#기초-수학--정보이론)
  - [Architecture](#architecture)
  - [Tokenization](#tokenization)
  - [Pretraining](#pretraining)
  - [Post-Training & SFT](#post-training--sft)
  - [Alignment (RLHF / DPO / GRPO)](#alignment-rlhf--dpo--grpo)
  - [PEFT](#peft)
  - [Optimization & Training Stability](#optimization--training-stability)
  - [Distributed Training](#distributed-training)
  - [Inference & Decoding](#inference--decoding)
  - [Inference Efficiency](#inference-efficiency)
  - [Quantization](#quantization)
  - [Serving Frameworks](#serving-frameworks)
  - [Serving Metrics](#serving-metrics)
  - [Long Context & RAG](#long-context--rag)
  - [Reasoning & Test-Time Compute](#reasoning--test-time-compute)
  - [Safety](#safety)
  - [Benchmarks & Evaluation](#benchmarks--evaluation)
  - [Model Families](#llm-model-families)
- [VLM](#vlm)
  - [Vision Encoder](#vision-encoder)
  - [VLM Architecture (Projector / Connector)](#vlm-architecture-projector--connector)
  - [Resolution Handling](#resolution-handling)
  - [Multimodal Training](#multimodal-training)
  - [VLM Benchmarks](#vlm-benchmarks)
  - [VLM Model Families](#vlm-model-families)
- [VLA](#vla)
  - [VLA 기초](#vla-기초)
  - [Action Representation](#action-representation)
  - [Representative Models](#vla-representative-models)
  - [Embodied AI Topics](#embodied-ai-topics)

---

# LLM

## 기초 수학 & 정보이론

- **MLE (Maximum Likelihood Estimation)**: 데이터를 가장 잘 설명하는 θ를 찾는 방법. LM에서 next token prediction loss 최소화와 동일함.
- **Cross-Entropy Loss**: 정답 token의 log probability에 음수를 취한 값. NLL 최소화 = MLE 수행 = Cross-Entropy 최소화, 모두 같은 것임.
- **KL Divergence**: 두 분포의 차이를 측정하는 비대칭 지표. RLHF의 KL penalty(`reward = r - β·KL(π‖π_ref)`)로 policy가 reference에서 너무 벗어나지 않게 제약하는 데 사용함.
- **Perplexity**: LM이 test set을 얼마나 잘 예측하는지 나타내는 지표. `exp(cross-entropy loss)`로 계산하며, 낮을수록 좋고 같은 tokenizer를 쓰는 모델끼리만 비교 가능함.
- **Entropy**: 분포의 불확실성을 측정하는 지표. 균등 분포일 때 최대, 한 값에 확률이 집중되면 0임.
- **Scaling Law (Chinchilla)**: Compute C가 고정될 때 모델 크기 N과 데이터 D를 균등하게 늘리는 것이 optimal함. `D_opt ≈ 20 × N_opt` (tokens per parameter). GPT-3는 undertrained, LLaMA-3는 inference-optimal을 위해 의도적으로 data-heavy하게 학습함.
- **Emergent Abilities**: 모델이 일정 규모를 넘어서면 갑자기 나타나는 능력들(CoT reasoning, few-shot ICL 등). 일부는 진짜 창발이고, 일부는 비선형 metric의 artifact임.
- **In-Context Learning (ICL)**: 파라미터 업데이트 없이 prompt에 예시를 넣는 것만으로 모델이 새 task를 수행하는 능력. Few-shot(예시 몇 개)과 zero-shot(instruction만)으로 나뉨.
- **AdamW**: Adam에 decoupled weight decay를 결합한 LLM 학습의 표준 optimizer. weight decay를 gradient가 아닌 파라미터에 직접 적용하는 것이 핵심임.
- **Learning Rate Schedule**: Warmup(초반 LR를 0에서 점진 증가) + Cosine Decay(이후 부드럽게 감소)가 LLM pretraining의 표준 조합임.

---

## Architecture

### Transformer Block

- **Self-Attention**: 각 token이 다른 모든 token과의 관련도를 계산하여 정보를 모으는 메커니즘. `Attention(Q,K,V) = softmax(QK^T / √d_k) · V`로 계산함.
- **Scaling by √d_k**: Q·K 내적의 분산이 d_k에 비례하여 커지는 것을 막기 위한 scaling. 이 없으면 softmax가 포화되어 gradient가 소실됨.
- **Causal Mask**: Decoder-only LM에서 future token을 보지 못하게 QK^T 이후 softmax 이전에 upper triangular 부분을 -inf로 마스킹하는 방법.
- **Multi-Head Attention (MHA)**: 여러 head가 서로 다른 관점(문법적, 의미적, 위치 등)에서 동시에 attention을 계산하는 방법. 각 head는 `d_k = d_model / h` 차원으로 독립 학습함.
- **Residual Connection**: 각 sublayer의 입출력을 더하는 shortcut. Gradient highway를 만들어 깊은 네트워크에서도 학습 가능하게 함.
- **Pre-LN vs Post-LN**: Pre-LN(LayerNorm을 sublayer 이전에 적용)이 학습 안정성이 훨씬 높아 GPT-3 이후 대부분의 LLM이 채택함.
- **RMSNorm**: 평균 제거 없이 RMS로만 정규화하는 방법. LayerNorm 대비 ~10-15% 빠르고 성능은 동등함. LLaMA, Mistral 등 현대 LLM의 표준임.
- **FFN (Feed-Forward Network)**: Attention 다음에 오는 `d_ff = 4 × d_model` 크기의 MLP. 각 token의 표현을 독립적으로 변환(position-wise)함.
- **SwiGLU / GeGLU**: 일반 FFN 대신 gating을 추가한 변형. `SwiGLU(x) = (W₁x) ⊙ Swish(W₃x)` 형태로, 불필요한 정보를 게이팅으로 억제하여 성능이 더 좋음. LLaMA, PaLM 등이 사용함.
- **Weight Tying**: Input embedding matrix와 output projection(LM head)의 파라미터를 공유하는 기법. vocab_size × d_model만큼 파라미터를 절약함.

### Attention Variants

- **MQA (Multi-Query Attention)**: 모든 head가 단 하나의 K, V를 공유하는 방법. KV-cache를 h배 줄이지만 성능이 약간 저하됨. PaLM, Falcon이 사용함.
- **GQA (Grouped Query Attention)**: MHA와 MQA의 중간. G개 그룹으로 나눠 그룹 내에서 K, V를 공유함. MQA만큼 빠르면서 MHA에 가까운 품질을 달성하여 현재 decoder-only LLM의 사실상 표준임. LLaMA-2/3, Mistral, Gemma 등이 사용함.
- **Sparse Attention**: 모든 n² 쌍 대신 선택된 쌍만 계산하는 방법. Local(window) attention, strided attention, global token 등의 패턴으로 O(n·w) 복잡도를 달성함.
- **Sliding Window Attention**: 각 token이 자신을 중심으로 w개의 이전 token들과만 attend하는 방법. Mistral 7B가 사용함.
- **FlashAttention v1/v2/v3**: attention 연산 자체는 변경하지 않고 GPU IO를 최적화하는 방법. Q, K, V를 SRAM에 맞는 tile로 나눠 n×n 행렬을 HBM에 저장하지 않아 메모리를 O(n²)에서 O(n)으로 줄이고 속도를 2~4배 높임.

### Positional Encoding

- **Sinusoidal PE**: sin/cos 파형으로 위치를 인코딩하는 원래 Transformer의 방법. 학습 없이 계산 가능하나 extrapolation이 잘 안 됨.
- **Learned PE**: 각 위치 임베딩을 직접 학습하는 방법. BERT, GPT-2가 사용하나 최대 길이 이상으로 extrapolation 불가함.
- **RoPE (Rotary Position Embedding)**: Q와 K 벡터를 위치에 따라 회전시켜 attention score가 상대 위치(m-n)에만 의존하게 만드는 방법. LLaMA, Mistral, Qwen 등 현대 LLM의 표준임.
- **ALiBi**: attention score에 거리에 비례한 선형 페널티를 직접 더하는 방법. 학습 파라미터 없이 구현이 단순하고 extrapolation이 강함. MPT, BLOOM이 사용함.
- **RoPE Scaling (NTK / YaRN / Position Interpolation)**: 학습 이후에 context length를 늘리기 위한 RoPE base 조정 기법들. NTK는 base frequency 조정, PI는 위치 압축, YaRN은 고주파/저주파 성분을 다르게 scaling함.
- **M-RoPE (Multimodal RoPE)**: 텍스트 token에는 1D position, 이미지 patch에는 (time, height, width) 3D position을 독립 인코딩하는 방법. Qwen2-VL이 도입함.

### Model Architecture Types

- **Encoder-only**: 양방향 attention으로 전체 시퀀스를 처리하는 구조. MLM으로 사전학습하며 분류·임베딩 task에 강함. BERT, RoBERTa가 대표적임.
- **Decoder-only**: Causal attention만 사용하여 left-to-right로 처리하는 구조. CLM으로 사전학습하며 generative task에 자연스럽고 scalability가 우수하여 현재 LLM의 표준임. GPT, LLaMA 계열이 해당됨.
- **Encoder-Decoder**: Encoder가 입력을 양방향 처리하고 Decoder가 cross-attention으로 참조하며 생성하는 구조. Span Corruption으로 사전학습하며 번역·요약에 강함. T5, BART가 대표적임.
- **MoE (Mixture-of-Experts)**: FFN layer를 N개의 expert FFN으로 교체하고 각 token마다 top-k expert만 활성화하는 구조. 같은 추론 비용으로 더 많은 파라미터를 갖게 되어 "학습 비용은 작게, 파라미터는 많게"를 달성함.
- **MoE Load Balancing Loss**: 특정 expert에만 token이 몰리는 expert collapse를 방지하기 위한 auxiliary loss. `L_balance = α · Σᵢ f_i · p_i` 형태로 모든 expert가 균등하게 사용되도록 유도함.

### Alternative Architectures (Non-Transformer)

- **SSM (State Space Model)**: hidden state를 재귀적으로 업데이트하는 구조로, 학습 시에는 parallel convolution, 추론 시에는 O(1) 고정 메모리 recurrent computation으로 동작함.
- **Mamba (Selective SSM)**: 기존 SSM의 A, B, C 행렬을 입력에 따라 동적으로 결정하는 selective state space. 중요한 정보를 선택적으로 기억/망각할 수 있으며, Transformer 대비 추론 메모리가 O(n) → O(d_state)로 고정됨.
- **Mamba-2**: SSD(Structured State Space Duality)로 SSM과 linear attention의 수학적 동등성을 보이고, 더 큰 state dimension을 효율적으로 처리하는 개선 버전.
- **RWKV**: Transformer의 병렬 학습과 RNN의 O(1) 추론을 결합한 구조. 학습 시에는 행렬 곱으로 병렬 계산, 추론 시에는 완전한 RNN으로 변환됨.
- **Hybrid Architecture (Jamba, Zamba)**: SSM layer와 Attention layer를 교차 배치하여 SSM의 효율 + Attention의 정확한 recall 능력을 결합한 구조.

---

## Tokenization

- **BPE (Byte Pair Encoding)**: 가장 빈번한 인접 symbol 쌍을 반복 merge하여 vocabulary를 구성하는 방법. GPT 계열, LLaMA 등이 사용함.
- **WordPiece**: BPE와 유사하나 빈도 대신 likelihood를 최대로 높이는 쌍을 merge하는 방법. BERT가 사용하며 subword 앞에 `##`를 붙임.
- **SentencePiece**: 공백 없이 raw text를 직접 처리하여 언어에 독립적으로 동작하는 tokenizer. 한국어/중국어 등 공백이 없는 언어에 특히 유리하며 T5, LLaMA, Gemma가 사용함.
- **Byte-Level Tokenization**: 0~255 byte를 기본 vocabulary로 사용하여 `<UNK>` token이 없이 어떤 텍스트든 표현 가능한 방법. GPT-2가 도입함.
- **Vocabulary Size Tradeoffs**: vocab이 클수록 같은 텍스트를 더 적은 token으로 표현 가능하여 context window를 효율적으로 사용함. 추세는 32K(LLaMA-2) → 128K(LLaMA-3) → 152K(Qwen2)로 확대됨.
- **Chat Template**: Instruction-tuned 모델들이 대화 구조를 위해 사용하는 special token 형식. LLaMA-3의 `<|start_header_id|>`, ChatML의 `<|im_start|>` 등이 있으며 SFT 데이터는 이 형식을 반드시 따라야 함.
- **SFT Loss Masking**: Fine-tuning 시 user turn에는 loss를 계산하지 않고 assistant 응답 부분에만 loss를 계산하는 방법. 이래야 "user처럼 말하는 것"이 아니라 "assistant처럼 응답하는 것"을 학습함.

---

## Pretraining

- **Causal Language Modeling (CLM)**: 이전 token들로부터 다음 token을 예측하는 decoder-only LLM의 pretraining objective. 하나의 시퀀스에서 n개의 학습 예시를 동시에 얻을 수 있어 데이터 효율적임.
- **Masked Language Modeling (MLM)**: token의 15%를 [MASK]로 가리고 양방향 context를 이용해 예측하는 BERT 계열의 pretraining objective. CLM과 달리 생성에는 부적합함.
- **Span Corruption (T5)**: 연속된 token span을 sentinel token으로 대체하고 decoder가 복원하는 T5의 pretraining objective. Encoder-Decoder 구조에 적합함.
- **Data Deduplication**: 학습 데이터에서 중복 문서를 제거하는 작업. MinHash + LSH(near-duplicate 탐지)나 SimHash로 수행하며, 중복이 많으면 memorization이 심해지고 perplexity가 저하됨.
- **Data Filtering**: heuristic(길이, 특수문자 비율 등), perplexity-based(저품질 제거), classifier-based(고품질 분류기 점수) 방법으로 웹 데이터 품질을 높이는 작업. FineWeb, CCNet이 대표적 결과물임.
- **Data Mixture Strategy**: 웹, 책, 코드, Wikipedia 등 여러 도메인의 데이터를 적절한 비율로 섞는 전략. 코드 비율을 높이면 reasoning 능력이 향상되는 등 target task에 따라 최적 비율이 달라짐.
- **Curriculum Learning**: 쉬운 데이터에서 시작하여 점점 어려운 데이터를 학습하는 전략. LLaMA-3는 마지막 단계에 long context 데이터를 집중적으로 넣어 128K context를 지원함.

---

## Post-Training & SFT

- **Supervised Fine-Tuning (SFT)**: (instruction, response) 쌍 데이터로 base model을 fine-tuning하는 방법. response 부분에만 loss를 계산하는 loss masking이 핵심임.
- **Instruction Tuning**: 다양한 NLP task를 instruction 형식으로 변환하여 multi-task 학습하는 SFT 방법. FLAN이 대표적이며, 다양한 task의 수가 데이터 개수보다 더 중요함.
- **Self-Instruct**: LLM 자신으로 instruction 데이터를 자동 생성하여 SFT 데이터를 확보하는 방법. Stanford Alpaca(GPT-3로 52K 생성), WizardLM(Evol-Instruct로 점진적 복잡화)가 대표적임.
- **Rejection Sampling Fine-tuning**: 모델에서 N개 샘플을 생성하고 reward model/verifier로 필터링하여 좋은 것만 SFT 데이터로 사용하는 방법. 반복하면 iterative rejection sampling이 되며 LLaMA-2 chat에 extensively 사용됨.
- **Knowledge Distillation**: 큰 teacher 모델의 지식을 작은 student에 전달하는 방법. Soft label(teacher의 전체 확률 분포)이 hard label보다 더 많은 정보를 담음. DeepSeek-R1이 긴 CoT 출력을 소형 모델에 distillation하여 작은 reasoning 모델을 만들었음.
- **Multi-Turn Training**: 여러 번의 대화로 이루어진 데이터로 학습하는 방법. 모든 assistant turn에 loss를 계산하거나 마지막 turn만 학습하는 두 가지 방식이 있음.

---

## Alignment (RLHF / DPO / GRPO)

- **Reward Modeling**: (chosen, rejected) preference 쌍에서 사람의 선호도를 수치 점수로 예측하는 모델. Bradley-Terry model로 `L = -E[log σ(r(x,y_w) - r(x,y_l))]`로 학습함.
- **PPO (RLHF)**: Reward model + PPO로 policy를 최적화하는 4모델 파이프라인(policy, reference, reward, value). On-policy 방식이라 매번 샘플 생성이 필요하며 구현이 복잡하나 성능이 높음.
- **KL Regularization**: RLHF에서 policy가 reference(SFT model)에서 너무 벗어나지 않도록 `reward_total = r - β·KL(π‖π_ref)`로 제약하는 방법. reward hacking과 distribution collapse를 방지함.
- **DPO (Direct Preference Optimization)**: Reward model 없이 preference data에서 직접 policy를 학습하는 방법. `L_DPO = -E[log σ(β·log(π_θ(y_w)/π_ref(y_w)) - β·log(π_θ(y_l)/π_ref(y_l)))]`으로 구현이 단순함. Mistral, Zephyr, Gemma 등이 사용함.
- **GRPO (Group Relative Policy Optimization)**: 각 prompt에 G개 샘플을 생성하고 상대적 reward로 advantage를 추정하는 방법. Value model(critic)이 불필요하여 PPO 대비 메모리 절약. DeepSeek-R1에 사용됨.
- **RLVR (RL with Verifiable Rewards)**: 수학 정답 여부, 코드 실행 결과 등 자동으로 검증 가능한 binary reward로 policy를 최적화하는 방법. DeepSeek-R1, o1 계열의 핵심 학습 방식임.
- **KTO (Kahneman-Tversky Optimization)**: 쌍(pair) 없이 (prompt, response, good/bad) 단일 피드백으로 학습하는 방법. Annotation 비용이 DPO보다 낮음.
- **Constitutional AI (Anthropic)**: 모델 자신이 헌법 원칙에 따라 자기 응답을 비판하고 수정하게 하는 방법. RLAIF와 결합하여 사람 annotation 없이 harmless model을 학습함.
- **Reward Overoptimization**: PPO 학습이 과도하면 reward model score는 오르지만 실제 quality는 떨어지는 Goodhart's Law 현상. 응답이 비정상적으로 길어지거나 과장된 표현이 늘어남.

---

## PEFT

- **LoRA (Low-Rank Adaptation)**: Weight matrix 변화량 ΔW를 두 작은 행렬의 곱 `ΔW = BA`로 근사하는 PEFT 방법. 원래 파라미터는 frozen하고 r « min(d,k)인 B, A만 학습하여 256배 이상 파라미터를 절약함. Inference 시 `W_final = W₀ + BA`로 merge하여 추가 비용 없음.
- **QLoRA (Quantized LoRA)**: NF4(4-bit NormalFloat 양자화) + double quantization + paged optimizer를 결합한 방법. 65B 모델을 단일 48GB GPU에서 fine-tuning 가능하게 함.
- **Adapters**: Transformer layer 사이에 bottleneck 구조 `(d → r → d)` 모듈을 삽입하는 방법. Houlsby(attention/FFN 뒤 둘 다 삽입)와 Pfeiffer(FFN 뒤만) 변형이 있음.
- **Prefix Tuning**: 각 layer의 K, V sequence 앞에 학습 가능한 virtual token을 삽입하여 모델 behavior를 제어하는 방법. 입력 자체는 변경하지 않음.
- **Prompt Tuning**: Embedding layer에만 learnable soft token을 추가하는 가장 단순한 PEFT. 모델이 클수록 full fine-tuning에 가까워짐.
- **Catastrophic Forgetting**: Fine-tuning 후 pre-training에서 학습한 일반 지식이 손상되는 현상. LoRA는 weight 변화를 low-rank space로 제한하여 implicit regularization 효과가 있음. EWC(중요 파라미터 보호)나 replay buffer(사전학습 데이터 일부 혼합)로도 완화 가능함.

---

## Optimization & Training Stability

- **Mixed Precision (BF16)**: 학습 속도와 메모리를 절약하면서 수치 안정성을 유지하는 방법. BF16은 FP32와 같은 exponent 범위를 가져 overflow가 없어 현대 LLM 학습의 표준임. FP16은 overflow 위험이 있어 loss scaling이 필요함.
- **FP8 Training**: H100에서 네이티브 지원하는 8-bit float. BF16 대비 2배 빠르고 DeepSeek-V3가 학습 비용 절감에 활용함.
- **Gradient Clipping (Global Norm)**: 모든 파라미터의 gradient를 모아 전체 norm을 계산하고, threshold(보통 1.0)를 초과하면 scale down하는 방법. 방향은 유지하고 크기만 제한함.
- **Gradient Checkpointing**: Backward pass를 위한 중간 activation을 저장하지 않고 그때그때 재계산하는 방법. 메모리를 O(n_layers) → O(√n_layers)로 줄이는 대신 forward pass를 1.33배 더 수행함.
- **Gradient Accumulation**: N번의 small batch에 걸쳐 gradient를 누적하고 N번마다 한 번 update하는 방법. 물리적 batch size 제한을 극복하여 `effective batch = physical_batch × accum_steps × n_GPUs`를 달성함.
- **Weight Decay (Decoupled)**: AdamW에서 weight decay를 gradient에 더하지 않고 파라미터에 직접 적용하는 방식. Bias, LayerNorm의 γ/β에는 적용하지 않음.
- **Label Smoothing**: One-hot 레이블 대신 `(1-ε)·y + ε/K` softened label을 사용하여 모델의 overconfidence를 방지하고 calibration을 개선하는 방법.

---

## Distributed Training

- **Data Parallel (DP/DDP)**: 모델 전체를 각 GPU에 복제하고 데이터를 나눠 처리한 후 gradient를 평균내는 가장 단순한 분산 학습. DDP는 Ring-AllReduce로 효율적으로 통신하고 backward와 통신을 overlap함.
- **Tensor Parallel (TP)**: 하나의 weight matrix 자체를 여러 GPU에 column/row 방향으로 분할하는 방법. 레이어 내 통신이 발생하여 같은 node의 NVLink GPU에서만 효율적임. Megatron-LM이 표준 구현임.
- **Pipeline Parallel (PP)**: 모델의 레이어들을 여러 GPU에 순서대로 분산하는 방법. GPU 대기(bubble)가 발생하며 micro-batch를 많이 사용할수록 bubble 비율(`(p-1)/(m+p-1)`)이 감소함. 1F1B schedule로 메모리를 절약함.
- **Sequence Parallel / Context Parallel**: 매우 긴 시퀀스를 여러 GPU에 분산하는 방법. Ring Attention으로 각 GPU가 local window를 처리하면서 K, V를 순차 전달함.
- **ZeRO (Zero Redundancy Optimizer)**: DeepSpeed의 메모리 최적화 기법. Stage 1(optimizer state 분산), Stage 2(+gradient 분산), Stage 3(+파라미터 분산)으로 단계적으로 메모리를 절약함. ZeRO-Infinity는 CPU RAM과 NVMe도 활용함.
- **FSDP**: PyTorch의 ZeRO Stage 3 유사 구현. 파라미터, gradient, optimizer state를 모두 N개 GPU에 분산함.
- **3D Parallelism**: TP × PP × DP를 결합한 전략. Megatron-LM이 구현하며 LLaMA-3 405B 같은 초대형 모델 학습에 사용됨.
- **AllReduce / AllGather / ReduceScatter**: 분산 학습의 핵심 collective operation들. AllReduce는 gradient 평균에, AllGather는 분산된 파라미터 수집에, ReduceScatter는 ZeRO의 효율적 통신에 사용됨.

---

## Inference & Decoding

- **Greedy Decoding**: 매 step마다 가장 높은 확률의 token을 선택하는 방법. 결정론적이고 빠르지만 반복 생성에 취약하고 open-ended generation에 부적합함.
- **Beam Search**: 상위 k개 가설을 동시에 유지하면서 탐색하는 방법. Greedy보다 global solution에 가깝지만, 번역·요약에는 유리하나 자유로운 대화 생성에는 다양성이 부족함.
- **Top-k Sampling**: 상위 k개 token 중에서만 랜덤 샘플링하는 방법. 분포가 뾰족할 때와 넓을 때를 구분하지 못한다는 단점이 있음.
- **Top-p (Nucleus) Sampling**: 누적 확률이 p를 넘을 때까지의 token으로만 샘플링하는 방법. 분포에 따라 후보 수가 자동 조정되어 top-k보다 robust하며 현재 가장 널리 사용됨.
- **Temperature Scaling**: Softmax 전에 logits를 T로 나눠 분포를 날카롭게(T<1) 또는 평탄하게(T>1) 만드는 방법. 코딩/수학은 T≈0, 창작은 T≈1.1을 사용함.
- **Constrained Decoding (JSON Schema)**: JSON 문법을 FSM으로 표현하고 각 step에서 invalid token의 logit을 -inf로 설정하여 항상 valid JSON을 생성하게 강제하는 방법. Outlines, SGLang이 구현함.

---

## Inference Efficiency

- **KV-Cache**: Decode 시 이전에 계산한 K, V를 재사용해 중복 계산을 제거하는 방법. 메모리는 `2 × n_layers × n_kv_heads × d_head × seq_len × dtype_bytes`에 비례함.
- **Prefill vs Decode**: Prefill(전체 prompt 처리)은 compute-bound, Decode(token 1개씩 생성)는 memory-bandwidth-bound로 병목이 다름. 각각 다른 최적화 전략이 필요함.
- **PagedAttention (vLLM)**: OS의 virtual memory에서 영감을 받아 KV-cache를 고정 크기 block으로 관리하는 방법. 메모리 fragmentation을 4% 미만으로 줄여 동시 처리 가능한 request 수를 크게 늘림.
- **Continuous Batching**: 각 iteration마다 완료된 request를 제거하고 새 request를 추가하는 방법. Static batching의 GPU idle time을 없애 throughput을 수십 배 향상시킴.
- **Speculative Decoding**: 작은 draft model로 여러 token을 미리 제안하고 큰 target model이 한 번의 forward pass로 accept/reject하는 방법. Target model의 분포를 정확히 유지하면서 2~3배 throughput을 높임.
- **EAGLE / Medusa**: Target model 자체에 draft head를 학습하거나(EAGLE) 여러 parallel decode head를 추가(Medusa)하여 speculative decoding을 수행하는 방법.

---

## Quantization

- **Post-Training Quantization (PTQ)**: 학습 완료된 모델을 재학습 없이 양자화하는 방법. Calibration dataset으로 activation 분포를 측정한 뒤 scale factor를 결정함.
- **Weight-Only Quantization**: Weight만 INT4/INT8로 양자화하고 activation은 FP16 유지. 추론 시 dequantize 후 연산하며 memory-bandwidth-bound인 decode에 특히 효과적임.
- **Activation Quantization (W8A8)**: Weight와 activation 둘 다 INT8로 양자화하여 INT8 tensor core를 완전 활용. LLM의 activation에 outlier가 있어 LLM.int8()처럼 mixed precision으로 처리함.
- **KV Quantization**: KV-cache를 INT8/INT4로 양자화하여 메모리를 2~4배 절약하는 방법. Key가 Value보다 quantization에 더 민감함.
- **AWQ (Activation-aware Weight Quantization)**: Activation 크기가 큰 salient weight를 per-channel scale로 보호하여 INT4에서도 FP16에 가까운 성능을 유지하는 방법.
- **GPTQ**: Hessian을 활용하여 weight를 순서대로 양자화하면서 오차를 나머지 weight에 분산 보상하는 방법. 단순 라운딩보다 훨씬 낮은 양자화 오차를 달성함.
- **FP8**: H100에서 네이티브 지원하는 8-bit float 포맷. E4M3(정밀도 중시)와 E5M2(범위 중시) 두 종류가 있으며 DeepSeek-V3가 FP8 학습으로 비용을 크게 줄임.

---

## Serving Frameworks

- **vLLM**: PagedAttention과 continuous batching이 핵심인 오픈소스 serving의 사실상 표준. Prefix caching, LoRA serving, speculative decoding을 지원하며 OpenAI-compatible API를 제공함.
- **TGI (HuggingFace)**: HuggingFace 생태계와 통합된 Docker 기반 serving solution. Flash Attention, tensor parallel, continuous batching을 지원함.
- **TensorRT-LLM (NVIDIA)**: H100 같은 최신 NVIDIA GPU에서 최고 성능을 내는 라이브러리. Kernel fusion으로 여러 연산을 하나의 kernel로 통합하여 메모리 round-trip을 최소화함.
- **SGLang**: Radix tree 기반 prefix caching(RadixAttention)과 structured generation DSL이 특징. 긴 공통 system prompt가 있는 경우에 특히 효율적임.
- **Triton (OpenAI)**: Python과 유사한 문법으로 custom GPU kernel을 작성하는 언어/컴파일러. FlashAttention-2가 Triton으로 구현되어 있음.
- **CUDA Kernel Fusion**: 여러 GPU 연산을 하나의 kernel로 합쳐 HBM 접근 횟수를 줄이는 기법. LayerNorm+Linear, QKV projection 등을 fused kernel로 구현하면 2~4배 speedup 가능함.

---

## Serving Metrics

- **Throughput (tokens/sec)**: 단위 시간당 생성된 output token 수. batch size, KV-cache 용량, quantization에 의해 결정되며 GPU utilization을 나타냄.
- **TTFT (Time To First Token)**: 첫 번째 output token이 반환될 때까지의 시간. Prefill 길이에 비례하며 streaming 서비스의 사용자 경험과 직결됨.
- **Tail Latency (P95/P99)**: 요청의 95% 또는 99%가 완료되는 시간. SLA의 기준이 되며 median보다 중요한 서비스 품질 지표임.
- **Cost per Token**: GPU 시간당 비용을 throughput으로 나눈 값. Quantization, speculative decoding, continuous batching으로 최적화 가능함.
- **Chunked Prefill**: 긴 prompt를 청크로 나눠 prefill과 decode를 번갈아 처리하는 방법. 긴 prompt가 다른 request의 decode를 blocking하는 것을 줄여 TTFT를 개선함.

---

## Long Context & RAG

- **Context Window 확장**: 2K(GPT-3) → 1M(Gemini 1.5)까지 확대된 역사. O(n²) attention memory 문제와 RoPE extrapolation 문제가 핵심 과제였음.
- **Lost in the Middle**: Context window가 길어지면 중간 부분의 정보를 잘 활용하지 못하는 현상. RAG 시스템에서는 가장 관련성 높은 chunk를 context의 앞/뒤에 배치해야 함.
- **RAG (Retrieval-Augmented Generation)**: 외부 knowledge base에서 관련 정보를 검색하여 LLM 생성에 활용하는 방법. Hallucination을 줄이고 최신 정보를 반영할 수 있음.
- **Chunking**: 긴 문서를 검색 가능한 단위로 나누는 작업. Fixed-size(단순하나 의미 파괴 위험), sentence/paragraph-aware(자연스러운 경계 존중), semantic(의미 기반) 방식이 있음.
- **Embeddings & Vector DB**: 텍스트를 벡터로 변환하여 저장하고 유사도 검색하는 인프라. HNSW/IVF 알고리즘으로 근사 최근접 이웃을 효율적으로 찾음. Faiss, Pinecone, Weaviate 등이 대표적임.
- **Reranking (Cross-Encoder)**: Bi-encoder로 빠르게 후보를 검색한 후 cross-encoder로 query-document 쌍을 정밀 scoring하는 2단계 검색 방법. 정확도를 크게 향상시킴.
- **HyDE (Hypothetical Document Embeddings)**: 실제 문서 대신 LLM이 생성한 가상의 답변 문서를 query로 사용하여 embedding 검색 품질을 높이는 방법.

---

## Reasoning & Test-Time Compute

- **Chain-of-Thought (CoT)**: LLM에게 단계별 풀이 과정을 생성하게 하여 복잡한 reasoning 성능을 향상시키는 방법. "Let's think step by step"이라는 zero-shot trigger가 효과적임. 충분히 큰 모델(~100B+)에서만 효과적임.
- **Self-Consistency**: 같은 질문에 여러 CoT 경로를 샘플링하고 최종 답의 다수결(majority vote)로 결정하는 방법. 단일 CoT보다 일관적으로 더 높은 성능을 달성함.
- **Tree-of-Thought (ToT)**: 추론 과정을 tree로 구성하고 BFS/DFS/MCTS로 탐색하는 방법. 중간 단계를 평가하여 유망한 경로를 선택적으로 탐색함.
- **PRM (Process Reward Model)**: 추론의 각 step마다 점수를 매기는 방법. 오류 위치를 정확히 파악하고 잘못된 경로를 조기 차단 가능하나, step-level annotation이 매우 어려움.
- **ORM (Outcome Reward Model)**: 최종 답만 보고 점수를 매기는 방법. 레이블 수집이 쉽지만 틀린 추론 과정도 "행운"으로 맞을 수 있다는 단점이 있음.
- **Best-of-N (BoN)**: N개의 샘플을 생성하고 가장 좋은 것을 선택하는 방법. 성공 확률 `P(success) = 1 - (1-p)^N`으로 N을 늘릴수록 성능이 향상됨.
- **Test-Time Compute Scaling**: 같은 모델에 더 많은 추론 계산을 투입하여 성능을 높이는 패러다임. 작은 모델 + 많은 inference compute가 큰 모델 + 적은 compute를 이길 수 있음(3B+256x ≈ 34B+1x).
- **Long CoT / Thinking Model**: 추론 시간에 긴 내부 "생각" 과정(extended thinking)을 생성하도록 학습된 모델. DeepSeek-R1, o1/o3가 대표적임. GRPO + RLVR로 학습함.

---

## Safety

- **Hallucination**: 사실이 아닌 내용을 자신있게 생성하는 현상. Factual(잘못된 사실), Intrinsic(소스와 모순된 요약), Extrinsic(소스에 없는 정보 추가) 유형이 있음. RAG와 RLHF로 완화함.
- **Jailbreak Attacks**: 모델의 안전 가이드라인을 우회하는 공격. Role-playing("DAN 역할"), obfuscation(Base64/역방향), multi-turn 점진적 유도, translation attack 등의 방법이 있음.
- **Prompt Injection**: RAG/agent 시스템에서 외부 문서·웹페이지에 숨겨진 명령으로 모델 행동을 조작하는 공격. Instruction hierarchy(시스템 프롬프트 > 사용자 > 외부 데이터)와 명확한 delimiter로 방어함.
- **Red Teaming**: 모델 취약점을 찾기 위해 의도적으로 공격적 입력을 시도하는 평가 방법. Manual(전문가 직접 시도)과 Automated(Attacker LLM + Target LLM + Judge LLM 루프) 방식이 있음.
- **Bias Evaluation**: 모델이 성별, 인종 등 특정 그룹에 대해 편향된 응답을 생성하는 문제. BBQ, WinoBias 벤치마크로 평가함.
- **Calibration**: 모델의 confidence가 실제 정확도와 얼마나 일치하는지의 척도. ECE로 측정하며 낮을수록 좋음. Label smoothing, temperature scaling으로 개선함.
- **Abstention**: 모델이 불확실하거나 유해한 요청일 때 거부하는 능력. SFT 데이터에 명시적 거부 예시를 포함하고 RLHF에서 거부에 높은 reward를 부여하여 학습함.

---

## Benchmarks & Evaluation

- **MMLU**: 57개 학문 분야의 4지선다 문제로 지식 폭을 평가하는 벤치마크. 2025년 기준 frontier model들이 모두 87~88%에 수렴하여 saturation됨.
- **GPQA (Diamond)**: 전문가 수준의 과학 문제 벤치마크. 인간 전문가도 ~65%인데 o3가 이를 초과하여 frontier model 구분의 새 기준이 되고 있음.
- **MATH / AIME**: 수학 능력 평가. GSM8K(초등)→MATH(고등)→AIME(올림피아드) 순으로 난이도가 높아짐. o1/o3가 AIME에서 전문가 수준에 도달함.
- **HumanEval / LiveCodeBench**: 코딩 능력 평가. HumanEval은 benchmark contamination 문제가 있고, LiveCodeBench는 최근 문제를 사용하여 contamination을 방지함.
- **Benchmark Contamination**: 모델의 학습 데이터에 벤치마크 test set이 포함되어 암기 vs 이해를 구분할 수 없는 문제. n-gram overlap, min-k% prob으로 탐지하고 LiveCodeBench처럼 동적 문제 생성으로 완화함.
- **LLM-as-a-Judge Biases**: LLM을 평가자로 사용할 때 발생하는 편향들. Position bias(순서 편향), verbosity bias(긴 답 선호), self-enhancement bias(자기 스타일 선호), sycophancy(사용자 의견 추종)가 대표적임. 여러 순서로 평가하거나 ensemble로 완화함.
- **Benchmark Saturation**: 모델들이 벤치마크의 ceiling에 근접하여 성능 차이가 구분되지 않는 현상. MMLU → MMLU-Pro, GSM8K → MATH → AIME로 점점 어려운 벤치마크가 필요해짐.
- **lm-evaluation-harness**: EleutherAI의 표준 LLM 평가 프레임워크. 60개 이상의 벤치마크를 지원하며, 같은 모델도 shot 수와 prompt 형식에 따라 결과가 달라지므로 표준 harness 사용이 중요함.

---

## LLM Model Families

- **GPT 계열 (OpenAI)**: GPT-1(언어 모델+fine-tuning), GPT-2(zero-shot), GPT-3(few-shot ICL, 175B), InstructGPT(RLHF 도입), ChatGPT, GPT-4(multimodal), GPT-4o(native multimodal), o1/o3(Long CoT, RLVR). 매 세대마다 새로운 패러다임을 도입했음.
- **LLaMA 계열 (Meta)**: LLaMA-1(inference-optimal, 오픈소스 혁신), LLaMA-2(GQA, 4K context, chat 버전), LLaMA-3(128K vocab, 128K context, 405B), LLaMA-3.1/3.2/3.3(경량화 및 vision 버전). 오픈소스 LLM의 사실상 표준이 됨.
- **Mistral / Mixtral (Mistral AI)**: Mistral 7B(sliding window attention, GQA로 크기 대비 최고 성능), Mixtral 8x7B(MoE 도입, 46.7B 총 파라미터/12.9B 활성). 효율성에 집중한 유럽 모델임.
- **DeepSeek 계열**: DeepSeek-V2(MLA attention으로 KV-cache 대폭 절약, MoE), V3(FP8 학습, 671B MoE를 저비용 학습), R1(GRPO+RLVR로 Long CoT, 오픈소스 공개). 비용 효율성과 오픈소스로 큰 파장을 일으킴.
- **Gemini 계열 (Google)**: Gemini 1.0(native multimodal), 1.5(1M context, Mixture-of-Experts), 2.0(native audio I/O, agentic 능력 강화). Google의 멀티모달 통합 전략을 보여줌.
- **Qwen 계열 (Alibaba)**: Qwen1/1.5/2/2.5(한중영 강점, 최대 72B), QwQ(reasoning 특화), Qwen2.5-Max(frontier 수준). 아시아 언어 강점이 특징임.
- **Claude 계열 (Anthropic)**: Claude 3(Haiku/Sonnet/Opus, Constitutional AI), Claude 3.5(Sonnet 2.0, 대폭 향상), Claude 3.7(hybrid reasoning, extended thinking 지원). 안전성과 추론 능력에 집중함.
- **Phi 계열 (Microsoft)**: Phi-1~4(소형이지만 교과서 수준 데이터로 강한 성능). "textbook quality" 합성 데이터로 작은 모델의 가능성을 보여줌.
- **Gemma 계열 (Google)**: Gemma 1/2/3(7B 이하 소형 모델, 오픈소스). Gemma 3은 vision 지원을 포함함.

---

# VLM

## Vision Encoder

- **CNN vs ViT**: CNN은 locality bias 덕분에 소규모 데이터에 강하지만 장거리 의존성이 약함. ViT는 이미지를 patch sequence로 처리하고 global self-attention으로 장거리 관계를 직접 계산하여 대규모 데이터에서 CNN을 초과함.
- **Patch Embedding**: 이미지를 `P×P` patch로 분할 후 flatten → linear projection → positional embedding을 거쳐 LLM이 처리할 수 있는 token sequence로 변환하는 ViT의 핵심 입력 처리 방법.
- **CLIP (Contrastive Language-Image Pretraining)**: 이미지-텍스트 쌍 수백만 개로 InfoNCE loss를 사용하여 contrastive 사전학습하는 방법. 같은 쌍의 image-text embedding은 가깝게, 다른 쌍은 멀게 학습함. VLM의 vision encoder로 가장 많이 사용됨.
- **SigLIP**: CLIP의 softmax 기반 InfoNCE 대신 sigmoid loss를 사용하는 방법. 대규모 batch 없이도 효과적으로 학습 가능하고 성능이 더 좋아 최신 VLM들이 채택함.
- **ViT-L/14 vs ViT-H/14**: patch size 14인 ViT-Large와 ViT-Huge. VLM의 vision encoder로 가장 많이 쓰이는 규격이며, 작은 patch는 더 세밀한 표현을 제공함.

---

## VLM Architecture (Projector / Connector)

- **Linear Projector**: Vision encoder 출력을 LLM embedding 차원으로 linear 변환하는 가장 단순한 연결 방법. LLaVA 초기 버전이 사용하여 단순함이 충분히 효과적임을 증명함.
- **MLP Projector**: Linear에 비선형 활성화를 추가한 2-layer MLP 연결 방법. LLaVA-1.5가 사용하여 linear보다 표현력이 향상됨.
- **Q-Former (BLIP-2)**: 고정된 수의 learnable query token이 cross-attention으로 image feature를 압축하는 방법. 이미지 token 수를 32개처럼 고정으로 줄여 LLM의 context 부담을 크게 줄임.
- **Perceiver Resampler (Flamingo)**: Q-Former와 유사하나 learnable query가 cross-attention으로 visual feature를 sampling하는 방법. 다양한 해상도/수의 이미지에서 고정 수의 visual token을 출력함.
- **Early Fusion vs Late Fusion**: Early fusion은 이미지와 텍스트를 초반 레이어부터 함께 처리(더 깊은 통합), late fusion은 각자 인코딩 후 합치는 방식임. 현재 주류는 projector로 연결 후 LLM에서 함께 처리하는 late fusion 계열임.
- **Unified Multimodal Transformer**: 별도의 vision encoder 없이 하나의 Transformer가 이미지 patch와 텍스트 token을 처음부터 함께 처리하는 완전 통합 구조. 더 깊은 cross-modal interaction이 가능함.

---

## Resolution Handling

- **고해상도의 중요성**: 기본 224×224로 강제 리사이즈하면 OCR·문서 이해에서 텍스트가 뭉개져서 판독 불가. 고해상도 처리 전략이 VLM 실용성의 핵심임.
- **AnyRes / Dynamic Tiling**: 이미지를 여러 tile로 분할하고 각 tile을 독립적으로 인코딩한 후, global thumbnail과 함께 LLM에 전달하는 방법. LLaVA-NeXT, InternVL이 사용함.
- **Aspect Ratio Preservation**: 이미지를 왜곡 없이 원래 비율로 처리하는 방법. 미리 정의된 resolution grid(예: 1×1, 2×1, 1×2, 2×2) 중 closest를 선택하여 padding으로 맞춤.
- **Token Pruning/Compression**: 시각 token이 너무 많아지는 문제를 해결하기 위해 redundant visual token을 제거하거나 병합하는 방법. Attention score 기반, 유사도 기반, learned compression 방식이 있음.
- **Multi-Scale Patching**: 여러 해상도의 patch를 함께 사용하여 coarse와 fine-grained 정보를 모두 포착하는 방법.

---

## Multimodal Training

- **2-Stage Training Pipeline**: Stage 1에서는 vision encoder와 LLM을 frozen하고 projector만 학습(vision-text alignment), Stage 2에서는 projector와 LLM을 함께 학습(instruction following). LLaVA가 확립한 VLM 학습의 표준 방법임.
- **Stage 1 (Alignment Pre-training)**: 대규모 이미지-캡션 쌍(CC3M, LAION 등)으로 projector만 학습하여 시각 특징을 LLM 토큰 공간에 정렬하는 단계.
- **Stage 2 (Instruction Fine-tuning)**: VQA, image captioning, OCR, 차트 이해 등 다양한 멀티모달 instruction 데이터로 LLM을 포함하여 fine-tuning하는 단계.
- **VLM Hallucination**: 이미지에 없거나 부정확한 정보를 사실처럼 생성하는 현상. Object hallucination(없는 물체 언급)이 가장 흔하며 POPE 벤치마크로 평가함.
- **POPE (Polling-based Object Probing Evaluation)**: 이미지에 특정 물체가 있는지 yes/no로 묻는 hallucination 평가 방법. Random/Popular/Adversarial 세 가지 설정으로 난이도가 다름.
- **Catastrophic Forgetting in VLM**: Vision encoder나 LLM을 fine-tuning할 때 사전학습에서 얻은 표현이 손상되는 문제. Vision encoder를 처음에 frozen하거나 낮은 LR로 fine-tuning하고 LoRA를 사용하여 완화함.
- **Grounding**: 텍스트의 특정 표현이 이미지의 어떤 영역(bounding box)에 대응하는지 학습하는 능력. VQA 이상의 정밀한 시각 이해에 필요함.
- **Multi-Image Position IDs**: 여러 이미지가 입력될 때 이미지 간 position encoding이 겹치지 않도록 position ID를 부여하는 방법. 비디오 이해에도 활용됨.
- **Video Processing**: 비디오를 frame으로 분할하고 frame sampling(uniform, fps-based), temporal pooling(인접 frame feature 집약)으로 token 수를 줄이는 방법. Qwen2-VL은 M-RoPE로 시간 차원을 position encoding에 통합함.

---

## VLM Benchmarks

- **MMMU**: 57개 전공 영역의 대학원 수준 멀티모달 문제. 도메인 지식 + 시각 추론이 필요하여 단순 인식을 넘어선 능력을 평가함.
- **MMBench / MMStar**: VLM의 20개 세부 능력을 평가하는 벤치마크. MMStar는 데이터 오염 방지를 위해 정교하게 설계됨.
- **TextVQA / DocVQA**: OCR 기반 텍스트 인식 능력을 평가하는 벤치마크. 이미지 내 텍스트를 읽고 질문에 답해야 하여 고해상도 처리 능력이 중요함.
- **ChartQA**: 차트·그래프에서 정보를 extractive하게 읽거나 reasoning하여 답변하는 능력을 평가함.

---

## VLM Model Families

- **LLaVA 계열**: Linear projector(LLaVA-1.0) → MLP projector(LLaVA-1.5) → AnyRes+고해상도(LLaVA-NeXT) → video 지원까지 확장한 오픈소스 VLM의 흐름. 아키텍처 단순성 속에 탁월한 성능을 보여줌.
- **InternVL 계열 (Shanghai AI Lab)**: InternViT(6B 대형 vision encoder) + LoRA fine-tuning으로 높은 성능. InternVL 2/2.5/3로 발전하며 dynamic resolution, multi-image, video 지원. 오픈소스 최강 VLM 중 하나임.
- **Qwen-VL / Qwen2-VL / Qwen2.5-VL**: M-RoPE로 2D 이미지와 3D 비디오를 통합 처리하는 능력이 특징. Qwen2-VL이 video understanding에서 오픈소스 최강 성능을 달성함.
- **GPT-4V / GPT-4o**: GPT-4V는 late fusion VLM, GPT-4o는 텍스트·이미지·오디오를 하나의 모델로 처리하는 native multimodal. 실시간 비디오 스트리밍과 voice conversation을 지원함.
- **Gemini 1/1.5/2**: Native multimodal로 처음부터 설계됨. Gemini 1.5는 1M context로 1시간 분량 비디오 처리 가능. Gemini 2.0은 oudio output도 지원함.
- **Claude 3/3.5/3.7 Vision**: Claude 3는 Haiku/Sonnet/Opus 3종. Claude 3.7은 extended thinking(hybrid reasoning)을 지원하며 시각 이해 능력이 대폭 향상됨.
- **LLaMA-3.2 Vision / Gemma 3 Vision**: Meta와 Google의 오픈소스 vision 모델. Gemma 3은 2~27B 규모로 multimodal을 지원함.
- **Pixtral (Mistral)**: Mistral의 VLM으로 native variable resolution 이미지 처리를 지원함.

---

# VLA

## VLA 기초

- **Behavior Cloning (BC)**: 전문가 시연 데이터로 `π_θ(action | observation)`을 지도학습으로 모방하는 가장 단순한 로봇 학습 방법. 구현이 단순하나 Covariate Shift(분포 이탈로 오류 누적) 문제가 있음.
- **Covariate Shift**: BC에서 작은 실행 오류가 학습 분포를 벗어나게 하고, 이것이 더 큰 오류로 누적되는 문제. DAgger(interactive 전문가 피드백)로 완화 가능함.
- **Offline RL (CQL, IQL)**: 환경과 직접 상호작용하지 않고 미리 수집된 데이터셋만으로 policy를 학습하는 방법. 데이터셋 외 (s,a) 쌍에 대한 Q-value 과대 추정 문제를 Conservative Q-Learning 등으로 해결함.
- **Autoregressive Action Modeling**: LLM처럼 action을 token by token으로 autoregressive 생성하는 방법. 기존 LLM 아키텍처를 재사용 가능하나 연속 action의 정밀도 손실과 느린 생성이 단점임.
- **Sim-to-Real Transfer**: 시뮬레이터에서 학습한 policy를 실제 로봇에 전이하는 방법. Domain randomization(물체 색상·질감·조명 등을 다양화)으로 sim-real 간 gap을 줄임.
- **World Model**: 환경의 미래 상태를 예측할 수 있는 내부 모델. RSSM, Dreamer 등이 있으며 Embodied AI에서 planning과 data efficiency 향상에 활용됨.
- **Hierarchical Control**: 고수준(task planning, LLM) - 중간수준(motion planning) - 저수준(joint control, 10Hz)으로 분리하는 구조. 복잡한 long-horizon task를 단계별로 분해함.

---

## Action Representation

- **Discrete Action Tokenization**: 연속 action 값을 N개 bin으로 양자화하여 LLM vocabulary에 추가하는 방법. 기존 LLM 아키텍처 재사용이 가능하나 정밀도 손실과 느린 autoregressive 생성이 단점임.
- **Continuous Regression**: LLM의 마지막 hidden state에 regression head를 붙여 action을 연속 벡터로 직접 예측하는 방법. 빠르고 정밀하나 multimodal action distribution 처리가 어려움.
- **Diffusion Policy**: Action을 DDPM/DDIM 방식으로 random noise에서 반복적 denoising으로 생성하는 방법. Multimodal action distribution(같은 상황의 여러 valid action)을 포착 가능한 것이 핵심 장점임.
- **Flow Matching**: Gaussian noise에서 data까지 직선 경로(straight-path ODE)로 이동하는 방법. Diffusion보다 샘플링 step이 적고 빠르며 π0가 사용함.
- **Chunked Action Prediction**: 한 번에 H개의 미래 action을 예측하고 receding horizon 방식으로 실행하는 방법. Temporal consistency를 높이고 고주파 제어에서 autoregressive의 latency 문제를 완화함.

---

## VLA Representative Models

- **RT-1 (Google 2022)**: EfficientNet + TokenLearner(512→8 token 압축) + causal Transformer 구조로 130K real robot 에피소드를 학습하여 700개 이상의 task를 97% 성공률로 수행하는 첫 대규모 multitask robot model임.
- **RT-2 (Google 2023)**: PaLI-X(55B) 또는 PaLM-E(562B) VLM에 robot action token을 vocabulary에 추가하고 co-finetuning하는 최초의 진정한 VLA. 웹 지식을 로봇으로 전이하여 학습에 없던 조합에도 generalization함.
- **Octo (UC Berkeley 2023)**: Modular 설계의 오픈소스 generalist robot policy. Open X-Embodiment 데이터로 학습하고 action을 diffusion head로 예측하며 누구나 새 로봇에 fine-tuning 가능함.
- **OpenVLA**: SigLIP(vision encoder) + LLaMA-2 7B 기반의 오픈소스 VLA. Discrete autoregressive action tokenization을 사용하며 RT-2급 성능을 오픈소스로 제공함.
- **π0 (Physical Intelligence 2024)**: PaliGemma VLM + flow matching action expert module 구조. 언어 이해와 정밀 manipulation을 분리하여 처리하며 dexterous manipulation에서 최고 성능을 달성함.
- **π0-FAST**: π0의 후속으로 action을 language token으로 변환하여 언어 모델과 통합하는 방법. Action expert module 없이도 빠르고 효율적인 action 생성을 달성함.

---

## Embodied AI Topics

- **Tool Use (Embodied)**: LLM의 API tool use를 물리적 도구 사용으로 확장하는 개념. pick-and-place, 도구 조작, 버튼 클릭 등 다양한 physical interaction을 언어로 명세하여 실행함.
- **Hierarchical Planning (SayCan 등)**: LLM이 high-level plan을 생성하고, affordance model이 현재 환경에서 실행 가능한지 평가하여 최종 action을 결정하는 방법. "Make coffee" 같은 long-horizon task를 분해함.
- **Language-Conditioned Manipulation**: 자연어 명령을 이해하고 그에 맞는 manipulation을 수행하는 능력. Zero-shot(새 물체), compositional(새 조합), analogical(유추), language-only(텍스트만으로 이해) 4가지 generalization 수준이 있음.
- **Data Collection Challenges**: Robot 학습 데이터 수집의 어려움. Kinesthetic teaching, VR teleoperation, human video와 robot video 매핑 등의 방법을 사용함. Open X-Embodiment가 여러 로봇의 데이터를 통합한 대표 데이터셋임.
