# LLM / VLM Basics (+ VLA) Roadmap

LLM/VLM Research Engineer 커리어를 위한 basic 개념 정리

---

## 전체 구조

```
llm-vlm-basics/
├── 00_foundations/          — 수학·정보이론·최적화 기초
├── 01_transformer_architecture/ — Transformer 아키텍처 심화
├── 02_training_pipeline/    — Pretraining, SFT, RLHF, PEFT
├── 03_distributed_training/ — 분산 학습 시스템
├── 04_inference_serving/    — 추론 최적화 및 서빙
├── 05_long_context_memory/  — Long Context, RAG, Memory
├── 06_reasoning_test_time_compute/ — Reasoning & Test-Time Compute
├── 07_llm_model_families/   — LLM 모델 변천사 및 분석
├── 08_vlm/                  — Vision-Language Models
├── 09_vla/                  — Vision-Language-Action Models
├── 10_benchmarks_evaluation/ — 벤치마크 및 평가 이슈
├── 11_safety/               — LLM 안전성
└── 12_emerging_trends/      — 2024+ 최신 트렌드
```

---

## 목차

### Section 0: Overall Foundations
[`00_foundations/foundations.md`](00_foundations/foundations.md)

- Probability Basics — MLE, Cross-Entropy, KL Divergence
- Information Theory — Entropy, Perplexity
- Optimization Basics — SGD, AdamW, LR Schedule, Warmup
- Scaling Laws — Chinchilla Scaling
- Compute vs Data Tradeoff
- Emergent Abilities
- In-Context Learning vs Fine-Tuning

---

## Part I: Core LLM

### Section 1: Transformer & Model Architecture

#### 1.1 Transformer Block
[`01_transformer_architecture/1.1_transformer_block.md`](01_transformer_architecture/1.1_transformer_block.md)
- Self-Attention, Multi-Head Attention, Cross-Attention
- Residual Connections
- LayerNorm vs RMSNorm / Pre-LN vs Post-LN
- Feedforward Network (FFN), SwiGLU / GeGLU
- Activation Functions (GELU, SiLU)
- Dropout, Weight Tying

#### 1.2 Attention Details
[`01_transformer_architecture/1.2_attention_details.md`](01_transformer_architecture/1.2_attention_details.md)
- Scaled Dot-Product Attention
- Attention Masking (Causal Mask, Padding Mask)
- Time Complexity O(n²)
- KV-Cache Mechanics
- Query/Key/Value Projection

#### 1.3 Attention Variants
[`01_transformer_architecture/1.3_attention_variants.md`](01_transformer_architecture/1.3_attention_variants.md)
- Multi-Query Attention (MQA)
- Grouped Query Attention (GQA)
- Sparse Attention, Sliding Window Attention
- Linear Attention (Performer)
- FlashAttention (v1/v2/v3)
- Memory-Efficient Attention Kernels

#### 1.4 Positional Encoding
[`01_transformer_architecture/1.4_positional_encoding.md`](01_transformer_architecture/1.4_positional_encoding.md)
- Sinusoidal PE, Learned PE
- RoPE (Rotary Position Embedding)
- ALiBi
- NTK Scaling, Long-Context RoPE Interpolation
- 2D Positional Encoding (Vision)
- Multimodal RoPE (M-RoPE)

#### 1.5 Tokenization
[`01_transformer_architecture/1.5_tokenization.md`](01_transformer_architecture/1.5_tokenization.md)
- BPE, WordPiece, SentencePiece
- Byte-Level Tokenization
- Vocabulary Size Tradeoffs
- Special Tokens Design

#### 1.6 Model Architecture Types
[`01_transformer_architecture/1.6_model_architecture_types.md`](01_transformer_architecture/1.6_model_architecture_types.md)
- Encoder-only, Decoder-only, Encoder-Decoder, Prefix LM
- Mixture-of-Experts (MoE)
- Routing Mechanisms, Capacity Factor

#### 1.7 Alternative Architectures
[`01_transformer_architecture/1.7_alternative_architectures.md`](01_transformer_architecture/1.7_alternative_architectures.md)
- State Space Models (SSM)
- Mamba / Mamba-2 (Selective SSM)
- RWKV
- Hybrid Architectures (Jamba, Zamba)

---

### Section 2: Training Pipeline

#### 2.1 Pretraining
[`02_training_pipeline/2.1_pretraining.md`](02_training_pipeline/2.1_pretraining.md)
- Next Token Prediction, Denoising Objective, Span Corruption
- Data Deduplication, Data Filtering
- Curriculum Learning, Data Mixture Strategies

#### 2.2 Post-Training
[`02_training_pipeline/2.2_post_training.md`](02_training_pipeline/2.2_post_training.md)
- Supervised Fine-Tuning (SFT)
- Instruction Tuning, Self-Instruct
- Distillation, Rejection Sampling
- Multi-Turn Training

#### 2.3 Alignment / RLHF
[`02_training_pipeline/2.3_alignment_rlhf.md`](02_training_pipeline/2.3_alignment_rlhf.md)
- Reward Modeling, Preference Data Collection
- PPO, DPO, GRPO, KTO
- RLAIF, Constitutional AI
- KL Regularization, Policy vs Reward Overoptimization
- Calibration & Abstention

#### 2.4 Parameter-Efficient Fine-Tuning (PEFT)
[`02_training_pipeline/2.4_peft.md`](02_training_pipeline/2.4_peft.md)
- LoRA, QLoRA
- Adapters, Prefix Tuning, Prompt Tuning
- IA³
- Catastrophic Forgetting Mitigation

#### 2.5 Optimization & Stability
[`02_training_pipeline/2.5_optimization_stability.md`](02_training_pipeline/2.5_optimization_stability.md)
- Mixed Precision (FP16, BF16, FP8)
- Gradient Clipping, Checkpointing, Accumulation
- EMA, Weight Decay, Label Smoothing, Loss Scaling
- Optimizers (AdamW, Adafactor)

---

### Section 3: Distributed Training & Systems
[`03_distributed_training/distributed_training.md`](03_distributed_training/distributed_training.md)
- Data Parallel (DP), DDP
- Tensor Parallel (TP), Pipeline Parallel (PP)
- Sequence Parallel, Context Parallel
- FSDP, ZeRO (Stage 1/2/3), ZeRO-Infinity
- DeepSpeed, Megatron-LM
- AllReduce / AllGather Communication
- Checkpoint Sharding, Fault Tolerance, Elastic Training

---

### Section 4: Inference & Serving

#### 4.1 Decoding
[`04_inference_serving/4.1_decoding.md`](04_inference_serving/4.1_decoding.md)
- Greedy, Beam Search
- Top-k Sampling, Top-p (Nucleus) Sampling
- Temperature Scaling, Repetition Penalty, Length Penalty
- Constrained Decoding, JSON Schema Decoding

#### 4.2 Efficiency
[`04_inference_serving/4.2_efficiency.md`](04_inference_serving/4.2_efficiency.md)
- KV-Cache, Prefill vs Decode Phase
- PagedAttention, Continuous Batching
- Speculative Decoding, Early Exit Decoding
- Draft Models

#### 4.3 Quantization
[`04_inference_serving/4.3_quantization.md`](04_inference_serving/4.3_quantization.md)
- Post-Training Quantization (PTQ)
- Weight-Only Quantization, Activation Quantization
- KV Quantization
- AWQ, GPTQ, FP8 Inference

#### 4.4 Serving Frameworks
[`04_inference_serving/4.4_serving_frameworks.md`](04_inference_serving/4.4_serving_frameworks.md)
- vLLM, TGI, TensorRT-LLM, SGLang
- Triton (OpenAI), CUDA Kernel Fusion

#### 4.5 Metrics
[`04_inference_serving/4.5_metrics.md`](04_inference_serving/4.5_metrics.md)
- Throughput (tokens/sec)
- TTFT (Time To First Token)
- Tail Latency
- Cost per Token

---

### Section 5: Long Context & Memory
[`05_long_context_memory/long_context_memory.md`](05_long_context_memory/long_context_memory.md)
- Context Window Scaling
- Chunking Strategies
- Retrieval-Augmented Generation (RAG)
- Embeddings, Vector DB
- Reranking, Query Rewriting
- Sliding Window Inference
- Memory Compression, Recurrent Memory Transformers

---

### Section 6: Reasoning & Test-Time Compute

#### 6.1 Reasoning Techniques
[`06_reasoning_test_time_compute/6.1_reasoning_techniques.md`](06_reasoning_test_time_compute/6.1_reasoning_techniques.md)
- Chain-of-Thought (CoT), Self-Consistency
- Tree-of-Thought, Reflection
- Program-of-Thought

#### 6.2 Test-Time Compute (Inference-Time Scaling)
[`06_reasoning_test_time_compute/6.2_test_time_compute.md`](06_reasoning_test_time_compute/6.2_test_time_compute.md)
- Process Reward Model (PRM) vs Outcome Reward Model (ORM)
- RLVR (RL with Verifiable Rewards)
- Long Chain-of-Thought Training
- Best-of-N Sampling
- Inference-Time Search (o1 / DeepSeek-R1 style)
- Test-Time Compute Scaling Laws

#### 6.3 Data Design
[`06_reasoning_test_time_compute/6.3_data_design.md`](06_reasoning_test_time_compute/6.3_data_design.md)
- Synthetic Data Generation
- Verification Models
- LLM-as-a-Judge
- Bias & Hallucination Analysis

---

### Section 7: LLM Model Families & Survey

#### 7.1 주요 모델 변천사 (2026년 2월 기준)
[`07_llm_model_families/7.1_llm_survey.md`](07_llm_model_families/7.1_llm_survey.md)
- GPT Family (OpenAI): GPT-2 → GPT-3 → InstructGPT → GPT-4 → GPT-4o → o1 → o3
- LLaMA Family (Meta): LLaMA 1/2/3/3.1/3.2/3.3
- Mistral / Mixtral Family
- Google Family: PaLM, Gemma 1/2/3, Gemini 1/1.5/2
- Alibaba Qwen Family: Qwen 1/1.5/2/2.5, QwQ, Qwen2.5-Max
- DeepSeek Family: V2, V3, R1
- Other: Phi, Command R, Claude 시리즈, Falcon

#### 7.2 Contribution Analysis
[`07_llm_model_families/7.2_contribution_analysis.md`](07_llm_model_families/7.2_contribution_analysis.md)
- Architectural Innovation Timeline
- Training Strategy Evolution
- What Each Generation Solved
- Efficiency Progression
- Key Lessons for Research

---

## Part II: Vision-Language

### Section 8: Vision-Language Models (VLM)

#### 8.1 Vision Encoder
[`08_vlm/8.1_vision_encoder.md`](08_vlm/8.1_vision_encoder.md)
- CNN vs ViT
- CLIP, SigLIP
- Contrastive Pretraining, Patch Embedding

#### 8.2 VLM Architecture
[`08_vlm/8.2_vlm_architecture.md`](08_vlm/8.2_vlm_architecture.md)
- Vision Encoder + LLM
- Linear Projector, MLP Projector
- Q-Former, Perceiver Resampler
- Early Fusion vs Late Fusion
- Unified Multimodal Transformer

#### 8.3 Resolution Handling
[`08_vlm/8.3_resolution_handling.md`](08_vlm/8.3_resolution_handling.md)
- Fixed / Dynamic Resolution
- AnyRes, Multi-Scale Patching
- Token Pruning, Aspect Ratio Preservation

#### 8.4 Multimodal Training Issues
[`08_vlm/8.4_multimodal_training_issues.md`](08_vlm/8.4_multimodal_training_issues.md)
- Vision-Text Alignment
- Hallucination, Grounding
- Multi-Image Input, Video Tokens
- Catastrophic Forgetting (Vision Encoder Drift)

#### 8.5 VLM Model Families (2026년 2월 기준)
[`08_vlm/8.5_vlm_model_families.md`](08_vlm/8.5_vlm_model_families.md)
- Foundation: CLIP, ALIGN, SigLIP
- Early VLMs: Flamingo, BLIP, BLIP-2
- LLaVA Lineage: LLaVA → 1.5 → NeXT → OneVision
- Strong Open-Source: InternVL 1/2/2.5/3, Qwen-VL/2-VL/2.5-VL
- Closed Frontier: GPT-4V/4o, Gemini 1/1.5/2, Claude 3/3.5/3.7
- 2025 Latest: LLaMA-3.2 Vision, Gemma 3 Vision, Pixtral

---

## Part III: Embodied AI

### Section 9: VLA (Vision-Language-Action)

#### 9.1 Foundations
[`09_vla/9.1_foundations.md`](09_vla/9.1_foundations.md)
- Behavior Cloning
- Offline RL / Imitation Learning
- Autoregressive Action Modeling
- Low-Level Control vs High-Level Planning
- Simulation-to-Real Transfer
- World Models

#### 9.2 Action Representation
[`09_vla/9.2_action_representation.md`](09_vla/9.2_action_representation.md)
- Action Tokenization (Continuous vs Discrete)
- Diffusion Policy
- Flow Matching for Action Prediction
- Chunked Action Prediction

#### 9.3 Representative Models
[`09_vla/9.3_representative_models.md`](09_vla/9.3_representative_models.md)
- RT-1, RT-2 (Google)
- Octo (UC Berkeley)
- OpenVLA (Stanford)
- π0 / π0-FAST (Physical Intelligence)
- Multi-Task Robotic Policies

#### 9.4 Embodied AI Topics
[`09_vla/9.4_embodied_ai_topics.md`](09_vla/9.4_embodied_ai_topics.md)
- Tool Use in Embodied Settings
- Hierarchical Planning
- Language-Conditioned Manipulation

---

## Part IV: Evaluation, Safety & Trends

### Section 10: Benchmarks & Evaluation

#### 10.1 LLM Benchmarks
[`10_benchmarks_evaluation/10.1_llm_benchmarks.md`](10_benchmarks_evaluation/10.1_llm_benchmarks.md)
- MMLU, HellaSwag, WinoGrande
- HumanEval, MBPP
- GSM8K, MATH
- GPQA, LiveCodeBench

#### 10.2 VLM Benchmarks
[`10_benchmarks_evaluation/10.2_vlm_benchmarks.md`](10_benchmarks_evaluation/10.2_vlm_benchmarks.md)
- MMMU
- MMBench, MMStar
- TextVQA, DocVQA
- ChartQA, POPE

#### 10.3 Long-Context Benchmarks
[`10_benchmarks_evaluation/10.3_long_context_benchmarks.md`](10_benchmarks_evaluation/10.3_long_context_benchmarks.md)
- NIAH (Needle-In-A-Haystack)
- RULER
- LongBench, Infinite Bench

#### 10.4 Evaluation Issues
[`10_benchmarks_evaluation/10.4_evaluation_issues.md`](10_benchmarks_evaluation/10.4_evaluation_issues.md)
- Benchmark Contamination
- Evaluation Harness (lm-evaluation-harness)
- LLM-as-a-Judge의 한계
- Benchmark Saturation 문제

---

### Section 11: Safety
[`11_safety/safety.md`](11_safety/safety.md)
- Hallucination, Toxicity
- Jailbreak Attacks, Prompt Injection
- Red Teaming
- Bias Evaluation
- Calibration, Abstention Mechanisms
- Robustness Testing

---

### Section 12: Emerging Trends (2024+)
[`12_emerging_trends/emerging_trends.md`](12_emerging_trends/emerging_trends.md)
- Multimodal Scaling Laws
- Large Multimodal Context
- Video LLMs, Audio-Language Models
- Multimodal Reasoning Benchmarks
- Self-Improving Models
- Tool-Augmented LLMs, Agentic Systems
- Structured Output Training
- Modular LLM Systems
