# LLM / VLM Basics

> LLM/VLM Research Engineer 커리어를 위한 basic 개념 정리

---

## 목차

| 섹션 | 내용 |
|------|------|
| [01. 기초 (Fundamentals)](./01-fundamentals/) | Transformer, Attention, Tokenization, Positional Encoding, Training Fundamentals |
| [02. 학습 (Training)](./02-training/) | Pretraining, SFT, Optimization, Data Pipeline, 분산 학습 |
| [03. 아키텍처 (Architectures)](./03-architectures/) | GPT/LLaMA, Efficient Architectures (Mamba/RWKV), MoE |
| [04. 추론 엔지니어링 (Inference Engineering)](./04-inference-engineering/) | KV Cache, Quantization, Batching, Serving Frameworks, GPU 아키텍처 |
| [05. 정렬·안전 (Alignment & Safety)](./05-alignment/) | RLHF/PPO, DPO 변형 (SimPO/ORPO/KTO), GRPO |
| [06. VLM (Vision-Language Models)](./06-vlm/) | Vision Encoder, Multimodal Fusion, VLM 학습·평가 |
| [07. 고급 주제 (Advanced Topics)](./07-advanced/) | Reasoning/CoT, Agents & Tools, RAG, Long Context |
| [08. ML Ops & 엔지니어링](./08-engineering-ops/) | PEFT, Evaluation, MLOps·배포 |
| [09. 인터뷰 준비 (Interview Prep)](./09-interview-prep/) | 연구자/엔지니어 질문, 필독 논문 리스트 |
| [10. 심화 주제 (Deep Dive)](./10-advanced-topics/) | 기계적 해석가능성, 안전성·Red Teaming, 모델 병합, Continual Learning, 심화 학습 기법 |
| [11. 연구 & 실무 (Research & Practice)](./11-research-and-practice/) | 미해결 연구 문제, 실무 의사결정 가이드 |

---

## 디렉토리 구조

```
llm-vlm-career-prep/
├── 01-fundamentals/
│   ├── 01-transformer-architecture.md
│   ├── 02-attention-mechanisms.md
│   ├── 03-tokenization.md
│   ├── 04-positional-encoding.md
│   └── 05-training-fundamentals.md
├── 02-training/
│   ├── 01-pretraining.md
│   ├── 02-finetuning-sft.md
│   ├── 03-optimization.md
│   ├── 04-data-pipeline.md
│   └── 05-distributed-training.md
├── 03-architectures/
│   ├── 01-decoder-only.md
│   ├── 02-efficient-architectures.md
│   └── 03-moe-architecture.md
├── 04-inference-engineering/
│   ├── 01-kv-cache.md
│   ├── 02-quantization.md
│   ├── 03-batching-scheduling.md
│   ├── 04-serving-frameworks.md
│   └── 05-hardware-gpu.md
├── 05-alignment/
│   ├── 01-rlhf-ppo.md
│   ├── 02-dpo-variants.md
│   └── 03-grpo-reasoning.md
├── 06-vlm/
│   ├── 01-vision-encoders.md
│   ├── 02-multimodal-fusion.md
│   └── 03-vlm-training-eval.md
├── 07-advanced/
│   ├── 01-reasoning-cot.md
│   ├── 02-agents-tools.md
│   ├── 03-rag.md
│   └── 04-long-context.md
├── 08-engineering-ops/
│   ├── 01-peft-methods.md
│   ├── 02-evaluation.md
│   └── 03-mlops-llm.md
├── 09-interview-prep/
│   ├── 01-researcher-questions.md
│   ├── 02-engineer-questions.md
│   └── 03-must-read-papers.md
├── 10-advanced-topics/
│   ├── 01-mechanistic-interpretability.md
│   ├── 02-safety-red-teaming.md
│   ├── 03-model-merging.md
│   ├── 04-continual-learning.md
│   └── 05-training-techniques-advanced.md
└── 11-research-and-practice/
    ├── 01-open-problems.md
    └── 02-practical-considerations.md
```

---

## 학습 로드맵

```
[기초 단계]
  Transformer → Attention → Tokenization → Positional Encoding

[모델 이해]
  GPT/BERT 계열 → LLaMA/Mistral → MoE → SSM (Mamba)

[학습 이해]
  Pretraining → SFT → RLHF/DPO → GRPO

[VLM 이해]
  Vision Encoder (ViT/CLIP) → Projection → Instruction Tuning

[추론 엔지니어링]
  KV Cache → Quantization → Batching → Serving (vLLM, TensorRT-LLM)

[고급/최신]
  Reasoning (o1/R1) → Agents → RAG → Long Context
```

---

## 키워드 인덱스 (빠른 참조)

| 키워드 | 위치 |
|--------|------|
| Transformer, RMSNorm, SwiGLU | 01-fundamentals/01 |
| Flash Attention, GQA, MQA | 01-fundamentals/02 |
| RoPE, ALiBi, YaRN, NTK Scaling | 01-fundamentals/04 |
| BPE, SentencePiece, Tiktoken | 01-fundamentals/03 |
| Backprop, Adam, Initialization | 01-fundamentals/05 |
| Pretraining, Chinchilla, Data Mix | 02-training/01 |
| SFT, Instruction Tuning, Chat Template | 02-training/02 |
| AdamW, LR Schedule, Gradient Clipping | 02-training/03 |
| Data Curation, Dedup, Filtering | 02-training/04 |
| ZeRO, Tensor/Pipeline Parallelism, FSDP | 02-training/05 |
| GPT, LLaMA, Mistral, Decoder-only | 03-architectures/01 |
| Mamba, RWKV, RetNet, SSM | 03-architectures/02 |
| MoE, Mixtral, DeepSeek-MoE, Expert Routing | 03-architectures/03 |
| KV Cache, PagedAttention, Prefix Caching | 04-inference-engineering/01 |
| AWQ, GPTQ, GGUF, FP8, BitNet | 04-inference-engineering/02 |
| Continuous Batching, Chunked Prefill | 04-inference-engineering/03 |
| vLLM, SGLang, TensorRT-LLM, llama.cpp | 04-inference-engineering/04 |
| H100, NVLink, HBM, MFU, Roofline | 04-inference-engineering/05 |
| PPO, RLHF, Reward Model | 05-alignment/01 |
| DPO, SimPO, ORPO, KTO, IPO, Step-DPO | 05-alignment/02 |
| GRPO, RLOO, Process Reward Model | 05-alignment/03 |
| CLIP, SigLIP, ViT, Vision Encoder | 06-vlm/01 |
| LLaVA, InternVL, Qwen2-VL, Flamingo | 06-vlm/02 |
| VLM Training, Hallucination, VQA Eval | 06-vlm/03 |
| Chain-of-Thought, MCTS, PRM, o1/R1 | 07-advanced/01 |
| ReAct, Function Calling, Tool Use | 07-advanced/02 |
| RAG, HyDE, ColBERT, RAGAS | 07-advanced/03 |
| Long Context, Sliding Window, ROPE Ext | 07-advanced/04 |
| LoRA, QLoRA, DoRA, Adapter | 08-engineering-ops/01 |
| MMLU, MT-Bench, LM-Eval, Contamination | 08-engineering-ops/02 |
| Blue-Green, Canary, Shadow, MLOps CI/CD | 08-engineering-ops/03 |
| Researcher Interview Q&A | 09-interview-prep/01 |
| Engineer Interview Q&A | 09-interview-prep/02 |
| Must-Read Papers (Attention~GRPO) | 09-interview-prep/03 |
| Mechanistic Interpretability, Probing | 10-advanced-topics/01 |
| Red Teaming, Jailbreak, Constitutional AI | 10-advanced-topics/02 |
| Task Vector, TIES, DARE, SLERP, mergekit | 10-advanced-topics/03 |
| EWC, Replay, Knowledge Editing, ROME | 10-advanced-topics/04 |
| Curriculum Learning, MuP, Warmup | 10-advanced-topics/05 |
| Data Wall, Model Collapse, Open Problems | 11-research-and-practice/01 |
| Open-source vs API, Eval Design | 11-research-and-practice/02 |
