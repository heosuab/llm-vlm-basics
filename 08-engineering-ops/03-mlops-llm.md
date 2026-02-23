# MLOps for LLM

## LLM MLOps 개요

```
전통적 MLOps vs LLM MLOps:
  Traditional:
    - 소규모 모델 (MB ~ GB)
    - 버전 관리: 가중치 + 코드
    - 재학습: 배치, 정기적
    - 평가: 정확도, F1 등 단일 메트릭

  LLM:
    - 대규모 모델 (수십~수백 GB)
    - 버전 관리: 베이스모델 + LoRA + 프롬프트
    - 재학습: LoRA SFT, RLHF (비용↑)
    - 평가: 다차원 (품질, 안전, 속도, 비용)
    - 외부 API 의존성 (OpenAI, Anthropic)
    - 프롬프트 관리 필수
```

---

## LLM 프로덕션 배포 체크리스트

```
모델 준비:
  [ ] 양자화 (GPTQ, AWQ, GGUF) 적용
  [ ] Benchmark 성능 검증 (기준치 설정)
  [ ] Safety 평가 (Llama Guard, TrustAI)
  [ ] 레이턴시 측정 (TTFT, TPOT, E2E)
  [ ] 처리량 측정 (tokens/sec, req/sec)
  [ ] OOM 경계 테스트 (max_model_len, batch_size)

서빙 인프라:
  [ ] 서빙 프레임워크 선택 (vLLM, TensorRT-LLM)
  [ ] GPU 스펙 결정 (A100, H100, A10G)
  [ ] 스케일링 정책 (HPA, KEDA)
  [ ] 로드밸런서 설정 (Nginx, Kubernetes Ingress)
  [ ] 모니터링 설정 (Prometheus, Grafana)

운영:
  [ ] 로깅 (입력/출력/메타데이터, PII 처리)
  [ ] 비용 추적 (토큰 수, GPU 시간, API 비용)
  [ ] 지연 모니터링 (P50/P95/P99)
  [ ] 에러율 추적 (4xx, 5xx, OOM)
  [ ] 알림 설정 (PagerDuty, Slack)
  [ ] 롤백 계획 수립
```

---

## GPU 선택 가이드

```
A100 80GB SXM:
  - LLM 학습 표준 GPU
  - HBM2e: 2TB/s 메모리 대역폭
  - 312 TFLOPS BF16 (Tensor Core)
  - NVLink 600GB/s (SXM)
  - AWS p4d.24xlarge (8×A100)

H100 80GB SXM:
  - 현재 최고 표준
  - HBM3: 3.35TB/s 메모리 대역폭
  - 989 TFLOPS BF16, 1979 TFLOPS FP8
  - NVLink 900GB/s
  - Transformer Engine (FP8 native)
  - AWS p5.48xlarge (8×H100)
  - ~$8-12/hour (On-demand)

H200 141GB SXM:
  - H100의 업그레이드 버전
  - HBM3e: 4.8TB/s (H100 대비 +43%)
  - 141GB 메모리 (H100 대비 +76%)
  - 7B 모델 ~13개 동시 서빙

A10G 24GB:
  - 비용 효율 추론 GPU
  - 250 TFLOPS BF16
  - AWS g5.xlarge ~ g5.48xlarge
  - ~$1-4/hour

RTX 4090 24GB:
  - 소비자용 최고 (NVLink 없음)
  - 로컬 개발/추론
  - 저렴한 비용

GPU 메모리 계산:
  모델 파라미터 (B) × 2 bytes/param (BF16) = 최소 VRAM

  Llama-3-8B: 8B × 2 = 16GB
  Llama-3-70B: 70B × 2 = 140GB → 2×A100 or 2×H100
  Llama-3-405B: 405B × 2 = 810GB → 12×H100 (TP+PP)

  KV Cache 추가 필요:
    = 2 × num_layers × num_kv_heads × head_dim × seq_len × 2bytes
    Llama-3-8B, seq=8K: ~8GB
```

---

## 배포 전략

### Blue-Green 배포

```yaml
# Kubernetes Blue-Green 예시
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  selector:
    app: llm-inference
    version: green  # blue → green으로 트래픽 전환
---
# Blue (구버전, 트래픽 없음)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-blue
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llm-inference
      version: blue
  template:
    spec:
      containers:
      - name: llm
        image: llm-server:v1.0
---
# Green (신버전, 트래픽 있음)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-green
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llm-inference
      version: green
  template:
    spec:
      containers:
      - name: llm
        image: llm-server:v1.1
```

### Canary 배포

```python
# Nginx로 트래픽 분할 (10% → 신버전)
# nginx.conf
upstream llm_backend {
    server llm-v1:8000 weight=9;
    server llm-v2:8000 weight=1;  # 10% canary
}

# Istio VirtualService로 더 세밀한 제어
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
spec:
  http:
  - route:
    - destination:
        host: llm-service
        subset: v1
      weight: 90
    - destination:
        host: llm-service
        subset: v2
      weight: 10  # 10% canary
```

### Shadow Mode (섀도 배포)

```python
class ShadowRouter:
    """신버전에 동일 트래픽 보내되, 응답은 구버전 사용"""

    def __init__(self, primary_llm, shadow_llm, shadow_logger):
        self.primary = primary_llm
        self.shadow = shadow_llm
        self.logger = shadow_logger

    def generate(self, prompt: str) -> str:
        # 주 모델로 실제 응답 생성
        primary_response = self.primary.generate(prompt)

        # 섀도 모델은 백그라운드에서 실행 (비교용)
        import threading
        def shadow_inference():
            shadow_response = self.shadow.generate(prompt)
            self.logger.log({
                "prompt": prompt,
                "primary": primary_response,
                "shadow": shadow_response,
                "timestamp": time.time()
            })

        threading.Thread(target=shadow_inference, daemon=True).start()

        return primary_response  # 사용자에게는 주 모델 응답만
```

---

## Auto-scaling

```yaml
# KEDA (Kubernetes-based Event-Driven Autoscaling)
# GPU 서버를 큐 메시지 수에 따라 스케일링
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm-scaler
spec:
  scaleTargetRef:
    name: llm-deployment
  minReplicaCount: 1
  maxReplicaCount: 10
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: llm_queue_depth
      threshold: "10"  # 큐에 10개 이상 시 스케일 업
      query: sum(llm_pending_requests)

# HPA: CPU/메모리 기반 (GPU 메트릭은 custom metrics 필요)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  scaleTargetRef:
    name: llm-deployment
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Pods
    pods:
      metric:
        name: llm_gpu_utilization
      target:
        type: AverageValue
        averageValue: "80"  # 80% GPU 사용률 목표
```

---

## 비용 최적화

### 클라우드 비용 전략

```
Spot/Preemptible Instances:
  AWS Spot: 60-90% 비용 절감
  GCP Preemptible: 60-91% 절감
  Azure Spot: 최대 90% 절감
  단점: 중단 가능 → 체크포인트 필수, 추론 전용

Reserved Instances (RI):
  1년: ~40% 절감
  3년: ~60% 절감
  안정적 프로덕션 워크로드에 적합

대안 클라우드:
  Lambda Labs: H100 ~$2.5/hr (AWS ~$9/hr 대비 저렴)
  Vast.ai: 개인 GPU 마켓플레이스, $1-3/hr
  CoreWeave: GPU 클라우드 전문
  RunPod: $1.5-4/hr

API 서비스 (학습/추론 외주):
  Together AI: 오픈소스 모델 API
  Groq: 초고속 추론 (LPU 하드웨어)
  Fireworks AI: 빠른 추론, 파인튜닝 API
  OpenRouter: 다양한 모델 통합 게이트웨이
```

### 추론 비용 최적화

```python
class CostOptimizedLLM:
    """비용 최적화 LLM 서비스"""

    def __init__(self, primary_model, cheap_model, classifier):
        self.primary = primary_model    # 고성능 (비쌈)
        self.cheap = cheap_model        # 저렴 (빠름)
        self.classifier = classifier    # 난이도 분류기

    def generate(self, prompt: str) -> str:
        # 쿼리 난이도 분류
        difficulty = self.classifier.predict(prompt)

        if difficulty == "easy":
            return self.cheap.generate(prompt)  # 저렴한 모델
        else:
            return self.primary.generate(prompt)  # 고성능 모델

    # 토큰 절약 기법
    def generate_with_cache(self, prompt: str, cache: dict) -> str:
        # 동일 프롬프트 캐싱 (시맨틱 캐시)
        cache_key = semantic_hash(prompt)
        if cache_key in cache:
            return cache[cache_key]

        result = self.generate(prompt)
        cache[cache_key] = result
        return result
```

```python
# 토큰 수 계산 및 비용 추적
import tiktoken

class CostTracker:
    PRICING = {
        "claude-opus-4-6": (15.0/1_000_000, 75.0/1_000_000),    # input, output per token
        "claude-sonnet-4-6": (3.0/1_000_000, 15.0/1_000_000),
        "claude-haiku-4-5": (0.25/1_000_000, 1.25/1_000_000),
        "gpt-4o": (5.0/1_000_000, 15.0/1_000_000),
        "gpt-4o-mini": (0.15/1_000_000, 0.60/1_000_000),
    }

    def __init__(self, model: str):
        self.model = model
        self.total_input = 0
        self.total_output = 0

    def log_request(self, input_tokens: int, output_tokens: int):
        self.total_input += input_tokens
        self.total_output += output_tokens

    def get_cost(self) -> float:
        input_price, output_price = self.PRICING.get(self.model, (0, 0))
        return (self.total_input * input_price +
                self.total_output * output_price)

    def report(self):
        cost = self.get_cost()
        print(f"Model: {self.model}")
        print(f"Input tokens: {self.total_input:,}")
        print(f"Output tokens: {self.total_output:,}")
        print(f"Total cost: ${cost:.4f}")
```

---

## 모니터링 & 관찰성

### 핵심 메트릭

```
레이턴시 메트릭:
  TTFT (Time To First Token):
    - 스트리밍에서 가장 중요
    - 사용자가 처음 느끼는 응답성
    - 목표: P99 < 1초

  TPOT (Time Per Output Token):
    - 생성 속도 (토큰/초)
    - 목표: > 30 tokens/sec (자연스러운 읽기 속도)

  E2E Latency:
    - TTFT + TPOT × output_tokens
    - 배치 처리에서 중요

처리량 메트릭:
  Throughput: requests/sec, tokens/sec
  Queue Depth: 대기 중인 요청 수
  GPU Utilization: 목표 70-85%
  KV Cache Hit Rate: Prefix caching 효율

품질 메트릭:
  Error Rate: 5xx, OOM 오류율
  Rejection Rate: 안전 필터 거부율
  User Feedback: 좋아요/싫어요
  Output Length Distribution: 예상치 이탈 감지
```

### Prometheus + Grafana 설정

```python
# vLLM Prometheus 메트릭 수집
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# 메트릭 정의
request_count = Counter("llm_requests_total", "Total requests", ["model", "status"])
request_latency = Histogram(
    "llm_request_duration_seconds",
    "Request duration",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)
token_throughput = Gauge("llm_tokens_per_second", "Token generation rate")
gpu_memory_usage = Gauge("llm_gpu_memory_bytes", "GPU memory used", ["gpu_id"])

class MonitoredLLM:
    def generate(self, prompt: str, **kwargs) -> str:
        start_time = time.time()
        try:
            response = self._model.generate(prompt, **kwargs)
            request_count.labels(model=self.model_name, status="success").inc()
            return response
        except Exception as e:
            request_count.labels(model=self.model_name, status="error").inc()
            raise
        finally:
            duration = time.time() - start_time
            request_latency.labels(model=self.model_name).observe(duration)

start_http_server(8001)  # /metrics 엔드포인트
```

### LLM 특화 관찰성

```python
# LangSmith (LangChain 생태계)
import langsmith
from langsmith import traceable

@traceable(name="rag_pipeline")
def rag_pipeline(query: str) -> str:
    with langsmith.trace("retrieval") as t:
        docs = retriever.retrieve(query)
        t.add_metadata({"num_docs": len(docs)})

    with langsmith.trace("generation") as t:
        response = llm.generate(build_prompt(query, docs))
        t.add_metadata({"input_tokens": count_tokens(query),
                       "output_tokens": count_tokens(response)})
    return response

# Arize Phoenix (로컬 관찰성)
import phoenix as px
px.launch_app()  # 로컬 UI 실행

from opentelemetry import trace
from openinference.instrumentation.openai import OpenAIInstrumentor
OpenAIInstrumentor().instrument()
```

---

## Prompt Engineering & 관리

### DSPy (프로그래매틱 프롬프트 최적화)

```python
import dspy

class RAGPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        # Signature: 입력/출력 타입 정의
        self.retrieve = dspy.Retrieve(k=5)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# 최적화: 학습 예시로 프롬프트 자동 최적화
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

optimizer = BootstrapFewShotWithRandomSearch(
    metric=rag_quality_metric,
    max_bootstrapped_demos=8
)
optimized_pipeline = optimizer.compile(
    RAGPipeline(),
    trainset=train_examples
)
```

### Prompt 버전 관리

```python
# Git 기반 프롬프트 관리
# prompts/v1.0.0/system_prompt.txt
# prompts/v1.1.0/system_prompt.txt

class PromptRegistry:
    def __init__(self, prompts_dir: str):
        self.prompts_dir = Path(prompts_dir)
        self.cache: dict[str, str] = {}

    def get(self, name: str, version: str = "latest") -> str:
        """이름과 버전으로 프롬프트 로드"""
        key = f"{name}:{version}"
        if key not in self.cache:
            if version == "latest":
                # 가장 높은 버전 찾기
                files = sorted(self.prompts_dir.glob(f"{name}_*.txt"))
                path = files[-1] if files else None
            else:
                path = self.prompts_dir / f"{name}_{version}.txt"

            self.cache[key] = path.read_text() if path else ""
        return self.cache[key]

    def render(self, name: str, version: str = "latest", **kwargs) -> str:
        """템플릿 프롬프트 렌더링"""
        template = self.get(name, version)
        return template.format(**kwargs)

# A/B 테스팅
class PromptABTest:
    def __init__(self, prompt_a: str, prompt_b: str, traffic_split: float = 0.5):
        self.prompts = [prompt_a, prompt_b]
        self.split = traffic_split
        self.results = {"a": [], "b": []}

    def get_prompt(self, user_id: str) -> tuple[str, str]:
        """사용자 ID로 일관된 버전 할당"""
        bucket = hash(user_id) % 100
        variant = "a" if bucket < (self.split * 100) else "b"
        return self.prompts[0 if variant == "a" else 1], variant
```

---

## 안전성 & 필터링

### Llama Guard

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class SafetyFilter:
    """Llama Guard 기반 안전 필터"""

    def __init__(self, model_id: str = "meta-llama/Llama-Guard-3-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )

    def check(self, user_input: str, assistant_output: str = None) -> dict:
        messages = [{"role": "user", "content": user_input}]
        if assistant_output:
            messages.append({"role": "assistant", "content": assistant_output})

        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(input_ids, max_new_tokens=20)

        result = self.tokenizer.decode(
            output[0][input_ids.shape[-1]:], skip_special_tokens=True
        )

        is_safe = result.strip().startswith("safe")
        category = result.split("\n")[1] if not is_safe else None

        return {"safe": is_safe, "category": category, "raw": result}

    def guard_generate(self, prompt: str, llm) -> str:
        """입력 검사 → 생성 → 출력 검사"""
        # 입력 안전성 검사
        input_check = self.check(prompt)
        if not input_check["safe"]:
            return "I cannot help with that request."

        # 생성
        response = llm.generate(prompt)

        # 출력 안전성 검사
        output_check = self.check(prompt, response)
        if not output_check["safe"]:
            return "I cannot provide that response."

        return response
```

### 다층 방어

```python
class MultiLayerSafety:
    """다층 안전 필터링"""

    def __init__(self):
        self.keyword_filter = KeywordFilter()      # 빠른 규칙 기반
        self.regex_filter = RegexFilter()          # 패턴 기반
        self.llama_guard = SafetyFilter()          # LLM 기반
        self.custom_classifier = CustomClassifier() # 도메인 특화

    def check_input(self, text: str) -> tuple[bool, str]:
        # Layer 1: 키워드 (µs)
        if self.keyword_filter.contains_harmful(text):
            return False, "keyword_blocked"

        # Layer 2: 정규식 (µs)
        if self.regex_filter.matches_harmful_pattern(text):
            return False, "pattern_blocked"

        # Layer 3: Llama Guard (ms)
        result = self.llama_guard.check(text)
        if not result["safe"]:
            return False, f"guard_blocked:{result['category']}"

        return True, "safe"
```

---

## Structured Output

```python
# Pydantic + Instructor (타입 안전 출력)
import instructor
from pydantic import BaseModel, Field
from anthropic import Anthropic

class AnalysisResult(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(ge=0.0, le=1.0)
    key_topics: list[str] = Field(max_items=5)
    summary: str = Field(max_length=200)

client = instructor.from_anthropic(Anthropic())

result = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    response_model=AnalysisResult,
    messages=[{
        "role": "user",
        "content": f"Analyze this text: {text}"
    }]
)
print(result.sentiment, result.confidence)

# Outlines: 로컬 모델에서 JSON Schema 강제
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-v0.1")
schema = AnalysisResult.model_json_schema()
generator = outlines.generate.json(model, schema)
result = generator(f"Analyze: {text}")

# LMQL: 프로그래밍 방식 생성 제어
import lmql

@lmql.query
async def classify_and_explain(text: str):
    '''lmql
    "Classify sentiment of: {text}\n"
    "Sentiment: [SENTIMENT]" where SENTIMENT in ["positive", "negative", "neutral"]
    "Explanation: [EXPLANATION]" where len(EXPLANATION) < 100
    return SENTIMENT, EXPLANATION
    '''
```

---

## CI/CD for LLM

```yaml
# GitHub Actions LLM 배포 파이프라인
name: LLM Model CI/CD

on:
  push:
    branches: [main]
    paths:
      - 'models/**'
      - 'prompts/**'

jobs:
  evaluate:
    runs-on: gpu-runner  # 자체 GPU 러너
    steps:
      - uses: actions/checkout@v3

      - name: Run Eval Suite
        run: |
          python eval/run_benchmarks.py \
            --model models/latest \
            --benchmarks mmlu,hellaswag,safety \
            --output eval_results.json

      - name: Check Quality Gates
        run: |
          python eval/check_gates.py \
            --results eval_results.json \
            --min-mmlu 0.75 \
            --max-unsafe-rate 0.01

      - name: Safety Evaluation
        run: |
          python eval/safety_eval.py \
            --model models/latest \
            --adversarial-prompts datasets/red-team.json

  deploy-canary:
    needs: evaluate
    runs-on: ubuntu-latest
    steps:
      - name: Deploy 10% Canary
        run: |
          kubectl set image deployment/llm-canary \
            llm=llm-server:${{ github.sha }}
          kubectl rollout status deployment/llm-canary

      - name: Monitor Canary (30 min)
        run: python ops/monitor_canary.py --duration 1800

      - name: Promote to Production
        if: success()
        run: |
          kubectl set image deployment/llm-production \
            llm=llm-server:${{ github.sha }}
```

---

## 드리프트 감지 & 모델 모니터링

```python
from scipy.stats import ks_2samp
import numpy as np

class LLMDriftDetector:
    """LLM 출력 품질 드리프트 감지"""

    def __init__(self, reference_window: int = 1000):
        self.reference = []  # 기준 기간 데이터
        self.current = []    # 현재 기간 데이터
        self.window = reference_window

    def log_response(self, response: str, quality_score: float):
        self.current.append({
            "length": len(response.split()),
            "quality": quality_score,
            "timestamp": time.time()
        })

        if len(self.current) > self.window:
            self.reference = self.current[-self.window*2:-self.window]
            self.current = self.current[-self.window:]

    def check_drift(self) -> dict:
        if len(self.reference) < 100 or len(self.current) < 100:
            return {"drift_detected": False, "reason": "insufficient_data"}

        ref_quality = [r["quality"] for r in self.reference]
        cur_quality = [r["quality"] for r in self.current]

        # KS 테스트: 분포 변화 감지
        ks_stat, p_value = ks_2samp(ref_quality, cur_quality)

        drift_detected = p_value < 0.05
        return {
            "drift_detected": drift_detected,
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "ref_mean_quality": np.mean(ref_quality),
            "cur_mean_quality": np.mean(cur_quality)
        }
```

---

## 피드백 루프 & Human-in-the-Loop

```python
class FeedbackCollector:
    """사용자 피드백 수집 및 학습 데이터 생성"""

    def __init__(self, db_connection):
        self.db = db_connection

    def log_interaction(self,
                        session_id: str,
                        prompt: str,
                        response: str,
                        metadata: dict) -> str:
        """상호작용 로깅"""
        interaction_id = generate_uuid()
        self.db.insert("interactions", {
            "id": interaction_id,
            "session_id": session_id,
            "prompt": prompt,
            "response": response,
            "model_version": metadata.get("model_version"),
            "latency_ms": metadata.get("latency_ms"),
            "timestamp": time.time()
        })
        return interaction_id

    def log_feedback(self,
                     interaction_id: str,
                     thumbs_up: bool,
                     correction: str = None):
        """피드백 수집"""
        self.db.update("interactions", interaction_id, {
            "user_rating": 1 if thumbs_up else -1,
            "correction": correction,
            "feedback_timestamp": time.time()
        })

    def export_training_data(self, min_rating: int = 1) -> list[dict]:
        """고품질 피드백을 SFT 학습 데이터로 변환"""
        good_interactions = self.db.query(
            "SELECT prompt, response FROM interactions WHERE user_rating >= ?",
            (min_rating,)
        )

        # 수정 피드백 → DPO 학습 데이터
        corrections = self.db.query(
            "SELECT prompt, response, correction FROM interactions "
            "WHERE correction IS NOT NULL"
        )

        dpo_data = [{
            "prompt": c["prompt"],
            "chosen": c["correction"],    # 사용자가 선호하는 응답
            "rejected": c["response"]     # 모델의 원래 응답
        } for c in corrections]

        return dpo_data
```

---

## 비용 추적 & SLA 관리

```python
class LLMSLAMonitor:
    """SLA 관리 및 알림"""

    SLA = {
        "ttft_p99_ms": 1000,      # 1초
        "tpot_p99_ms": 50,         # 50ms/token
        "availability": 0.999,     # 99.9%
        "error_rate_max": 0.01     # 1%
    }

    def __init__(self, alert_webhook: str):
        self.metrics = []
        self.webhook = alert_webhook

    def check_sla(self) -> dict:
        recent = [m for m in self.metrics
                  if m["timestamp"] > time.time() - 3600]  # 1시간

        if not recent:
            return {}

        ttft_values = [m["ttft_ms"] for m in recent]
        ttft_p99 = np.percentile(ttft_values, 99)

        error_count = sum(1 for m in recent if m["status"] != "ok")
        error_rate = error_count / len(recent)

        violations = []
        if ttft_p99 > self.SLA["ttft_p99_ms"]:
            violations.append(f"TTFT P99 {ttft_p99:.0f}ms > {self.SLA['ttft_p99_ms']}ms")

        if error_rate > self.SLA["error_rate_max"]:
            violations.append(f"Error rate {error_rate:.2%} > {self.SLA['error_rate_max']:.2%}")

        if violations:
            self.send_alert(violations)

        return {"violations": violations, "ttft_p99": ttft_p99, "error_rate": error_rate}
```

---

## Model Registry & 버전 관리

```python
# MLflow 모델 레지스트리
import mlflow

# 모델 등록
with mlflow.start_run():
    mlflow.log_params({
        "base_model": "meta-llama/Llama-3-8B",
        "training_type": "LoRA",
        "dataset": "custom_sft_v2.jsonl",
        "lora_rank": 16
    })
    mlflow.log_metrics({
        "eval_loss": 1.23,
        "mmlu_score": 0.78,
        "safety_score": 0.99
    })

    # 모델 아티팩트 저장
    mlflow.pyfunc.log_model(
        artifact_path="llm_model",
        python_model=LLMWrapper(),
        artifacts={"lora_weights": "./lora_weights.safetensors"}
    )

    # 레지스트리 등록
    mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/llm_model",
        "customer_support_llm"
    )

# 스테이지 전환
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name="customer_support_llm",
    version=3,
    stage="Production"  # Staging → Production
)
```

---

## Further Questions

**Q. LLM 서비스의 SLA를 정의할 때 고려할 지표는?**
> TTFT P99 (스트리밍의 체감 응답성), TPOT P99 (생성 속도), E2E P99 (배치 처리), 가용성 (99.9% = 월 44분 다운타임), 에러율 (< 1%), 토큰/비용 효율. 스트리밍이면 TTFT 최우선, 배치 처리면 E2E Latency와 처리량. 비즈니스 요구에 따라 우선순위 결정.

**Q. LLM 서빙 비용을 줄이는 방법은?**
> 1) 양자화 (INT4/AWQ로 메모리 75% 절감). 2) 작은 모델로 증류 (Teacher-Student). 3) Speculative Decoding (같은 품질, 2-3× 빠름). 4) Prefix caching (공통 시스템 프롬프트 캐싱). 5) 적절한 배치 크기 (GPU 활용률 최대화). 6) Spot instances (60-90% 절감). 7) 시맨틱 캐싱 (동일 쿼리 재사용). 8) 쿼리 라우팅 (쉬운 질문 → 작은 모델).

**Q. LLM 배포 시 Canary vs Blue-Green 선택 기준은?**
> Blue-Green: 빠른 전환, 롤백 즉각적, 리소스 2배 필요, 완전히 새 버전으로 전환 시. Canary: 점진적 트래픽 이동, 실제 사용자로 검증, 리소스 효율적, 품질 위험이 클 때. LLM은 예측 불가한 출력 특성 때문에 Canary 선호. Shadow Mode로 먼저 검증 후 Canary 권장.

**Q. LLM 출력 품질 드리프트 어떻게 감지하나?**
> 1) 사용자 피드백 (좋아요/싫어요) 추적. 2) LLM-as-Judge로 주기적 품질 평가. 3) 출력 길이 분포 변화 (KS 테스트). 4) Perplexity/엔트로피 변화. 5) 특정 벤치마크 주기적 재평가. 이상 감지 시 자동 알림 → 원인 분석 (데이터 드리프트, 모델 가중치 이슈, 인프라 변경).
