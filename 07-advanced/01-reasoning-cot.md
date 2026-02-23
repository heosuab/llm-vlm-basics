# Reasoning & Chain-of-Thought

## Chain-of-Thought (CoT) [Wei et al., 2022]

```
핵심: 최종 답 전에 추론 과정을 명시적으로 생성

Without CoT:
  Q: 4명의 학생이 각각 2개씩 사탕을 갖고 있다. 총 몇 개?
  A: 6개  ← 틀림

With CoT:
  Q: 4명의 학생이 각각 2개씩 사탕을 갖고 있다. 총 몇 개?
  A: 각 학생이 2개씩, 학생이 4명이므로 2 × 4 = 8개  ← 맞음

왜 효과적인가?
  1. 중간 계산을 "scratch pad"로 활용
  2. 복잡한 문제를 작은 단계로 분해
  3. 오류를 중간 단계에서 발견/수정 가능
  4. 더 깊은 layer 처리 효과 (더 많은 토큰 = 더 많은 computation)
```

### Zero-shot CoT vs Few-shot CoT
```
Zero-shot CoT (Kojima et al., 2022):
  "Let's think step by step." 추가만으로 효과
  → 별도 예시 없이 CoT 유도

Few-shot CoT (Wei et al., 2022):
  CoT 예시를 few-shot prompt에 포함
  "Q: ... A: 먼저 X를 계산하면... 따라서 답은 Y이다."
  → 원하는 추론 형식 명시적 제시

Auto-CoT:
  문제 클러스터링 → 각 클러스터에서 Zero-shot CoT로 예시 자동 생성
  → Few-shot 예시 수동 작성 불필요
```

---

## 고급 CoT 기법

### Self-Consistency [Wang et al., 2022]
```
같은 질문에 여러 추론 경로 생성 → 다수결

과정:
  1. Temperature > 0으로 N개 (20~40개) 생성
  2. 각 추론 경로에서 최종 답 추출
  3. 가장 많이 등장하는 답 선택

효과:
  단일 CoT 대비 10~15% 향상
  오류 경로 중 하나가 맞을 가능성 활용

비용: N배 추론 비용 (병렬 처리 가능)
```

```python
from anthropic import Anthropic
from collections import Counter

def self_consistency(client, question: str, n_samples: int = 10) -> str:
    """Self-consistency: sample N paths, take majority vote"""
    answers = []

    for _ in range(n_samples):
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            temperature=0.8,  # Higher temp for diverse paths
            messages=[{
                "role": "user",
                "content": f"Solve step by step:\n{question}\n\nFinal answer (number only):"
            }]
        )
        # Extract final answer from response
        text = response.content[0].text
        # Parse last number in response
        import re
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        if numbers:
            answers.append(numbers[-1])

    if not answers:
        return "No answer found"

    # Majority vote
    counter = Counter(answers)
    return counter.most_common(1)[0][0]
```

### Tree of Thoughts (ToT) [Yao et al., 2023]
```
선형 추론 → 트리 구조 탐색

과정:
  1. 각 단계에서 여러 "thought" 생성
  2. LLM으로 각 thought 평가 (점수)
  3. BFS: 넓게 탐색 (여러 경로 동시)
  4. DFS: 깊게 탐색 (하나씩 시도, 실패 시 backtrack)
  5. 최고 점수 경로 선택

Game of 24 (수식으로 24 만들기):
  표준 CoT: 4%
  ToT (BFS):  74%

장점: 복잡한 계획 태스크에서 강력
단점: 매우 높은 API 비용 (N단계 × M thought × evaluation)
```

```python
from dataclasses import dataclass
from typing import Optional
import heapq

@dataclass
class ThoughtNode:
    thought: str
    score: float
    depth: int
    parent: Optional['ThoughtNode'] = None

    def __lt__(self, other):
        return self.score > other.score  # Max-heap (higher score first)

def tree_of_thoughts_bfs(
    client,
    problem: str,
    n_thoughts: int = 3,  # thoughts per node
    n_levels: int = 3,    # tree depth
    beam_width: int = 5   # BFS beam width
) -> str:
    """BFS-based Tree of Thoughts"""

    def generate_thoughts(context: str) -> list[tuple[str, float]]:
        """Generate candidate thoughts with scores"""
        prompt = f"""Problem: {problem}
Current reasoning: {context}

Generate {n_thoughts} different next reasoning steps.
For each, rate its promise (0-10).
Format: STEP: <reasoning> | SCORE: <0-10>"""

        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )

        thoughts = []
        for line in response.content[0].text.strip().split('\n'):
            if 'STEP:' in line and 'SCORE:' in line:
                parts = line.split('|')
                if len(parts) == 2:
                    thought = parts[0].replace('STEP:', '').strip()
                    score_str = parts[1].replace('SCORE:', '').strip()
                    try:
                        score = float(score_str)
                        thoughts.append((thought, score))
                    except ValueError:
                        pass
        return thoughts

    # Initialize with root
    beam = [ThoughtNode(thought="", score=10.0, depth=0)]

    for level in range(n_levels):
        candidates = []
        for node in beam:
            context = node.thought
            new_thoughts = generate_thoughts(context)
            for thought, score in new_thoughts:
                child = ThoughtNode(
                    thought=context + "\n" + thought if context else thought,
                    score=score,
                    depth=level + 1,
                    parent=node
                )
                heapq.heappush(candidates, child)

        # Keep top beam_width
        beam = []
        for _ in range(min(beam_width, len(candidates))):
            beam.append(heapq.heappop(candidates))

    # Return best path
    return beam[0].thought if beam else "No solution found"
```

### Graph of Thoughts (GoT) [Besta et al., 2023]
```
ToT 확장: 트리 → 그래프
  노드 간 임의 연결 (합치기, 분기)
  여러 추론 경로를 "merge" 가능

예: 여러 요약 → 하나로 통합 (Aggregation 노드)
```

### ReAct [Yao et al., 2023]
```
Reasoning + Acting 결합

반복:
  Thought: 현재 상황 분석, 다음 행동 계획
  Action: 도구 호출 (Search, Calculator, Code)
  Observation: 도구 결과 관찰

예:
  Q: 최근 한국 대통령은 누구인가?
  Thought: 웹 검색이 필요하다
  Action: Search["한국 대통령 2024"]
  Observation: "윤석열..."
  Thought: 검색 결과로 답할 수 있다
  Answer: ...

장점: 외부 정보로 환각 감소
코드: LangChain, LlamaIndex ReAct Agent
```

### Least-to-Most Prompting [Zhou et al., 2022]
```
복잡한 문제 → 쉬운 하위 문제 순서로 분해

단계:
  1. Decompose: 복잡한 문제를 하위 문제로 분해
  2. Solve sequentially: 쉬운 것부터 순서대로 해결
  3. 각 답을 다음 문제의 컨텍스트로 사용

예:
  Q: "어제 한 시간에 5개를 만들었고, 오늘은 어제의 2배 속도로 3시간..."
  1. Decompose: "어제 몇 개? → 오늘 속도는? → 오늘 몇 개?"
  2. Solve: 5개 → 10개/시 → 30개

효과: Compositional generalization 향상
```

### Program-of-Thoughts (PoT) / PAL
```
코드로 추론:
  수식적 문제 → Python 코드로 작성 → 실행

장점:
  계산 오류 없음 (코드가 실행됨)
  복잡한 수식 정확하게 처리

예:
  Q: 사과가 처음 50개, 30% 판매, 남은 것의 절반 기부...
  A: """
  apples = 50
  sold = apples * 0.3
  remaining = apples - sold
  donated = remaining / 2
  final = remaining - donated
  print(final)
  """
  → 실행: 17.5
```

### Step-Back Prompting [Zheng et al., 2023]
```
추상화 후 구체화:
  1. Step Back: 구체적 문제 → 더 일반적인 원칙/개념 묻기
  2. Reason: 일반 원칙을 구체 문제에 적용

예:
  Q: "에탄올이 물에 녹는 이유?"
  Step Back: "극성 분자 용해 원리는 무엇인가?"
  Apply: 에탄올의 극성 OH기 → 물의 극성 → 상호 용해

효과:
  MMLU에서 ~7% 향상
  물리, 화학 지식 문제에서 특히 효과
```

---

## Process Reward Model (PRM) vs Outcome Reward Model (ORM)

```
ORM (Outcome RM):
  최종 답만 평가 (맞음/틀림)
  단순, 구현 쉬움
  중간 추론 과정의 오류 탐지 불가

  한계:
    운좋게 맞은 답 (틀린 추론 → 우연히 맞음) 구분 불가
    RL에서 sparse reward (마지막에만 신호)

PRM (Process RM):
  각 추론 단계 평가
  잘못된 단계 조기 탐지 가능

  PRM800K (OpenAI):
    수학 문제 추론 단계별 레이블 데이터셋
    인간이 각 단계에 좋음/나쁨 레이블

  Math-Shepherd (Peng et al., 2024):
    각 단계에서 Monte Carlo rollout으로 자동 레이블
    끝까지 굴려서 맞으면 이 단계도 좋음, 틀리면 나쁨

  활용:
    Best-of-N: PRM 점수 기준 최고 응답 선택
    MCTS: 각 단계 평가로 탐색 효율화
    RL: Dense reward (매 단계)

비교:
  실용성: ORM이 훨씬 구현 쉬움
  성능: PRM이 복잡한 추론에서 더 좋음
  데이터: PRM 학습 데이터 수집 어려움 (단계별 레이블)
```

```python
class ProcessRewardModel:
    """PRM: 각 추론 단계 평가"""

    def __init__(self, model, tokenizer):
        self.model = model  # Fine-tuned for step scoring
        self.tokenizer = tokenizer

    def score_steps(self, problem: str, steps: list[str]) -> list[float]:
        """각 추론 단계에 점수 부여"""
        scores = []
        context = problem

        for step in steps:
            # 현재 단계까지의 컨텍스트로 점수 예측
            inp = f"Problem: {context}\nStep: {step}\nIs this step correct?"
            tokens = self.tokenizer(inp, return_tensors="pt")

            with torch.no_grad():
                logits = self.model(**tokens).logits

            # Binary: correct (1) vs incorrect (0)
            prob_correct = torch.softmax(logits[0, -1], dim=-1)[self.tokenizer.convert_tokens_to_ids("Yes")]
            scores.append(prob_correct.item())
            context += f"\n{step}"

        return scores

    def best_of_n_with_prm(self, solutions: list[list[str]], problem: str) -> list[str]:
        """PRM으로 N개 솔루션 중 최선 선택"""
        best_score = -1
        best_solution = solutions[0]

        for solution_steps in solutions:
            step_scores = self.score_steps(problem, solution_steps)
            # 최소 단계 점수 (약한 링크 찾기)
            min_score = min(step_scores)
            avg_score = sum(step_scores) / len(step_scores)

            # Combined: 최소 × 평균 (모든 단계가 좋아야 함)
            combined = min_score * avg_score
            if combined > best_score:
                best_score = combined
                best_solution = solution_steps

        return best_solution

def monte_carlo_step_labeling(
    model, problem: str, steps_so_far: list[str], n_rollouts: int = 8
) -> float:
    """Math-Shepherd: MC rollout로 현재 단계 레이블 자동 생성"""
    correct_count = 0

    for _ in range(n_rollouts):
        # 현재 단계에서 끝까지 완성
        completion = generate_solution_from_steps(model, problem, steps_so_far)
        answer = extract_answer(completion)
        ground_truth = get_ground_truth(problem)

        if check_answer(answer, ground_truth):
            correct_count += 1

    # 현재 단계의 추정 가치 = 이후 성공 확률
    return correct_count / n_rollouts
```

---

## Test-Time Compute Scaling

```
핵심 아이디어: "더 많은 추론 시간 = 더 좋은 결과"

방법들:
  1. Best-of-N Sampling:
     N개 생성 → RM/Verifier로 최고 선택
     단순, 효과적
     비용: N× 추론

  2. Beam Search:
     여러 경로 병렬 탐색 + 가지치기
     NLP 전통적 방법

  3. MCTS (Monte Carlo Tree Search):
     AlphaGo에서 영감
     각 단계에서 여러 선택지 탐색
     PRM으로 각 노드 평가
     가장 강력, 가장 비쌈

  4. Iterative Self-Refinement:
     1. 초기 답 생성
     2. 스스로 검토 (오류 찾기)
     3. 개선된 답 생성
     4. 반복

  5. Sequential Revision:
     이전 시도와 피드백 참조하여 개선

Scaling Law (Test-Time):
  compute ↑ → accuracy ↑ (하지만 diminishing returns)
  특히 검증 가능한 문제 (수학, 코딩)에서 강함
  open-ended 태스크에서는 효과 제한적
```

### MCTS for LLM Reasoning
```python
import math
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class MCTSNode:
    state: str          # 현재까지의 추론 텍스트
    parent: Optional['MCTSNode'] = None
    children: list['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    is_terminal: bool = False

    def ucb_score(self, c: float = 1.414) -> float:
        """UCB1: 탐색-활용 균형"""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

class LLM_MCTS:
    """Monte Carlo Tree Search for LLM Step-by-Step Reasoning"""

    def __init__(self, policy_model, value_model, problem: str):
        self.policy = policy_model   # Generates next reasoning steps
        self.value = value_model     # PRM: evaluates reasoning quality
        self.problem = problem
        self.root = MCTSNode(state=f"Problem: {problem}\n")

    def select(self, node: MCTSNode) -> MCTSNode:
        """Selection: UCB로 리프 노드까지 내려가기"""
        while node.children and not node.is_terminal:
            node = max(node.children, key=lambda n: n.ucb_score())
        return node

    def expand(self, node: MCTSNode, n_actions: int = 3) -> list[MCTSNode]:
        """Expansion: 현재 상태에서 가능한 다음 단계 생성"""
        if node.is_terminal:
            return []

        # Policy model로 다음 단계 N개 생성
        next_steps = self.policy.generate_steps(node.state, n=n_actions)

        for step in next_steps:
            child_state = node.state + f"\nStep: {step}"
            is_final = self._is_final_answer(step)
            child = MCTSNode(
                state=child_state,
                parent=node,
                is_terminal=is_final
            )
            node.children.append(child)

        return node.children

    def simulate(self, node: MCTSNode) -> float:
        """Simulation: Value model로 현재 상태 평가"""
        return self.value.score(node.state)

    def backpropagate(self, node: MCTSNode, value: float):
        """Backpropagation: 결과를 루트까지 전파"""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def search(self, n_iterations: int = 50) -> str:
        """Main MCTS loop"""
        for _ in range(n_iterations):
            # Select
            leaf = self.select(self.root)

            # Expand
            if leaf.visits > 0 and not leaf.is_terminal:
                children = self.expand(leaf)
                if children:
                    leaf = children[0]  # 첫 자식 선택

            # Simulate
            value = self.simulate(leaf)

            # Backpropagate
            self.backpropagate(leaf, value)

        # 가장 많이 방문된 경로 선택 (가장 안정적)
        best_child = max(self.root.children, key=lambda n: n.visits)
        return best_child.state

    def _is_final_answer(self, step: str) -> bool:
        return "Therefore" in step or "Answer:" in step or "= " in step.split()[-1:][0:1]
```

---

## Large Reasoning Models (o1, R1)

### OpenAI o1 (2024)

```
핵심: "생각하는 시간" 허용 → 긴 내부 추론 후 답

훈련 방법 (추정):
  RLHF로 긴 추론 과정 학습
  수학/코딩 검증 가능한 보상
  Process reward 또는 outcome reward

특징:
  내부 reasoning trace (private, 사용자에게 숨김)
  외부에 summary만 노출
  토큰 수가 많아 느리고 비쌈
  어려운 수학, 코딩, 과학에서 GPT-4 대비 큰 향상

o1 vs GPT-4 (AIME 2024):
  GPT-4o: 13%
  o1: 83%

o3 (2025): 더 강력한 버전
o1-mini/o3-mini: 효율적인 소형 버전
```

### DeepSeek-R1 (2025)

```
오픈소스 Reasoning 모델 (최초로 o1 수준)

R1-Zero (순수 RL 실험):
  SFT 없이 직접 GRPO RL
  보상:
    정확성: 수학/코딩 정답 여부 (0/1)
    형식: <think>...</think> 태그 사용
  결과:
    CoT reasoning 자발적 등장!
    "Aha moment": 복잡한 문제에서 자체 반성
    BUT 가독성 문제 (중국어/영어 혼용)

R1 (SFT → RL → 재증류):
  1. Cold Start SFT: 소량 고품질 CoT 데이터
  2. GRPO RL: 수학/코딩 정확성 보상
  3. Rejection Sampling + SFT: 고품질 응답 선별
  4. GRPO RL + DPO: Helpfulness + Safety

증류 모델:
  R1-Distill-Qwen-1.5B ~ 70B
  DeepSeek-R1-Distill-LLaMA-8B/70B
  → 소형 모델로 reasoning 전이

성능:
  MATH-500: 97.3% (GPT-4o: 74.6%)
  AIME 2024: 79.8% (o1: 79.2%)
```

### Claude (Extended Thinking)

```python
import anthropic

client = anthropic.Anthropic()

def solve_with_extended_thinking(problem: str, budget_tokens: int = 10000):
    """Claude Extended Thinking으로 복잡한 문제 해결"""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": budget_tokens  # 사용할 생각 토큰 예산
        },
        messages=[{
            "role": "user",
            "content": problem
        }]
    )

    # Thinking block과 Answer block 분리
    thinking_content = ""
    answer_content = ""

    for block in response.content:
        if block.type == "thinking":
            thinking_content = block.thinking  # 내부 추론 과정
        elif block.type == "text":
            answer_content = block.text        # 최종 답변

    return {
        "thinking": thinking_content,
        "answer": answer_content,
        "thinking_tokens": len(thinking_content.split())  # 근사치
    }

# 난이도별 적응적 사용
def adaptive_thinking(problem: str, problem_type: str):
    budgets = {
        "easy": 1000,
        "medium": 5000,
        "hard": 20000,
        "very_hard": 50000
    }
    budget = budgets.get(problem_type, 10000)
    return solve_with_extended_thinking(problem, budget)
```

### QwQ-32B / Qwen Thinking (2025)
```
Qwen2.5 기반 오픈소스 추론 모델

특징:
  32B 파라미터로 o1 수준 접근
  <think> 태그로 추론 분리
  DeepSeek-R1과 유사한 방식

Thinking 패턴:
  <think>
  Let me break this down step by step...
  Wait, I need to reconsider...
  Actually, let me try a different approach...
  </think>
  Final answer: ...

→ 자체 수정, 반성적 추론 패턴 포함
```

---

## Reasoning 능력 평가

| 벤치마크 | 내용 | 난이도 |
|---------|------|--------|
| GSM8K | 초등 수학 | 쉬움 |
| MATH | 경시 수학 (5단계) | 어려움 |
| AMC/AIME | 수학 올림피아드 | 매우 어려움 |
| HumanEval | 코딩 (164문제) | 중간 |
| MBPP | 코딩 (500문제) | 쉬움 |
| LiveCodeBench | 실시간 코딩 | 어려움 |
| ARC-AGI | 추상 추론 | 매우 어려움 |
| GPQA | 전문가 과학 | 매우 어려움 |
| MMLU-Pro | 지식+추론 | 어려움 |

---

## Scratchpad / Working Memory

```
LLM의 추론 메커니즘:
  각 forward pass = 단일 레이어 스택 통과
  → "생각할 공간" 없음 (context window만 있음)

CoT의 역할:
  Context window를 "working memory"로 활용
  중간 계산 결과를 텍스트로 외재화
  → 다음 토큰 예측 시 중간 결과 활용 가능

이론적 해석:
  CoT = 더 깊은 계산 그래프 (serial in time)
  각 추론 단계 = 레이어 1회 추가와 유사한 효과
  → "Depth-breadth tradeoff": 층 수 vs 시퀀스 길이

Implicit Reasoning vs Explicit CoT:
  최신 연구: 모델이 내부적으로 이미 다단계 추론
  CoT는 이를 외재화하여 검증/개선 가능하게 함
  but CoT가 실제 추론과 항상 일치하지는 않음 (사후 합리화 가능)
```

---

## Self-Critique and Refinement

### Reflexion [Shinn et al., 2023]
```python
def reflexion_loop(client, problem: str, max_iterations: int = 3) -> str:
    """Reflexion: 실패 시 자기 반성 후 재시도"""
    history = []

    for iteration in range(max_iterations):
        # 이전 시도 + 반성을 컨텍스트로 포함
        context = f"Problem: {problem}\n"
        if history:
            context += "Previous attempts and reflections:\n"
            for i, (attempt, reflection) in enumerate(history):
                context += f"Attempt {i+1}: {attempt}\nReflection: {reflection}\n\n"

        # 새 시도
        attempt_response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": context + "Now provide your best solution:"
            }]
        )
        attempt = attempt_response.content[0].text

        # 정확도 검증 (코딩/수학의 경우 자동 가능)
        if is_correct(attempt, problem):
            return attempt

        # 자기 반성 생성
        reflection_response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"Problem: {problem}\nYour attempt: {attempt}\n"
                           f"This was incorrect. What went wrong? How to improve?"
            }]
        )
        reflection = reflection_response.content[0].text
        history.append((attempt, reflection))

    # 최선의 시도 반환
    return history[-1][0] if history else "Failed to solve"
```

### Constitutional Self-Critique
```
Constitutional AI (Anthropic):
  1. Generate: 초기 응답 생성
  2. Critique: 헌법 원칙에 따라 자체 비판
     "이 응답이 안전한가? 도움이 되는가? 정직한가?"
  3. Revise: 비판을 반영하여 개선
  4. 반복 (선택적)

수학 추론에 적용:
  1. 답 생성
  2. 자체 검토: "각 단계가 수학적으로 정확한가?"
  3. 오류 발견 시 수정

→ 단순 재생성보다 더 타겟팅된 개선
```

---

## Reasoning Distillation

```
대형 추론 모델 → 소형 모델로 추론 능력 전이

방법:
  1. Chain-of-Thought Distillation:
     교사 모델(o1, R1)의 CoT traces를 학습 데이터로
     학생 모델이 같은 추론 패턴 모방

  2. Rejection Sampling:
     교사로 N개 생성 → 정답인 것만 선별
     선별된 데이터로 학생 SFT

  3. Step-level Distillation:
     각 단계를 별도로 학습 (PRM 활용)
     품질 높은 단계만 전수

DeepSeek-R1 → R1-Distill-Qwen-1.5B:
  R1의 reasoning traces로 1.5B 모델 SFT
  AIME에서 28.9% 달성 (기존 1.5B 대비 극적 향상)

주의:
  추론 패턴은 전이되지만 지식은 제한적
  소형 모델은 도메인 외 추론에서 실패 가능
```

---

## Further Questions

**Q. CoT가 왜 성능을 향상시키나?**
> 1) 중간 계산을 scratch pad로 활용. 2) 복잡한 문제를 작은 단계로 분해. 3) 오류를 중간 단계에서 발견/수정. 4) 더 많은 토큰 = 더 많은 computation (implicit depth 증가). 특히 수학/논리처럼 단계별 계산이 필요한 태스크에서 극적 효과.

**Q. o1과 일반 GPT-4의 핵심 차이는?**
> o1은 답 전에 긴 내부 추론 생성 (test-time compute 증가). 단순히 더 큰 모델이 아닌 사고 과정 자체를 학습. RL로 "유용한 추론 과정"을 학습 (verifiable reward). 특히 검증 가능한 문제 (수학, 코딩)에서 극적 향상. o1-mini처럼 더 작은 모델도 충분한 생각 시간으로 큰 모델 성능 가능.

**Q. PRM이 ORM보다 좋지만 왜 잘 안 쓰이나?**
> 데이터 수집 어려움: 각 추론 단계마다 인간이 레이블 달아야 함 (비용). 자동화 어려움: 단계 경계를 어떻게 정의할지 모호. Math-Shepherd처럼 MC rollout으로 자동화 시도 중. 실용적으로는 ORM + 충분한 Best-of-N으로 대부분 해결 가능.

**Q. MCTS를 LLM Reasoning에 적용하는 방법은?**
> 1) State: 현재 추론 단계까지의 텍스트. 2) Action: 다음 추론 단계 생성. 3) Reward: 최종 답 정확도 (ORM) 또는 단계 평가 (PRM). 4) Selection (UCB): 탐색/활용 균형. AlphaZero 스타일로 강화 → 매우 강력하지만 많은 rollout 필요. 실용적: 5~10 MCTS iteration으로도 효과.

**Q. Test-Time Compute Scaling의 한계는?**
> Diminishing returns: compute가 늘어날수록 이득 감소. Open-ended 태스크에서 "정답" 검증 불가 → Verifiable domain (수학, 코딩)에서만 강력. 비용: 1000× compute = 1000× 비용. 특정 hard problems에서는 아무리 compute 늘려도 해결 불가. Budget 대비 최적 전략 선택 필요.

**Q. Reasoning Distillation이 왜 잘 작동하는가?**
> 교사 모델의 추론 traces가 학생에게 "intermediate supervision" 제공. 학생은 최종 답뿐 아니라 추론 과정 자체를 모방. 직접 학습하기 어려운 추론 패턴을 데이터로 제공. 단, 교사보다 훨씬 작은 모델은 복잡한 추론 패턴을 내재화하기 어려움 (용량 한계).
