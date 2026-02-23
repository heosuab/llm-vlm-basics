# LLM Agents & Tool Use

## LLM Agent 기본 구조

```
                 ┌─────────────────┐
                 │   LLM (Brain)   │
                 └────────┬────────┘
                          │ Plan & Reason
               ┌──────────┼──────────┐
               ↓          ↓          ↓
          Memory      Action      Tools
          (단기/장기)  생성         (API, Search, Code)
               ↑          ↓
               └──────────┘
                  Observation

Agent Loop:
  1. Observe: 현재 상태/환경 관찰
  2. Think: 다음 행동 계획 (LLM reasoning)
  3. Act: 도구 호출 또는 최종 답변
  4. Update: 결과 반영, 다음 iteration
```

---

## Tool Use (Function Calling)

### OpenAI Function Calling 형식

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name, e.g. 'Seoul, Korea'"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"],
              "description": "Temperature unit"
            }
          },
          "required": ["location"]
        }
      }
    }
  ]
}
```

### Function Calling 전체 구현

```python
import json
import anthropic
from typing import Any

# 도구 정의
tools = [
    {
        "name": "search_web",
        "description": "Search the web for current information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"]
        }
    },
    {
        "name": "execute_python",
        "description": "Execute Python code and return the result",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"]
        }
    }
]

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """도구 실제 실행"""
    if tool_name == "search_web":
        return web_search(tool_input["query"])
    elif tool_name == "execute_python":
        return run_python_sandbox(tool_input["code"])
    return "Tool not found"

def run_agent(user_message: str, max_iterations: int = 10):
    """Agent loop 실행"""
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_message}]

    for iteration in range(max_iterations):
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )

        # Assistant 응답 추가
        messages.append({"role": "assistant", "content": response.content})

        # 완료 여부 체크
        if response.stop_reason == "end_turn":
            # 최종 텍스트 응답 반환
            for block in response.content:
                if hasattr(block, 'text'):
                    return block.text

        # 도구 호출 처리
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"Calling tool: {block.name}({block.input})")
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })

            # 도구 결과를 대화에 추가
            messages.append({"role": "user", "content": tool_results})

    return "Max iterations reached"
```

---

## ReAct (Reasoning + Acting) [Yao et al., 2023]

```python
# ReAct 프롬프팅 예시
react_system_prompt = """You are a helpful assistant that can use tools.
Follow this format strictly:

Thought: [Analyze the current situation and decide what to do]
Action: [tool_name]
Action Input: [input to the tool]
Observation: [result from the tool]
... (repeat as needed)
Thought: [Final analysis]
Final Answer: [Your answer to the user]

Available tools:
- search: Search the web for information
- calculator: Perform calculations
- code: Execute Python code
"""

# ReAct 파싱
def parse_react_response(response: str) -> dict:
    """ReAct 형식 응답 파싱"""
    import re
    result = {
        "thought": None,
        "action": None,
        "action_input": None,
        "final_answer": None
    }

    thought_match = re.search(r"Thought: (.+?)(?=Action:|Final Answer:|$)", response, re.DOTALL)
    action_match = re.search(r"Action: (\w+)", response)
    input_match = re.search(r"Action Input: (.+?)(?=Observation:|$)", response, re.DOTALL)
    final_match = re.search(r"Final Answer: (.+?)$", response, re.DOTALL)

    if thought_match: result["thought"] = thought_match.group(1).strip()
    if action_match: result["action"] = action_match.group(1).strip()
    if input_match: result["action_input"] = input_match.group(1).strip()
    if final_match: result["final_answer"] = final_match.group(1).strip()

    return result
```

---

## Planning 패턴

### Task Decomposition (Least-to-Most)

```python
def hierarchical_planning(goal: str, llm) -> list[str]:
    """
    복잡한 목표를 하위 태스크로 분해
    """
    decomposition_prompt = f"""Break down this goal into concrete subtasks:

Goal: {goal}

Output format:
1. [Subtask 1]
2. [Subtask 2]
...

Each subtask should be specific and actionable."""

    plan = llm.generate(decomposition_prompt)
    subtasks = parse_numbered_list(plan)

    # 의존성 분석
    dependency_prompt = f"""Analyze dependencies between these tasks:
{chr(10).join(f'{i+1}. {t}' for i, t in enumerate(subtasks))}

Which tasks must be completed before others?"""

    dependencies = llm.generate(dependency_prompt)
    return subtasks, dependencies
```

### Plan-and-Execute

```python
class PlanAndExecuteAgent:
    """
    계획 수립 → 실행 → 계획 업데이트 에이전트
    """
    def __init__(self, planner_llm, executor_llm, tools):
        self.planner = planner_llm
        self.executor = executor_llm
        self.tools = tools

    def run(self, goal: str) -> str:
        # 1. 전체 계획 수립
        plan = self.planner.create_plan(goal)
        results = {}

        for i, step in enumerate(plan.steps):
            # 2. 각 단계 실행
            context = {
                "current_step": step,
                "completed_steps": results,
                "remaining_steps": plan.steps[i+1:]
            }
            result = self.executor.execute(context, self.tools)
            results[step] = result

            # 3. 실행 결과 기반 계획 업데이트
            if result.needs_replanning:
                plan = self.planner.replan(goal, results)

        return self.planner.synthesize_results(results)
```

### Self-Reflection (Reflexion) [Shinn et al., 2023]

```
실패 후 자기 반성 → 개선된 시도:

알고리즘:
  1. 태스크 시도
  2. 실패 시 성찰 생성:
     "왜 실패했는가? 다음엔 어떻게 할 것인가?"
  3. 성찰을 메모리에 저장
  4. 이전 시도 + 성찰을 컨텍스트로 재시도

예:
  시도 1: HotpotQA 문제 실패
  성찰: "Wikipedia에서 잘못된 엔티티를 검색했음.
         정확한 이름으로 다시 검색해야 함."
  시도 2: 수정된 검색어로 성공

효과: HotpotQA 정확도 69% → 91%
```

---

## 주요 Agent 프레임워크

### LangGraph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_results: list

# 노드 정의
def llm_node(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    results = execute_tools(last_message.tool_calls)
    return {"tool_results": results, "messages": [format_results(results)]}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 그래프 구성
workflow = StateGraph(AgentState)
workflow.add_node("llm", llm_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("llm")
workflow.add_conditional_edges("llm", should_continue)
workflow.add_edge("tools", "llm")

agent = workflow.compile()
result = agent.invoke({"messages": [("user", "What's the weather in Seoul?")]})
```

### AutoGen (Microsoft)

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat

# 여러 에이전트 협업
assistant = AssistantAgent(
    name="coder",
    system_message="You are an expert Python coder.",
    llm_config={"model": "gpt-4", "temperature": 0}
)

reviewer = AssistantAgent(
    name="reviewer",
    system_message="You review code for bugs and improvements.",
    llm_config={"model": "gpt-4"}
)

user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",    # 자동 실행
    code_execution_config={"work_dir": "workspace"},
    max_consecutive_auto_reply=10
)

# 그룹 채팅
group_chat = GroupChat(
    agents=[user_proxy, assistant, reviewer],
    messages=[],
    max_round=20
)
```

---

## Code Interpreter / Execution

```python
import subprocess
import tempfile
import os
from typing import tuple

def execute_python_safe(code: str, timeout: int = 30) -> tuple[str, str]:
    """
    안전한 Python 코드 실행 (sandbox)
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        result = subprocess.run(
            ["python", temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            # 리소스 제한 (Linux)
            # preexec_fn=set_resource_limits
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", "Execution timed out"
    finally:
        os.unlink(temp_file)

# E2B Sandbox: 클라우드 기반 안전한 실행 환경
from e2b_code_interpreter import CodeInterpreter

def execute_with_e2b(code: str):
    with CodeInterpreter() as sandbox:
        execution = sandbox.notebook.exec_cell(code)
        return execution.text, execution.error
```

---

## Memory 시스템

### 계층적 메모리

```python
from typing import Optional
import chromadb
from datetime import datetime

class AgentMemory:
    def __init__(self, max_context_tokens=4096):
        self.short_term = []           # 현재 대화 (in-context)
        self.max_tokens = max_context_tokens

        # 벡터 DB (long-term)
        self.chroma_client = chromadb.Client()
        self.long_term = self.chroma_client.create_collection("memories")

    def add_to_short_term(self, message: dict):
        """단기 메모리에 추가 (sliding window)"""
        self.short_term.append(message)
        # 컨텍스트 초과 시 오래된 것 제거
        while self.count_tokens(self.short_term) > self.max_tokens:
            # 중요한 것은 long-term으로 이동
            old_msg = self.short_term.pop(0)
            self.archive_to_long_term(old_msg)

    def archive_to_long_term(self, message: dict):
        """중요한 메시지를 장기 메모리로"""
        content = message.get("content", "")
        embedding = get_embedding(content)
        self.long_term.add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[{"timestamp": datetime.now().isoformat()}],
            ids=[f"mem_{datetime.now().timestamp()}"]
        )

    def retrieve_relevant(self, query: str, k: int = 3) -> list[str]:
        """현재 쿼리에 관련된 장기 메모리 검색"""
        query_embedding = get_embedding(query)
        results = self.long_term.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return results["documents"][0]

    def build_context(self, query: str) -> list[dict]:
        """단기 메모리 + 관련 장기 메모리로 컨텍스트 구성"""
        relevant_memories = self.retrieve_relevant(query)

        context = []
        if relevant_memories:
            context.append({
                "role": "system",
                "content": f"Relevant past information: {chr(10).join(relevant_memories)}"
            })

        context.extend(self.short_term)
        return context
```

---

## Multi-Agent Systems

### Society of Mind / MetaGPT

```python
# 전문화된 에이전트들이 협업
class SoftwareDevelopmentTeam:
    def __init__(self):
        self.product_manager = Agent("PM", "Define requirements and write PRD")
        self.architect = Agent("Architect", "Design system architecture")
        self.developer = Agent("Developer", "Write production code")
        self.qa_engineer = Agent("QA", "Write and run tests")
        self.reviewer = Agent("Reviewer", "Review code and give feedback")

    def develop_feature(self, feature_request: str) -> dict:
        # PRD 작성
        prd = self.product_manager.create_prd(feature_request)

        # 아키텍처 설계
        architecture = self.architect.design(prd)

        # 코드 작성
        code = self.developer.implement(architecture, prd)

        # 테스트
        tests = self.qa_engineer.write_tests(code, prd)
        test_results = self.qa_engineer.run_tests(code, tests)

        # 코드 리뷰
        review = self.reviewer.review(code, architecture)

        # 피드백 반영
        if review.has_issues:
            code = self.developer.fix(code, review.issues)

        return {
            "prd": prd,
            "architecture": architecture,
            "code": code,
            "tests": tests,
            "review": review
        }
```

### Debate / Critique

```
두 에이전트가 서로 토론/비평:

에이전트 A: 답변 생성
에이전트 B: A의 답변 비평 (오류, 개선점)
에이전트 A: 비평 반영 답변 수정
반복...

Society of Mind 논문:
  여러 에이전트 논쟁 → 최종 합의
  단일 에이전트보다 정확도 향상
  특히 수학/논리 문제에서 효과적
```

---

## 최신 Agentic 시스템

### Computer Use (Claude)

```python
import anthropic
import base64
from PIL import ImageGrab

def computer_use_agent(task: str):
    """
    실제 컴퓨터를 제어하는 Agent
    """
    client = anthropic.Anthropic()

    tools = [
        {"type": "computer_20241022", "name": "computer",
         "display_width_px": 1920, "display_height_px": 1080},
        {"type": "text_editor_20241022", "name": "str_replace_editor"},
        {"type": "bash_20241022", "name": "bash"},
    ]

    messages = [{"role": "user", "content": task}]

    while True:
        response = client.beta.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            tools=tools,
            messages=messages,
            betas=["computer-use-2024-10-22"],
        )

        if response.stop_reason == "end_turn":
            break

        # 스크린샷 및 컴퓨터 액션 처리
        for block in response.content:
            if block.type == "tool_use" and block.name == "computer":
                action = block.input["action"]
                if action == "screenshot":
                    screenshot = take_screenshot()
                    # 스크린샷을 다음 메시지에 포함
                elif action == "left_click":
                    x, y = block.input["coordinate"]
                    perform_click(x, y)
                elif action == "type":
                    text = block.input["text"]
                    type_text(text)

        messages.append({"role": "assistant", "content": response.content})
        # 다음 iteration에 화면 상태 전달
```

### SWE-agent [Princeton, 2024]

```
소프트웨어 엔지니어링 자동화:

  GitHub Issue 수신
    ↓
  Repository 클론
    ↓
  코드 이해 (grep, find, cat)
    ↓
  버그 재현
    ↓
  수정 구현
    ↓
  테스트 실행
    ↓
  PR 생성

SWE-bench 성능 (2024):
  GPT-4: 12.5% 해결
  Claude-3-Opus: 13.9% 해결
  SWE-agent: 18.6% 해결 (specialized)
  Best SOTA: ~50%+ (아직 도전적)
```

---

## Agent 평가

```python
# AgentBench: LLM Agent 종합 평가
# 8개 환경: OS, DB, Knowledge Graph, Digital Card Game,
#           Lateral Thinking Puzzles, HouseHold, WebShopping, WebArena

evaluation_metrics = {
    "task_completion_rate": "태스크 완료율 (핵심 지표)",
    "efficiency": "태스크 완료까지 평균 스텝 수",
    "error_rate": "도구 호출 오류율",
    "cost": "API 호출 비용",
    "safety": "위험한 액션 실행 여부",
}

# WebArena: 실제 웹 환경에서 태스크 수행 평가
# tau-bench: 소매/항공 예약 에이전트 평가
# GAIA: 일반 AI 보조자 평가
```

---

## Agent 핵심 도전 과제

```
1. 오류 전파 (Error Propagation):
   한 단계 실수 → 이후 모든 단계에 영향
   해결: 중간 검증, Rollback, 오류 감지 → 재계획

2. Long-horizon Planning:
   수십~수백 스텝의 장기 계획
   현재 LLM: 5-10 스텝까지는 잘하지만 그 이상은 어려움

3. 비용 (Cost):
   각 스텝마다 API 호출 → 비용 폭발
   100 스텝 에이전트 × $0.01/call = $1/태스크
   해결: 효율적 스케줄링, 작은 모델 활용

4. 환각된 Tool 호출:
   존재하지 않는 함수 호출, 잘못된 파라미터
   해결: Function schema 엄격히, 검증 로직

5. 안전성 (Safety):
   자율 에이전트의 의도치 않은 행동
   실수로 파일 삭제, 이메일 발송 등
   해결: Sandboxing, 확인 단계, 권한 제한

6. 반복/루프 (Loops):
   에이전트가 같은 행동 반복
   해결: 방문 이력 추적, 다양성 유도, max_iterations
```

---

## Further Questions

**Q. LLM Agent의 핵심 도전 과제는?**
> 1) 오류 전파: 한 단계 실수가 뒤에 영향. 2) Long-horizon planning (5-10 스텝 이상). 3) 비용: 많은 API 호출. 4) 환각된 Tool 호출. 5) 안전성 (자율 에이전트의 의도치 않은 행동). 6) 루프 탐지 및 종료. 최근 연구: Reflexion, Self-RAG 등으로 오류 복구 능력 개선.

**Q. ReAct와 단순 CoT의 차이는?**
> CoT: 내부 추론만 (모델 내부 지식에만 의존). ReAct: 추론 + 실제 행동(외부 도구 호출) + 관찰. 외부 정보 필요한 태스크(최신 정보, 계산, 파일 접근)에서 ReAct 우수. CoT 오류: 환각 가능. ReAct: 실제 도구 결과로 검증. 단, ReAct는 도구 호출 오류 가능성 있음.

**Q. Multi-agent 시스템이 단일 에이전트보다 나은 경우는?**
> 역할 분담이 명확한 경우 (코딩 + 리뷰 + 테스팅). 상호 검증이 필요한 경우 (Debate, Critique). 병렬 실행 가능한 경우 (독립적 하위 태스크). 전문 지식이 다른 경우 (도메인별 전문 에이전트). 단점: 에이전트 간 통신 비용, 오류 전파 복잡화.

**Q. Function Calling vs ReAct 프롬프팅 차이는?**
> Function Calling: 공식 API 지원 (structured JSON 출력). 신뢰할 수 있는 파라미터 파싱, 더 안정적. ReAct 프롬프팅: 일반 텍스트로 도구 지정. 유연하지만 파싱 오류 가능. 현대 LLM API: 대부분 Function Calling 지원 (더 신뢰적). ReAct 프롬프팅: Function Calling 미지원 모델에서 사용.
