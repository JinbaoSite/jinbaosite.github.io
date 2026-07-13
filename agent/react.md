# 深入浅出大模型智能体经典范式：ReAct

## 1 ReAct定义

ReAct 概念由 Shunyu Yao 等人于 2022 年提出。在它诞生之前，大模型的应用方法主要分为两类：

1. **纯思考型（如思维链 Chain-of-Thought）**：引导模型进行复杂的逻辑推理，但无法与外部世界交互，容易产生事实幻觉。
2. **纯行动型**：模型直接输出要执行的动作，但缺乏规划和纠错能力。

ReAct 的巧妙之处在于**将“推理”（Reasoning）与“行动”（Acting）显式地结合起来**，形成了一个“思考 $\rightarrow$ 行动 $\rightarrow$ 观察”的协同循环。

* **Thought (思考)**：智能体的“内心独白”。分析当前情况、拆解任务、制定下一步计划，或反思上一步的结果。
* **Action (行动)**：智能体决定采取的具体动作，通常是调用外部工具（如搜索引擎、计算器、API 等）。
* **Observation (观察)**：执行 Action 后从外部环境或工具返回的结果，作为新的事实依据输入给模型。

通过这种循环，推理使得行动更具目的性，而行动的观察结果又反过来修正推理，从而完美弥补了 LLM 缺乏实时信息和易产生幻觉的短板。


## 2 ReAct 的核心设计与实现

要从零构建一个 ReAct 智能体，主要包含三个核心组件：提示词模板、工具执行器和核心循环。

### 2.1 提示词模板设计（System Prompt）

提示词是强制模型进行结构化输出的基石。我们需要定义好规范，让模型知道自己有什么工具，以及必须按照 `Thought`、`Action`、`Observation` 的格式推进。

```text
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:
Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- tool_name[tool_input] : 调用一个可用工具。
- Finish[最终答案] : 当你收集到足够的信息，能够回答用户的最终问题时使用。

现在，请开始解决以下问题:
Question: {question}
History: {history}

```

### 2.2 工具执行器（Tool Executor）

工具是智能体的“手和脚”。以网页搜索工具（如 SerpApi）为例，工具需要包含**名称**、**描述**（供 LLM 判断何时调用）和**执行逻辑**。

```python
class ToolExecutor:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name, description, func):
        self.tools[name] = {"description": description, "func": func}

    def get_available_tools(self):
        return "\n".join([f"- {name}: {info['description']}" for name, info in self.tools.items()])

    def execute(self, name, tool_input):
        if name in self.tools:
            return self.tools[name]["func"](tool_input)
        return f"错误: 未找到名为 {name} 的工具。"

```

### 2.3 核心循环（Execution Loop）

智能体的主体是一个 `while` 循环。它不断把当前历史记录喂给 LLM，解析模型的输出，如果遇到 `Action` 就去执行工具，并将 `Observation` 追加到历史中；如果捕获到 `Finish`，则退出循环输出答案。

```python
import re

class ReActAgent:
    def __init__(self, llm_client, tool_executor, max_steps=5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps

    def run(self, question):
        history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            
            # 1. 组装 Prompt
            tools_desc = self.tool_executor.get_available_tools()
            history_str = "\n".join(history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc, question=question, history=history_str
            )

            # 2. 调用 LLM 思考
            response = self.llm_client.think([{"role": "user", "content": prompt}])
            
            # 3. 解析 Thought 和 Action
            thought, action = self._parse_output(response)
            history.append(f"Thought: {thought}\nAction: {action}")

            # 4. 判断是否结束
            if "Finish" in action:
                final_answer = re.search(r"Finish\[(.*)\]", action).group(1)
                return final_answer

            # 5. 执行工具并获取观察结果
            tool_name, tool_input = self._parse_action(action)
            observation = self.tool_executor.execute(tool_name, tool_input)
            
            # 6. 将结果带入下一次循环
            history.append(f"Observation: {observation}")

```

## 3 ReAct 的优势与局限性

### 3.1 优势：

* **高可解释性**：智能体的每一步思考（Thought）和工具调用（Action）都清晰可见，极易于人类开发者调试和排查错误。
* **准确性高**：通过引入外部真实世界的数据（Observation），大幅降低了模型的幻觉率。

### 3.2 局限与工程挑战：

* **令牌（Token）消耗大**：由于每一步都需要把前面的所有对话历史、思考和观察结果重新作为上下文发给模型，导致 Token 消耗呈滚雪球式增长。
* **易陷入死循环**：若大模型对工具返回的错误 Observation 缺乏反思能力，可能会换着法子反复调用同一个错误工具，导致任务失败。为了防止这种情况，在工程实现上通常必须设置 `max_steps` 作为安全阀。

## 4 总结

ReAct 范式通过精妙的提示词设计，将 LLM 的**内部推理**与外部世界的**动作执行**完美连接。虽然在面对超级复杂的长文本任务时，它可能会显得有些“势单力薄”（后续衍生出了如 Plan-and-Solve、Reflection 等更高级的范式），但作为智能体架构的基石，理解并亲手实现一次 ReAct，是每一位智能体应用创造者的必经之路。
