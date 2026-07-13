# 解锁大模型智能体的高阶范式 Plan-and-Solve

在构建大语言模型（LLM）智能体时，经典的 ReAct 范式（思考-行动-观察）以其卓越的灵活性深入人心。然而，在实际工程中，ReAct 就像一个“走一步看一步”的探险家，容易在面对多步骤、高难度的复杂任务时“迷失方向”或陷入无效的死循环。

为了攻克这一痛点，业界提出了另一种更加结构化的智能体范式——**Plan-and-Solve（规划与解决）**。如果说 ReAct 是根据现场蛛丝马迹随时调整方向的探侦，那么 Plan-and-Solve 则更像一位严谨的建筑师，在动工之前，必须先绘制出完整的蓝图。

## 1 Plan-and-Solve范式

Plan-and-Solve 概念由 Lei Wang 等人于 2023 年提出，其核心动机是为了解决思维链（CoT）及传统步进式方法在处理多步骤问题时容易“偏离轨道”的缺陷。

与 ReAct 将思考和行动揉碎在每一步不同，Plan-and-Solve 将整个任务流程彻底解耦为两个核心阶段：

1. **规划阶段 (Planning Phase)**：智能体接收到用户的完整问题后，**不直接去解决问题或调用工具**。而是先调用一次 LLM，将大任务拆解，制定出一个清晰、分步骤的**行动计划（Plan）**。
2. **执行阶段 (Solving Phase)**：智能体获得完整的计划后，进入执行状态。它会**严格按照计划中的步骤，逐一执行**（如调用工具、加工数据等），直到所有步骤完成，最终输出答案。

## 2 核心架构与编码实现

构建一个 Plan-and-Solve 智能体，关键在于**计划的生成**与**状态管理器的维护**。以下是底层的核心实现逻辑。

### 2.1 规划阶段：提示词设计

在规划阶段，我们需要强迫大模型输出一个结构化的步骤列表。

```text
你是一个高效率的任务规划专家。
请分析用户的原始问题，并将其拆解为一系列逻辑清晰、可执行的步骤计划。

请严格按照以下格式输出你的计划：
Plan:
1. [步骤1：具体要做的事情]
2. [步骤2：具体要做的事情]
3. ...

原始问题: {question}

```

### 2.2 状态管理器与执行驱动器

在执行阶段，智能体需要维持一个“指针”，记录当前执行到了第几步，并把历史步骤的执行结果（上下文）不断喂给执行器。

```python
class PlanAndSolveAgent:
    def __init__(self, llm_client, tool_executor):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.plan = []
        self.execution_history = []

    def run(self, question):
        # 1. 规划阶段：生成完整计划
        print("📋 正在制定全局计划...")
        plan_prompt = PLAN_PROMPT_TEMPLATE.format(question=question)
        plan_output = self.llm_client.think([{"role": "user", "content": plan_prompt}])
        self.plan = self._parse_plan(plan_output)
        
        print(f"✅ 计划制定完成，共 {len(self.plan)} 步。")

        # 2. 执行阶段：严格按计划推进
        for step_idx, step_desc in enumerate(self.plan):
            print(f"\n⚡ 正在执行第 {step_idx + 1} 步: {step_desc}")
            
            # 构建单步执行的上下文，包含历史执行结果
            solve_prompt = SOLVE_PROMPT_TEMPLATE.format(
                question=question,
                current_step=step_desc,
                history="\n".join(self.execution_history)
            )
            
            # 调用 LLM 决定该步如何行动（如调用什么工具）
            response = self.llm_client.think([{"role": "user", "content": solve_prompt}])
            
            # 解析并执行工具
            tool_name, tool_input = self._parse_action(response)
            if tool_name:
                observation = self.tool_executor.execute(tool_name, tool_input)
                self.execution_history.append(f"步骤: {step_desc}\n结果: {observation}")
            else:
                self.execution_history.append(f"步骤: {step_desc}\n结果: {response}")

        # 3. 汇总输出
        print("\n🏁 所有步骤执行完毕，正在生成最终答案...")
        final_prompt = f"基于以下执行历史，回答问题：{question}\n历史：{self.execution_history}"
        return self.llm_client.think([{"role": "user", "content": final_prompt}])

```

## 3 Plan-and-Solve vs ReAct

| 特性 | ReAct (反应式) | Plan-and-Solve (规划式) |
| --- | --- | --- |
| **决策机制** | 动态调整，边想边做 | 全局规划，三思而后行 |
| **Token 消耗** | **高**（历史记录呈雪崩式累积） | **中**（各步骤相对独立，上下文受控） |
| **死循环风险** | **高**（易在同一个错误里打转） | **低**（按步骤线性推进，天然具备终止边界） |
| **适用场景** | 开放式探索、未知环境、强交互任务 | 目标明确的多步骤任务、复杂数学推理 |


## 4 局限性与工程思考

尽管 Plan-and-Solve 在多步骤任务中表现出极高的稳定性和 Token 经济性，但它也有致命的弱点：**对未知变化的应对能力较差**。

如果在第一步执行工具时，返回了完全出乎意料的错误结果，原本在“温室”里设计好的第 2、3、4 步计划就会全盘崩溃。由于它缺乏 ReAct 那种随时修正计划的弹性，往往会“硬着头皮”把错误的计划执行到底。

**工程启示：**
在实际生产中，目前主流的大模型 Agent 工具往往会将两者融合——**采用 Plan-and-Solve 搭建宏观骨架（顶层规划），在具体的单步执行中嵌套 ReAct 循环（底层容错）**。这种动静结合的混合架构，才是让智能体走向工业级落地的核心解法。
