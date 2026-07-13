# 构建具备“反思”能力的 Reflection 智能体

ReAct（边想边做）和 Plan-and-Solve（三思而后行）范式中，智能体一旦执行完既定步骤并输出答案，其工作流程便宣告结束。然而，大模型生成的初始答案往往可能存在逻辑漏洞、事实错误或未完全满足边界条件。

如何让智能体像人类一样，在交卷前进行自我检查，甚至在发现错误后主动修正？这就需要引入智能体的高阶进化范式——**Reflection（反思机制）**。


## 1 Reflection范式

Reflection 机制的核心思想是在智能体架构中引入一个**事后（post-hoc）自我校正循环**。它的灵感来源于人类的创作与修正过程：写完初稿 $\rightarrow$ 审校反思 $\rightarrow$ 修改润色。该工作流程可以概括为简洁的三步循环：**执行 $\rightarrow$ 反思 $\rightarrow$ 优化**。

1. **执行 (Execution)**：智能体使用常规方法（如 ReAct）尝试完成任务，生成一个初步的解决方案，这可以看作“初稿”。
2. **反思 (Reflection)**：智能体转换角色为“评审员”，审视第一步生成的初稿，从**事实性错误**、**逻辑漏洞**以及**边界遗漏**等维度进行评估，生成结构化的反馈意见（Feedback）。
3. **优化 (Refinement)**：智能体将“初稿”和“反馈意见”作为新的上下文重新输入给大模型，要求其针对具体问题进行定向修正，产出“修订稿”。

这个循环可以重复进行，直到评审员无法发现新问题，或达到了预设的迭代上限。


## 2 核心架构与编码实现

构建一个 Reflection 智能体，关键在于将执行器（Generator）**与**反思器（Evaluator）解耦，让它们各司其职。

### 2.1 反思阶段：提示词设计

反思器的提示词需要极其严厉且具批判性，强迫模型吹毛求疵。

```text
你是一个挑剔的代码与逻辑评审专家。
请审视用户的原始问题以及当前给出的初始解答，找出其中存在的所有缺陷。

请重点检查以下维度：
1. 是否存在事实性或计算错误？
2. 是否遗漏了问题中的关键约束条件？
3. 是否有更优、更高效的解法？

请严格按照以下格式输出你的反思结果：
Critique: [指出具体的问题所在，若完全正确则输出 None]
Suggestions: [给出具体的修正建议]

原始问题: {question}
初始解答: {generation}

```

### 2.2 自我修正循环的实现

在代码层面，通过一个 `for` 循环控制迭代次数，动态将反思结果喂回给执行器。

```python
class ReflectionAgent:
    def __init__(self, llm_client, max_iters=3):
        self.llm_client = llm_client
        self.max_iters = max_iters

    def run(self, question):
        # 1. 执行阶段：生成初稿
        print("✍️ 正在生成初始解答...")
        gen_prompt = f"请回答以下问题：{question}"
        current_solution = self.llm_client.think([{"role": "user", "content": gen_prompt}])
        
        # 进入“反思-优化”循环
        for i in range(self.max_iters):
            print(f"\n🔍 正在进行第 {i+1} 轮自我反思...")
            
            # 2. 反思阶段：评审员上线
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(
                question=question, generation=current_solution
            )
            critique_output = self.llm_client.think([{"role": "user", "content": reflect_prompt}])
            
            # 检查是否无需修改
            if "Critique: None" in critique_output or "没有问题" in critique_output:
                print("🎉 评审通过，无需修正！")
                break
                
            # 3. 优化阶段：根据反馈进行修正
            print(f"🛠️ 发现缺陷，正在根据反馈进行优化...")
            refine_prompt = f"""
            原始问题: {question}
            当前解答: {current_solution}
            评审意见: {critique_output}
            
            请结合评审意见和修正建议，输出全面优化后的最终解答。
            """
            current_solution = self.llm_client.think([{"role": "user", "content": refine_prompt}])
            
        return current_solution

```


## 3 三大范式横向对比

至此，我们已经集齐了智能体的三大经典范式：

| 范式 | 核心隐喻 | 核心优势 | 致命缺陷 |
| --- | --- | --- | --- |
| **ReAct** | 边想边做的探险家 | 动态调整，灵活度极高 | 缺乏全局规划，易死循环 |
| **Plan-and-Solve** | 先画蓝图的建筑师 | 逻辑清晰，节约 Token 消耗 | 容错差，无法应对环境突变 |
| **Reflection** | 闭门思过的创作者 | 质量上限极高，具备纠错能力 | 极度消耗 Token，响应延迟高 |

## 4 局限性与工程思考

Reflection 机制虽然强大，但在落地时必须进行**成本收益分析**。
由于每多一轮反思，就需要反复调用大模型，这会导致 **Token 消耗和响应延迟（Latency）成倍增加**。如果任务本身非常简单，引入反思无异于大炮打蚊子。

此外，反思的成败高度依赖于大模型自身的“元认知”能力。如果模型本身能力不足，极易陷入“自己看不出自己的错”**（老好人现象），或者**“没病找病，越改越错”的窘境。

因此，在工业级 Agent 设计中，反思机制通常配合**外部强校验工具**（如编译报错信息、静态代码检查器、运行单元测试）共同使用。用确定性的工具结果提供反思依据，才能真正让智能体具备工业级的自我进化能力。
