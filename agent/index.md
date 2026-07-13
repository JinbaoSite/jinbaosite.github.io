---
layout: listpage
title: Agent
subtitle: 工具调用、多智能体
article-list:
  - article-title: Agent Skills 完整指南
    article-url: /agent/agentskills
    article-date: 2026-04-08
    article-desc: 一个 skill 就是一个文件夹，其中包含 `SKILL.md` 文件。这个文件包含元数据（name、description）和指令，告诉 agent 如何执行特定任务。Skill 还可以捆绑脚本、参考资料、模板等资源。
    article-tags: [skill]
  - article-title: 解锁大模型智能体的高阶范式: Plan-and-Solve
    article-url: /agent/plan_and_solve
    article-date: 2026-04-08
    article-desc: Plan-and-Solve将任务解耦为两个阶段：首先是规划阶段，大模型对复杂问题进行全局分析，拆解并制定出结构化的多步骤行动计划；其次是执行阶段，智能体严格按照计划步骤线性推进，逐一调用工具并管理状态，直至产出最终答案。。
    article-tags: [Plan-and-Solve, agent]
  - article-title: 深入浅出大模型智能体经典范式：ReAct
    article-url: /agent/react
    article-date: 2026-04-08
    article-desc: ReAct 范式通过将“推理”与“行动”深度结合，让大模型在每个时间步按 “思考（分析规划） $\rightarrow$ 行动（调用外部工具） $\rightarrow$ 观察（获取环境反馈）” 的闭环交替推进，并不断将结果追加至上下文，实现动态规划与问题解决。
    article-tags: [react, agent]
---
