# Agent Skills 完整指南

## 一、什么是 Agent Skills？

**Agent Skills** 是 Anthropic 提出的**开放标准**，用于给 AI Agent 扩展专业化能力和工作流程。

### 核心定义

> 一个 skill 就是一个文件夹，其中包含 `SKILL.md` 文件。这个文件包含元数据（name、description）和指令，告诉 agent 如何执行特定任务。Skill 还可以捆绑脚本、参考资料、模板等资源。

```
my-skill/
├── SKILL.md          # 必需：元数据 + 指令
├── scripts/          # 可选：可执行代码
├── references/       # 可选：文档
├── assets/           # 可选：模板、资源
└── ...               # 任意额外文件或目录
```

---

## 二、为什么需要 Agent Skills？

Agent 越来越强大，但**缺乏完成真实工作所需的上下文**。Skill 解决这一问题：

| 价值 | 说明 |
|------|------|
| **领域专业知识** | 将专业领域知识（法律审查、数据分析、演示文稿格式）封装为可复用指令 |
| **可重复的工作流** | 将多步骤任务转为一致、可审计的流程 |
| **跨产品复用** | 一次构建，可在任何兼容 agent 中使用 |

---

## 三、如何工作：渐进式披露

Agent 通过**三阶段渐进式加载**技能：

| 阶段 | 加载内容 | 目的 |
|------|----------|------|
| **Discovery（发现）** | 仅加载 name + description | 判断是否相关 |
| **Activation（激活）** | 加载完整 SKILL.md 指令 | 任务匹配时触发 |
| **Execution（执行）** | 执行指令、运行脚本、读取参考文件 | 完成任务 |

**关键优势**：完整指令只在需要时加载，agent 可以同时管理大量技能而不会撑爆 context。

---

## 四、SKILL.md 格式详解

### Frontmatter 必填字段

| 字段 | 约束 |
|------|------|
| **name** | 最多 64 字符，仅小写字母+数字+连字符，不能以连字符开头/结尾，不能有连续 `--` |
| **description** | 最多 1024 字符，非空，描述技能做什么 + 何时使用 |

### 可选字段

| 字段 | 说明 |
|------|------|
| `license` | 许可证名称或文件路径 |
| `compatibility` | 环境需求（如 "Requires Python 3.14+"） |
| `metadata` | 任意 key-value 映射（author、version 等） |
| `allowed-tools` | 预批准可执行工具列表（实验性） |

### 示例

```yaml
---
name: pdf-processing
description: >
  Extract text and tables from PDF files, fill forms, and merge
  multiple PDFs. Use when working with PDF documents or when
  the user mentions PDFs, forms, or document extraction.
license: Apache-2.0
metadata:
  author: example-org
  version: "1.0"
---
```

---

## 五、最佳实践核心原则

### 5.1 从真实专业知识出发

❌ **错误做法**：让 LLM 基于通用知识生成技能 → 结果是泛泛的程序（"handle errors appropriately"）

✅ **正确做法**：从真实领域知识出发：
- 与 agent 一起完成实际任务，提供上下文和纠正
- 从现有项目 artifact 合成（incident reports、runbooks、API specs）
- 用真实失败案例和修复迭代优化

### 5.2 上下文要精准——只加 agent 缺乏的

❌ **过度说明**（agent 本来就知道）：
> PDF (Portable Document Format) files are a common file format that contains text...

✅ **直击要害**：
> Use pdfplumber for text extraction. For scanned documents, fall back to pdf2image with pytesseract.

**自测问题**：如果 agent 没有这个技能会出错吗？如果不会 → 删掉。

### 5.3 设计有边界的单元

- **太窄**：一个简单任务需要触发多个技能 → 开销高、指令冲突
- **太宽**：技能难以精确激活
- 一个技能应该覆盖一个连贯的工作单元（如"查数据库+格式化结果"）

### 5.4 指令要有粗细粒度

| 任务类型 | 策略 | 示例 |
|----------|------|------|
| 多种方案可行 | 给出默认值 + 提及替代选项 | "Use pdfplumber... for OCR use pdf2image instead" |
| 操作脆弱/一致性重要 | 给出精确指令 | "Run exactly: `python scripts/migrate.py --verify --backup`" |
| 灵活任务 | 给出目标 + 解释原因 | "Do X because Y tends to cause Z" |

### 5.5 实用模板

**Gotchas 列表**（最高价值内容）：
```markdown
## Gotchas
- `users` 表使用软删除，查询必须包含 `WHERE deleted_at IS NULL`
- user ID 在数据库是 `user_id`，在 auth service 是 `uid`，在 billing API 是 `accountId`
- `/health` 返回 200 不代表数据库连接正常，用 `/ready` 检查
```

**输出格式模板**：
```markdown
## Report structure
# [Analysis Title]
## Executive summary
[概述]

## Key findings
- Finding 1 with supporting data

## Recommendations
1. 具体可操作建议
```

**多步骤工作流检查清单**：
```markdown
## Form processing workflow
- [ ] Step 1: Analyze the form (run `scripts/analyze_form.py`)
- [ ] Step 2: Create field mapping (edit `fields.json`)
- [ ] Step 3: Validate mapping (run `scripts/validate_fields.py`)
...
```

### 5.6 验证循环

```
1. 执行任务
2. 运行验证脚本
3. 若失败 → 修复 → 重新验证
4. 通过后才继续
```

### 5.7 不要过度堆砌

- SKILL.md **建议 < 500 行 / < 5000 tokens**
- 详细参考材料 → 放到 `references/` 目录
- 告诉 agent **何时**加载哪个文件（"if API returns non-200, read references/api-errors.md"）

---

## 六、技能输出质量评估

### 测试用例结构

```json
{
  "skill_name": "csv-analyzer",
  "evals": [
    {
      "id": 1,
      "prompt": "I have a CSV in data/sales_2025.csv. Find top 3 months by revenue and make a bar chart.",
      "expected_output": "A bar chart image showing top 3 months by revenue, with labeled axes.",
      "files": ["evals/files/sales_2025.csv"],
      "assertions": [
        "The output includes a bar chart image file",
        "The chart shows exactly 3 months",
        "Both axes are labeled"
      ]
    }
  ]
}
```

### 评估流程

1. **运行两次**：一次有 skill，一次无 skill（baseline）
2. **收集 timing**：token 消耗和耗时
3. **写断言**：可编程验证的声明（如"输出是有效 JSON"）
4. **分级**：PASS/FAIL + 证据
5. **聚合**：pass rate、delta、stddev

### 断言原则

| ✅ 强断言 | ❌ 弱断言 |
|-----------|-----------|
| "Output is valid JSON" | "Output is good" |
| "Chart has labeled axes" | "Uses exactly phrase 'Total Revenue'" |
| "Report includes at least 3 recommendations" | - |

### 分析模式

| 信号 | 行动 |
|------|------|
| 断言两版本都 pass | 移除这个断言（模型本来就行） |
| 断言两版本都 fail | 修复断言本身或测试用例 |
| 有 skill 通过 / 无 skill 失败 | 这是技能真正带来的价值 |
| 时间/token 异常高 | 读执行 transcript 找瓶颈 |

### 迭代循环

```
评估当前描述 → 识别 train 集失败 → 改进 description
        ↑                                      ↓
   验证集检查   ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
   
当 pass rate 稳定或无明显改善时停止
```

---

## 七、描述优化详解

### 触发原理

description 是 agent **决定是否加载技能的唯一依据**。描述不精确 → 该触发时没触发 / 不该触发时乱触发。

### 写作原则

- **用命令式**："Use this skill when..." 而非 "This skill does..."
- **聚焦用户意图**，不聚焦实现细节
- **宁多勿少**：明确列出适用场景（包括用户没有直接提到领域关键词的情况）
- **保持简洁**：几句话到一段落，< 1024 字符

### 评估查询集

```json
[
  { "query": "...", "should_trigger": true },
  { "query": "...", "should_trigger": false }
]
```

- **should-trigger**：正式/随意措辞、不同显式程度、不同复杂度
- **should-not-trigger**：关键词重叠但任务不同的 near-miss（如"Excel editing" vs "CSV analysis"）
- 每个 query 运行 3 次，计算触发率

### 训练/验证分离

- **Train set（60%）**：引导改进方向
- **Validation set（40%）**：检查改进是否泛化（不要用它来调优）

### 常见失败

| 失败类型 | 原因 | 解决方案 |
|----------|------|----------|
| should-trigger 失败 | 描述太窄 | 扩大范围或增加上下文 |
| should-not-trigger 误触发 | 描述太宽 | 增加边界描述 |
| 过拟合 | 用了 query 中的具体关键词 | 泛化到类别/概念层面 |

---

## 八、支持 Agent Skills 的产品

Agent Skills 已形成**生态**。按类别整理：

### AI Coding Agent / IDE

| 产品 | 说明 |
|------|------|
| Claude Code | Anthropic 官方 Agent 编程工具 |
| Cursor | AI 编辑器和 coding agent |
| GitHub Copilot | 代码补全+Pair programming |
| VS Code | Agent Skills 原生支持 |
| Junie | 基于 IntelliJ 平台，理解项目和编辑器 |
| Roo Code | 深度项目上下文，多步 agent 编程 |
| TRAE | 自适应 AI IDE |
| Workshop | 跨平台 coding agent，多 LLM + 子 agent |

### Terminal / CLI Agent

| 产品 | 说明 |
|------|------|
| Gemini CLI | Google 开源终端 Agent |
| Claude CLI | Anthropic 官方命令行 |
| pi | 最小化终端 coding harness |
| OpenAI Codex | OpenAI 编程 Agent |
| OpenHands | 云端 coding agent，支持扩展到千量级 |
| nanobot | 轻量跨平台 agent，支持 MCP |

### 专业/垂直 Agent

| 产品 | 说明 |
|------|------|
| Letta | 有记忆的有状态 Agent 平台 |
| Mux | 并行 coding agent，browser/desktop 运行 |
| Ona | 云端后台 agent 团队 |
| Databricks Cortex Code | 数据工程/ML 专用 |
| Agentman | 医疗工作流自动化 |

### 框架/平台

| 产品 | 说明 |
|------|------|
| Spring AI | Java AI 应用开发框架 |
| Laravel Boost | Laravel 最佳实践扩展 |
| fast-agent | 简单可扩展 LLM 交互框架 |

### 其他

- Kiro（规范驱动开发）、Qodo（代码质量审查）、Firebender（Android 原生 Agent）、Command Code（学习编码风格的个性化 agent）

---

## 九、快速上手示例

### 创建 roll-dice 技能

路径：`.agents/skills/roll-dice/SKILL.md`

```yaml
---
name: roll-dice
description: >
  Roll dice using a random number generator.
  Use when asked to roll a die (d6, d20, etc.),
  roll dice, or generate a random dice roll.
---

To roll a die, use the following command that generates
a random number from 1 to the given number of sides:

```bash
echo $((RANDOM % <sides> + 1))
```
```

在 VS Code 中开启 Copilot Chat → Agent 模式 → 输入 `/skills` 确认技能已加载 → 输入 "Roll a d20" 即可触发。

---

## 十、核心设计哲学

> **Progressive Disclosure（渐进式披露）**是 Agent Skills 的核心。
> - 启动时 agent 只知道每个 skill 的 name + description
> - 只有在任务匹配时，才加载完整指令
> - 参考文件仅在需要时读取
> 这使得 agent 可以管理大量技能而不浪费 context

---

## 一句话总结

> **Agent Skills 是一个轻量级、开放的 Agent 能力扩展标准，通过 SKILL.md + 可选脚本/资源文件，让 AI Agent 在需要时精确加载领域专业知识和工作流程，解决 agent 缺乏真实工作上下文的根本问题。**
