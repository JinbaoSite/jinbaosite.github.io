# Jinbao Blog

这个仓库只维护博客的 Markdown 内容，网站由
[JinbaoSite/markdown-pages](https://github.com/JinbaoSite/markdown-pages)
自动构建并发布到 GitHub Pages。

访问地址：<https://jinbaosite.github.io/blog/>

## 添加文章

按照主题把 Markdown 放入相应目录：

```text
ml/          机器学习
dl/          深度学习
llm/         LLM
recsys/      推荐算法
agent/       Agent
projects/    项目
```

每篇文章至少包含一个一级标题：

```markdown
# 文章标题

正文内容。
```

推送到 `main` 后，GitHub Actions 会自动重新发布网站。

