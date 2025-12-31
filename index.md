---
layout: default
title: Arxiv Daily Paper - 2025-12-31
---

[查看所有摘要归档](archive.md) | 更新日期: 2025-12-31

# Arxiv Daily Paper - 2025-12-31

Updated at: 2025-12-31 10:53:33

## 1. Multilingual Hidden Prompt Injection Attacks on LLM-Based Academic Reviewing
- **Authors**: Panagiotis Theocharopoulos, Ajinkya Kulkarni, Mathew Magimai. -Doss
- **Link**: https://arxiv.org/abs/2512.23684v1
- **Summary**: 【中文标题】基于大语言模型的学术评审中多语言隐藏提示注入攻击研究    【中文总结】本研究针对大语言模型(LLMs)在学术同行评审中的应用风险，构建了包含约500篇ICML录用论文的数据集，通过嵌入四种语言的隐藏对抗性指令进行实验。结果显示，英语、日语和中文的提示注入会显著改变评审分数和录用决定，而阿拉伯语注入几乎无效。该研究揭示了基于LLM的评审系统对文档级提示注入的脆弱性，并发现不同语言之间的攻击效果存在显著差异。

---

## 2. BOAD: Discovering Hierarchical Software Engineering Agents via Bandit Optimization
- **Authors**: Iris Xu, Guangtao Zeng, Zexue He, Charles Jin, Aldo Pareja, Dan Gutfreund, Chuang Gan, Zhang-Wei Hong
- **Link**: https://arxiv.org/abs/2512.23631v1
- **Summary**: 【中文标题】   BOAD：通过多臂老虎机优化发现分层软件工程代理    【中文总结】   该论文针对大语言模型（LLM）在解决长周期、分布外真实世界软件工程（SWE）问题时泛化能力不足的局限性，提出了一种分层多代理系统框架BOAD。该框架将复杂任务分解为由协调器管理的专业化子代理（如定位、编辑、验证），并通过多臂老虎机模型自动优化子代理组合，以高效探索协作策略。实验表明，BOAD在SWE-bench基准测试中超越单代理和人工设计的多代理系统，其36B参数模型在最新SWE-bench-Live榜单上排名第二，优于GPT-4等更大模型，验证了自动分层设计对长周期任务的泛化提升有效性。

---

## 3. Divergent-Convergent Thinking in Large Language Models for Creative Problem Generation
- **Authors**: Manh Hung Nguyen, Adish Singla
- **Link**: https://arxiv.org/abs/2512.23601v1
- **Summary**: 【中文标题】大型语言模型中发散-收敛思维在创造性问题生成的应用    【中文总结】   该论文针对大型语言模型（LLMs）生成教育问题时存在的“人工蜂群思维”效应（即同模型输出相似、跨模型输出同质化）导致问题多样性不足的问题，受Wallas创造力理论和Guilford发散-收敛思维框架启发，提出了CreativeDC——一种将LLM推理显式分解为探索与约束两阶段的新型提示方法。实验表明，该方法在多样性、新颖性指标上显著优于基线模型，同时保持高效用性，且随着采样量增加，其生成独特问题的有效数量增速更快。研究为LLM在创造性教育内容生成中的局限性提供了可扩展的解决方案。

---

## 4. Toward Trustworthy Agentic AI: A Multimodal Framework for Preventing Prompt Injection Attacks
- **Authors**: Toqeer Ali Syed, Mishal Ateeq Almutairi, Mahmoud Abdel Moaty
- **Link**: https://arxiv.org/abs/2512.23557v1
- **Summary**: 【中文标题】   迈向可信的代理人工智能：预防提示注入攻击的多模态框架    【中文总结】   本文针对由大型语言模型（LLM）和视觉语言模型（VLM）驱动的自主代理系统（如LangChain、GraphChain）中可能出现的多模态提示注入（PI）攻击，提出了一种跨代理多模态溯源感知防御框架。该框架通过文本清理代理、视觉清理代理和输出验证代理的协同工作，结合溯源账本记录模态、来源和信任等级元数据，确保代理间通信遵循明确的信任框架，从而阻断恶意指令在工作流中的传播。实验表明，该框架显著提高了多模态注入检测的准确性，最小化了跨代理信任泄漏，并增强了代理执行路径的稳定性，为构建安全可靠的代理人工智能系统提供了有效解决方案。

---

## 5. Lie to Me: Knowledge Graphs for Robust Hallucination Self-Detection in LLMs
- **Authors**: Sahil Kale, Antonio Luca Alfeo
- **Link**: https://arxiv.org/abs/2512.23547v1
- **Summary**: 【中文标题】   骗我试试：基于知识图谱的大语言模型幻觉自检测方法    【中文总结】   本文提出了一种利用知识图谱增强大语言模型（LLMs）幻觉自检测能力的新方法，通过将模型生成的文本转化为实体关系知识图谱，并基于图谱结构评估幻觉可能性。实验在GPT-4o和Gemini-2.5-Flash两种主流模型上进行，结果显示该方法相比传统自检测技术和当前最优方法（如SelfCheckGPT），准确率相对提升16%，F1分数提升20%。研究同时发布了人工增强的基准数据集，验证了知识图谱结构即使面对初始输出错误仍能有效辅助模型分析原子事实，为构建更安全可靠的语言模型提供了低成本、模型无关的解决方案。

---