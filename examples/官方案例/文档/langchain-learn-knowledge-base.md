# LangChain 官方教程知识库

> 来源: https://docs.langchain.com/oss/python/learn
> 更新时间: 2025

本文档整理了 LangChain 官方 Learn 板块的全部教程案例，包含每个案例的链接、主要技术栈和学习目标。

---

## 目录

- [一、Deep Agents 案例](#一deep-agents-案例)
- [二、LangChain 案例](#二langchain-案例)
- [三、LangGraph 案例](#三langgraph-案例)
- [四、Multi-agent 多代理案例](#四multi-agent-多代理案例)
- [五、概念概述 (Conceptual)](#五概念概述-conceptual)
- [六、附加资源](#六附加资源)

---

## 一、Deep Agents 案例

### 1.1 Data Analysis Agent (数据分析代理)

| 项目 | 内容 |
|------|------|
| **链接** | https://docs.langchain.com/oss/python/deepagents/data-analysis |
| **中文名** | 构建数据分析代理 |
| **主要技术** | `deepagents` 库、sandboxed code execution、Slack API 集成、LangGraph Checkpointer |
| **核心概念** | Backends（沙盒代码执行）、自定义 Tools、文件系统操作、消息推送 |
| **学习目标** | 1. 理解 Deep Agents 的 backend 系统<br>2. 学会构建能分析 CSV 数据并生成可视化的代理<br>3. 掌握自定义工具的开发（文件读写、Slack 推送）<br>4. 理解多轮对话的 checkpointer 配置 |

**案例概述**：构建一个能分析 CSV 数据、生成可视化图表、并通过 Slack 发送报告的数据分析代理。

---

## 二、LangChain 案例

### 2.1 Semantic Search (语义搜索)

| 项目 | 内容 |
|------|------|
| **链接** | https://docs.langchain.com/oss/python/langchain/knowledge-base |
| **中文名** | 构建语义搜索引擎 |
| **主要技术** | Document Loaders、Text Splitters、Embeddings、Vector Stores、Retrievers |
| **核心概念** | PDF 文档加载、文本分块、向量嵌入、相似度搜索、语义检索 |
| **学习目标** | 1. 掌握 LangChain 的文档加载器使用<br>2. 理解文本分割策略（RecursiveCharacterTextSplitter）<br>3. 学会使用多种 Embedding 模型（OpenAI、Google、Cohere 等）<br>4. 掌握 Vector Store 的选择与使用（Chroma、FAISS、Pinecone 等）<br>5. 理解 Retriever 的构建与应用 |

### 2.2 RAG Agent (检索增强生成代理)

| 项目 | 内容 |
|------|------|
| **链接** | https://docs.langchain.com/oss/python/langchain/rag |
| **中文名** | 构建 RAG 代理 |
| **主要技术** | WebBaseLoader、RAG、Agent、Tool、create_agent、Streaming |
| **核心概念** | 索引（Indexing）、检索（Retrieval）、生成（Generation）、语义搜索、Agent 工具调用 |
| **学习目标** | 1. 理解 RAG 的完整流程（索引→检索→生成）<br>2. 学会构建基于 Agent 的 RAG 系统<br>3. 掌握 RAG Agent 与 RAG Chain 的区别与适用场景<br>4. 理解动态上下文注入（middleware） |

### 2.3 SQL Agent (SQL 代理)

| 项目 | 内容 |
|------|------|
| **链接** | https://docs.langchain.com/oss/python/langchain/sql-agent |
| **中文名** | 构建 SQL 代理 |
| **主要技术** | SQLDatabaseToolkit、SQLDatabase、create_agent、Human-in-the-Loop |
| **核心概念** | SQL 查询生成、Schema 探索、查询验证、Human-in-the-loop 审核 |
| **学习目标** | 1. 学会构建与 SQL 数据库交互的代理<br>2. 掌握 SQLDatabaseToolkit 的使用<br>3. 理解查询验证机制（sql_db_query_checker）<br>4. 实现 Human-in-the-loop 审核机制 |

### 2.4 Voice Agent (语音代理)

| 项目 | 内容 |
|------|------|
| **链接** | https://docs.langchain.com/oss/python/langchain/voice-agent |
| **中文名** | 构建语音代理 |
| **主要技术** | Speech-to-Text (STT)、Text-to-Speech (TTS)、WebSocket、RunnableGenerator、异步流处理 |
| **核心概念** | Sandwich 架构、S2S 架构、实时语音流处理、AssemblyAI、Cartesia |
| **学习目标** | 1. 理解两种语音代理架构（STT→Agent→TTS vs S2S）<br>2. 掌握实时音频流处理<br>3. 学会集成 STT/TTS 提供商<br>4. 理解异步生成器在流处理中的应用 |

---

## 三、LangGraph 案例

### 3.1 Custom RAG Agent (自定义 RAG 代理)

| 项目 | 内容 |
|------|------|
| **链接** | https://docs.langchain.com/oss/python/langgraph/agentic-rag |
| **中文名** | 使用 LangGraph 构建自定义 RAG 代理 |
| **主要技术** | LangGraph Graph API、State、Nodes、Edges、Conditional Edges、ToolNode |
| **核心概念** | 状态管理、节点定义、边路由、条件边、文档相关性评分、问题重写 |
| **学习目标** | 1. 深入理解 LangGraph 的 StateGraph API<br>2. 学会构建自定义的 Agentic RAG 流程<br>3. 掌握文档相关性评分（GradeDocuments）<br>4. 实现查询重写（Query Rewriting）机制<br>5. 理解条件边在路由中的应用 |

### 3.2 Custom SQL Agent (自定义 SQL 代理)

| 项目 | 内容 |
|------|------|
| **链接** | https://docs.langchain.com/oss/python/langgraph/sql-agent |
| **中文名** | 使用 LangGraph 构建自定义 SQL 代理 |
| **主要技术** | LangGraph、StateGraph、ToolNode、Human-in-the-Loop (interrupt) |
| **核心概念** | 强制工具调用、查询检查节点、图可视化、持久化层、interrupt 机制 |
| **学习目标** | 1. 掌握 LangGraph 中强制工具调用的实现<br>2. 理解 SQL 查询验证流程<br>3. 学会实现 Human-in-the-loop 审核<br>4. 掌握图的编译与可视化 |

---

## 四、Multi-agent 多代理案例

### 4.1 Subagents: Personal Assistant (子代理：个人助手)

| 项目 | 内容 |
|------|------|
| **链接** | https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant |
| **中文名** | 构建基于子代理的个人助手 |
| **主要技术** | Supervisor Pattern、多代理协作、create_agent、Human-in-the-Loop Middleware |
| **核心概念** | 主管模式、子代理封装、工具包装、对话记忆、多域任务协调 |
| **学习目标** | 1. 理解 Supervisor 模式的设计原理<br>2. 学会构建分层代理架构<br>3. 掌握子代理作为工具的封装方法<br>4. 实现 Human-in-the-loop 审核<br>5. 理解信息流控制与上下文传递 |

### 4.2 Handoffs: Customer Support (交接：客户支持)

| 项目 | 内容 |
|------|------|
| **链接** | https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support |
| **中文名** | 构建客户支持状态机 |
| **主要技术** | State Machine Pattern、Middleware、Command、Checkpointer、Message Summarization |
| **核心概念** | 状态机工作流、步骤配置、状态转换、对话记忆压缩 |
| **学习目标** | 1. 掌握基于状态机的代理设计<br>2. 学会使用 Middleware 动态配置代理行为<br>3. 实现状态转换与工作流控制<br>4. 理解消息摘要压缩长对话 |

### 4.3 Router: Knowledge Base (路由：知识库)

| 项目 | 内容 |
|------|------|
| **链接** | https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base |
| **中文名** | 构建多源知识库路由器 |
| **主要技术** | Router Pattern、StateGraph、Send API、Parallel Execution、Structured Output |
| **核心概念** | 查询分类、并行执行、结果合成、多源知识整合 |
| **学习目标** | 1. 理解 Router 模式与 Subagents 模式的区别<br>2. 学会构建多源知识路由系统<br>3. 掌握并行执行与结果收集（Send API、Reducer）<br>4. 实现查询分类与结果合成 |

### 4.4 Skills: SQL Assistant (技能：SQL 助手)

> ⚠️ 详情待补充，可参考官方文档

---

## 五、概念概述 (Conceptual)

### 5.1 Memory (记忆)

| 项目 | 内容 |
|------|------|
| **链接** | https://docs.langchain.com/oss/python/concepts/memory |
| **中文名** | 记忆系统概述 |
| **主要概念** | Short-term Memory、Long-term Memory、Thread、Checkpointer、Store |
| **记忆类型** | Semantic Memory（语义记忆）、Episodic Memory（情景记忆）、Procedural Memory（程序记忆） |
| **学习目标** | 1. 区分短期记忆与长期记忆<br>2. 理解 Checkpointer 与 Thread 的关系<br>3. 掌握 Store 的使用（语义搜索、过滤）<br>4. 理解记忆的写入时机（热路径 vs 后台） |

### 5.2 Context Engineering (上下文工程)

> 链接: https://docs.langchain.com/oss/python/concepts/context

| 核心概念 | 说明 |
|---------|------|
| Prompt Engineering | 提示词设计与优化 |
| System Prompt | 系统级指令配置 |
| Few-shot Learning | 少样本学习与示例提供 |
| Context Window | 上下文窗口管理 |

### 5.3 Graph API (图 API)

> 链接: https://docs.langchain.com/oss/python/langgraph/graph-api

| 核心概念 | 说明 |
|---------|------|
| State | 图状态定义 |
| Nodes | 节点（处理单元） |
| Edges | 边（连接与流向） |
| Conditional Edges | 条件边（动态路由） |
| StateGraph | 状态图构建器 |

### 5.4 Functional API (函数式 API)

> 链接: https://docs.langchain.com/oss/python/langgraph/functional-api

| 核心概念 | 说明 |
|---------|------|
| Runnable | 可运行单元 |
| LCEL | LangChain Expression Language |
| Pipe Operator | 管道操作符 |
| Streaming | 流式处理 |

---

## 六、附加资源

### 6.1 LangChain Academy

| 项目 | 内容 |
|------|------|
| **链接** | https://academy.langchain.com/ |
| **说明** | LangChain 官方课程与练习平台 |

### 6.2 Case Studies (案例研究)

| 项目 | 内容 |
|------|------|
| **链接** | https://docs.langchain.com/oss/python/langgraph/case-studies |
| **说明** | 线上生产环境中团队使用 LangChain/LangGraph 的案例 |

---

## 附录：技术栈快速参考

### 核心库

| 库名 | 用途 |
|------|------|
| `langchain` | 核心框架 |
| `langgraph` | 图构建与代理 |
| `deepagents` | Deep Agents 核心 |
| `langchain-community` | 社区集成 |
| `langchain-core` | 核心抽象 |

### 常用组件

| 组件类型 | 示例 |
|---------|------|
| Document Loaders | WebBaseLoader、PyPDFLoader |
| Text Splitters | RecursiveCharacterTextSplitter |
| Embeddings | OpenAIEmbeddings、CohereEmbeddings |
| Vector Stores | Chroma、FAISS、Pinecone、Milvus |
| Chat Models | ChatOpenAI、ChatAnthropic |
| Tools | @tool 装饰器 |

### 存储与持久化

| 类型 | 组件 |
|------|------|
| 短期记忆 | InMemorySaver、PostgresSaver |
| 长期记忆 | InMemoryStore、PostgresStore |
| 数据库 | SQLDatabase、SQLite |

---

## 文档结构说明

本文档按照以下维度组织：

1. **框架分类**：Deep Agents → LangChain → LangGraph → Multi-agent
2. **每个案例包含**：
   - 官方链接
   - 中文名称
   - 主要技术栈
   - 核心概念
   - 学习目标

适合用于：
- 学习路径规划
- 技术选型参考
- 快速查找示例代码
- 导入到其他 AI 工具进行问答

---

*本文档由自动化工具生成，源自 LangChain 官方文档*
