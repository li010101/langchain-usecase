# =============================================================================
# LangChain Agent 模板 - 基于 LCEL 的链式架构
# 适用场景：简单任务、快速原型、线性流程、RAG
# =============================================================================

from typing import Any, Optional, List, Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import (
    Runnable, 
    RunnableLambda, 
    RunnableParallel, 
    RunnableBranch,
    RunnablePassthrough
)
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory

import os
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# 1. 基础组件 - LLM + Prompt + Parser
# =============================================================================
def create_llm(
    model: str = "gpt-4o",
    temperature: float = 0,
    **kwargs
) -> ChatOpenAI:
    """创建 LLM 实例"""
    return ChatOpenAI(model=model, temperature=temperature, **kwargs)


def create_prompt(
    system_message: str,
    input_variables: Optional[List[str]] = None,
    message_history: bool = False
) -> ChatPromptTemplate:
    """
    创建提示词模板
    
    Args:
        system_message: 系统提示
        input_variables: 输入变量
        message_history: 是否包含消息历史
    
    Returns:
        ChatPromptTemplate 实例
    """
    if message_history:
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
        ])
    else:
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{input}"),
        ])


# =============================================================================
# 2. 简单 Chain - 最基础的链式结构
# =============================================================================
def create_simple_chain(
    llm: ChatOpenAI,
    system_prompt: str,
    output_parser: Optional[Runnable] = None
) -> Runnable:
    """
    创建简单链: prompt | llm | output_parser
    
    Args:
        llm: 语言模型
        system_prompt: 系统提示
        output_parser: 输出解析器
    
    Returns:
        可执行的 Chain
    """
    prompt = create_prompt(system_prompt)
    parser = output_parser or StrOutputParser()
    return prompt | llm | parser


# =============================================================================
# 3. RAG Chain - 检索增强生成
# =============================================================================
def create_rag_chain(
    llm: ChatOpenAI,
    retriever: Any,
    system_prompt: Optional[str] = None,
) -> Runnable:
    """
    创建 RAG 链: retriever -> format_docs -> prompt -> llm -> output
    
    Args:
        llm: 语言模型
        retriever: 检索器
        system_prompt: 系统提示（可选）
    
    Returns:
        RAG Chain
    """
    default_prompt = """Answer the question based only on the following context:
    
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(system_prompt or default_prompt)
    
    def format_docs(docs: List[Any]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
    
    return (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough()
        )
        | prompt
        | llm
        | StrOutputParser()
    )


# =============================================================================
# 4. Agent with Tools - 工具调用 Agent
# =============================================================================
def create_tool_agent(
    llm: ChatOpenAI,
    tools: List[Tool],
    system_prompt: Optional[str] = None,
    memory: Optional[ConversationBufferMemory] = None
) -> AgentExecutor:
    """
    创建带工具的 Agent
    
    Args:
        llm: 语言模型
        tools: 工具列表
        system_prompt: 系统提示
        memory: 对话记忆
    
    Returns:
        AgentExecutor
    """
    default_prompt = """You are a helpful assistant. Use the tools when needed."""
    
    if memory:
        prompt = create_prompt(
            system_prompt or default_prompt,
            message_history=True
        )
    else:
        prompt = create_prompt(system_prompt or default_prompt)
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# =============================================================================
# 5. 条件分支 - RunnableBranch
# =============================================================================
def create_branch_chain(
    condition: RunnableLambda,
    branches: Dict[Runnable, Runnable],
    default: Optional[Runnable] = None
) -> RunnableBranch:
    """
    创建条件分支链
    
    Args:
        condition: 条件判断
        branches: 分支映射 {runnable: next_runnable}
        default: 默认分支
    
    Returns:
        RunnableBranch
    """
    return RunnableBranch(*branches.items(), default or RunnablePassthrough())


# =============================================================================
# 6. 并行执行 - RunnableParallel
# =============================================================================
def create_parallel_chain(
    steps: Dict[str, Runnable]
) -> RunnableParallel:
    """
    创建并行执行链
    
    Args:
        steps: 步骤映射 {name: runnable}
    
    Returns:
        RunnableParallel
    """
    return RunnableParallel(steps)


# =============================================================================
# 7. 完整 Agent 示例 - 综合使用
# =============================================================================
class LangChainAgent:
    """
    通用 LangChain Agent 类
    支持：简单对话、RAG、工具调用、条件分支
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        retriever: Optional[Any] = None,
        use_memory: bool = False,
    ):
        self.llm = llm or create_llm()
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.tools = tools or []
        self.retriever = retriever
        self.use_memory = use_memory
        
        # 初始化记忆
        self.memory = None
        if use_memory:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        
        # 构建执行器
        self._build_executor()
    
    def _build_executor(self):
        """构建执行器"""
        if self.tools:
            # 带工具的 Agent
            self.executor = create_tool_agent(
                self.llm,
                self.tools,
                self.system_prompt,
                self.memory
            )
        elif self.retriever:
            # RAG Chain
            self.chain = create_rag_chain(
                self.llm,
                self.retriever,
                self.system_prompt
            )
        else:
            # 简单 Chain
            self.chain = create_simple_chain(
                self.llm,
                self.system_prompt
            )
    
    def invoke(self, input_text: str, **kwargs) -> str:
        """
        执行 Agent
        
        Args:
            input_text: 用户输入
            **kwargs: 其他参数
        
        Returns:
            Agent 响应
        """
        if self.tools:
            # Agent 模式
            if self.memory:
                result = self.executor.invoke(
                    {"input": input_text},
                    **kwargs
                )
                return result["output"]
            else:
                result = self.executor.invoke(
                    {"input": input_text},
                    **kwargs
                )
                return result["output"]
        else:
            # Chain 模式
            return self.chain.invoke(input_text, **kwargs)
    
    def stream(self, input_text: str, **kwargs):
        """流式输出"""
        if self.tools:
            return self.executor.stream({"input": input_text}, **kwargs)
        else:
            return self.chain.stream(input_text, **kwargs)


# =============================================================================
# 使用示例
# =============================================================================
if __name__ == "__main__":
    # 方式1: 简单对话
    agent = LangChainAgent(
        system_prompt="You are a helpful assistant."
    )
    result = agent.invoke("What is LangChain?")
    print(result)
    
    # 方式2: 带工具的 Agent
    # from langchain.tools import tool
    # 
    # @tool
    # def get_weather(city: str) -> str:
    #     return f"Weather in {city} is sunny."
    # 
    # agent = LangChainAgent(
    #     tools=[get_weather],
    #     system_prompt="You are a weather assistant."
    # )
    # result = agent.invoke("What's the weather in Beijing?")
    # print(result)
    
    # 方式3: RAG
    # from langchain.vectorstores import Chroma
    # from langchain.embeddings import OpenAIEmbeddings
    # 
    # vectorstore = Chroma.from_texts(...)
    # retriever = vectorstore.as_retriever()
    # 
    # agent = LangChainAgent(
    #     retriever=retriever,
    #     system_prompt="Answer based on the context."
    # )
    # result = agent.invoke("What is the main topic?")
    # print(result)
