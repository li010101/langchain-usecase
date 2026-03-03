# =============================================================================
# LangGraph Agent 模板 - 基于图的多 Agent 架构
# 适用场景：复杂任务、需要状态机、人类介入、多轮对话
# =============================================================================

from typing import Annotated, TypedDict, Optional, Literal
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import tools_condition
from langgraph.checkpoint.memory import MemorySaver

import os
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# 1. 状态定义 - 扩展点：根据业务需求添加字段
# =============================================================================
class AgentState(TypedDict):
    """Agent 状态：所有需要持久化的数据"""
    messages: Annotated[list[AnyMessage], add_messages]  # 对话历史
    dialog_state: Annotated[list[str], update_dialog_stack]  # 当前阶段
    # === 扩展字段示例 ===
    # user_profile: Optional[dict]  # 用户画像
    # task_result: Optional[str]   # 任务结果
    # context: Optional[dict]       # 额外上下文


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """更新对话栈：push/pop 操作"""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


# =============================================================================
# 2. 工具定义 - 扩展点：添加业务相关的 @tool 函数
# =============================================================================
from langchain.agents import tool


@tool
def example_tool(param: str) -> str:
    """
    工具描述：告诉 LLM 什么时候调用这个工具
    
    Args:
        param: 参数描述
    
    Returns:
        工具执行结果
    """
    # 业务逻辑实现
    return f"Result for {param}"


# 更多工具...
# @tool
# def search_tool(query: str) -> str: ...


# =============================================================================
# 3. 路由器 Pydantic 模型 - 扩展点：添加新的 Agent 触发器
# =============================================================================
class ToSubAgentA(BaseModel):
    """触发子 Agent A"""
    param: str = Field(description="参数描述")


class ToSubAgentB(BaseModel):
    """触发子 Agent B"""
    param: str = Field(description="参数描述")


# =============================================================================
# 4. Agent 工厂函数 - 扩展点：创建不同的专业化 Agent
# =============================================================================
class Assistant:
    """通用 Agent 包装器"""
    
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: AgentState, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


def create_specialized_agent(
    llm: Runnable,
    system_prompt: str,
    tools: list,
    agent_name: str = "Assistant"
) -> Assistant:
    """
    创建专业化 Agent
    
    Args:
        llm: 语言模型
        system_prompt: 系统提示词
        tools: 工具列表
        agent_name: Agent 名称
    
    Returns:
        Assistant 实例
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])
    runnable = prompt | llm.bind_tools(tools)
    return Assistant(runnable)


# =============================================================================
# 5. 路由函数 - 扩展点：根据业务逻辑修改流向
# =============================================================================
def create_entry_node(agent_name: str, dialog_state: str):
    """创建入口节点：切换到子 Agent 时执行"""
    def entry_node(state: AgentState) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now {agent_name}. "
                            f"Use the provided tools to assist the user.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": dialog_state,
        }
    return entry_node


def route_primary_assistant(state: AgentState) -> Literal["enter_agent_a", "enter_agent_b", END]:
    """主 Agent 路由：判断用户意图，分发给对应子 Agent"""
    route = tools_condition(state)
    if route == END:
        return END
    
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        tool_name = tool_calls[0]["name"]
        if tool_name == ToSubAgentA.__name__:
            return "enter_agent_a"
        elif tool_name == ToSubAgentB.__name__:
            return "enter_agent_b"
    raise ValueError("Invalid route")


def should_continue(state: AgentState) -> Literal["tools", END]:
    """子 Agent 内部路由：判断是否继续调用工具"""
    route = tools_condition(state)
    if route == END:
        return END
    
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        return "tools"
    return END


# =============================================================================
# 6. 提示词模板 - 扩展点：定义不同 Agent 的行为
# =============================================================================
BASE_TEMPLATE = """You are a helpful assistant."""


# =============================================================================
# 7. 图构建 - 核心编排逻辑
# =============================================================================
def create_agent_graph(llm: Runnable) -> StateGraph:
    """
    构建 LangGraph 状态图
    
    Args:
        llm: 语言模型实例
    
    Returns:
        编译后的可执行图对象
    """
    builder = StateGraph(AgentState)
    
    # === 主 Agent ===
    builder.add_node("primary_assistant", create_specialized_agent(
        llm, BASE_TEMPLATE, [ToSubAgentA, ToSubAgentB]
    ))
    builder.add_conditional_edges(
        "primary_assistant",
        route_primary_assistant,
        {
            "enter_agent_a": "enter_agent_a",
            "enter_agent_b": "enter_agent_b",
            END: END,
        }
    )
    
    # === 子 Agent A ===
    builder.add_node("enter_agent_a", create_entry_node("Agent A", "agent_a"))
    builder.add_node("agent_a", create_specialized_agent(
        llm, "You are Agent A.", [example_tool]
    ))
    builder.add_edge("enter_agent_a", "agent_a")
    builder.add_node("agent_a_tools", create_tool_node_with_fallback([example_tool]))
    builder.add_edge("agent_a_tools", "agent_a")
    builder.add_conditional_edges("agent_a", should_continue)
    
    # === 子 Agent B (同上...) ===
    # builder.add_node("enter_agent_b", ...)
    # builder.add_node("agent_b", ...)
    
    builder.set_entry_point("primary_assistant")
    
    # 持久化支持
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# 8. 入口函数
# =============================================================================
def initialize_agent():
    """初始化 Agent"""
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    graph = create_agent_graph(llm)
    return graph


if __name__ == "__main__":
    # 使用示例
    graph = initialize_agent()
    
    # 同步调用
    result = graph.invoke({"messages": [("user", "Hello")])
    print(result["messages"][-1].content)
    
    # 流式调用
    for chunk in graph.stream({"messages": [("user", "Hello")]):
        print(chunk)
