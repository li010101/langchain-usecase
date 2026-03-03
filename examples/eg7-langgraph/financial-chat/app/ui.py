"""
Streamlit UI 模块
该模块提供了一个基于 Streamlit 的 Web 界面，用于交互式展示金融分析 Agent。
"""

import sys
import os
# 将父目录添加到系统路径，以支持导入 app 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Callable, TypeVar

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.delta_generator import DeltaGenerator
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.runnables import RunnableConfig

from app.chains.clear_results import with_clear_container
from app.chains.agent import create_agent_graph

import warnings
import inspect
import uuid
import streamlit as st

# 忽略不必要的警告
warnings.filterwarnings("ignore")

# Streamlit 页面配置
st.set_page_config(
    page_title="Financial Chat | LangChain 多 Agent 演示",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

T = TypeVar("T")

def get_streamlit_cb(parent_container: DeltaGenerator):
    """
    配置并返回 StreamlitCallbackHandler，用于在 UI 中实时显示 LangChain 的运行步骤。
    为了在多线程/异步环境（如 LangGraph 内部）中正确使用 Streamlit 上下文，
    需要使用装饰器对回调方法进行处理。
    """
    def decor(fn: Callable[..., T]) -> Callable[..., T]:
        # 获取当前的 Streamlit 脚本运行上下文
        ctx = get_script_run_ctx()

        def wrapper(*args, **kwargs) -> T:
            # 在执行具体回调前重新附加上下文
            add_script_run_ctx(ctx=ctx)
            return fn(*args, **kwargs)

        return wrapper

    # 创建基础的回调处理器
    st_cb = StreamlitCallbackHandler(parent_container=parent_container)

    # 遍历所有回调方法（以 on_ 开头的方法）并注入装饰器
    for name, fn in inspect.getmembers(st_cb, predicate=inspect.ismethod):
        if name.startswith("on_"):
            setattr(st_cb, name, decor(fn))

    return st_cb

# 在会话状态中初始化 LangGraph 对象
if "graph" not in st.session_state:
    st.session_state.graph = create_agent_graph()

st.title("金融分析助手 - 多 Agent 协作演示 📈")

# 初始化消息历史
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "您好！我是您的 AI 金融顾问，可以帮您分析股票。目前支持 AAPL, GOOGL, MSFT, AMZN, TSLA 等。"}
    ]

# 在 UI 中显示历史消息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 用户输入表单
with st.form(key="form"):
    user_input = st.text_input(
        "请在此输入您的问题：",
        placeholder="请对 AAPL 进行全面分析",
    )
    submit_clicked = st.form_submit_button("发送")

# 处理用户提交
output_container = st.empty()
if with_clear_container(submit_clicked):
    output_container = output_container.container()

    # 显示用户的提问
    output_container.markdown(f"**用户:** {user_input}")
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 创建助手回复的占位容器
    answer_container = output_container.chat_message("assistant", avatar="💸")
    # 获取 Streamlit 回调处理器以展示 Agent 执行细节
    st_callback = get_streamlit_cb(answer_container)

    # 配置 Runnable，设置递归限制和回调
    cfg = RunnableConfig(recursion_limit=150)
    cfg["callbacks"] = [st_callback]
    cfg["configurable"] = {"thread_id": uuid.uuid4()}

    # 准备图输入（LangGraph 期待字典形式的状态）
    question = {"messages": [("user", user_input)]}

    # 调用 Agent 图执行逻辑
    response = st.session_state.graph.invoke(question, cfg)
    # 获取最后一条助手消息作为回复内容
    answer = response["messages"][-1].content

    # 保存回复到会话历史并更新 UI
    st.session_state.messages.append({"role": "assistant", "content": answer})
    answer_container.write(answer)
