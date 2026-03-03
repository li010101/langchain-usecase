"""
FastAPI 服务器模块
该模块使用 LangServe 将 LangGraph 构建的金融分析 Agent 部署为 Web API。
"""

from typing import List, Any, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langserve import add_routes

from app.chains.agent import create_agent_graph

# 初始化 FastAPI 应用
app = FastAPI(
    title="Financial Chat",
    version="1.0",
    description="LangChain 多 Agent 股票分析演示系统",
)

# 配置 CORS 中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 创建 LangGraph 编译后的图对象
graph = create_agent_graph()

# 定义 API 输入模型
class AgentInput(BaseModel):
    """
    Agent 的输入结构
    messages: 包含 HumanMessage 或 AIMessage 的列表，用于表示对话历史
    """
    messages: List[Union[HumanMessage, AIMessage]] = Field(
        ...,
        description="代表当前对话的聊天消息列表。",
        extra={"widget": {"type": "chat", "input": "messages"}},
    )

# 定义 API 输出模型
class AgentOutput(BaseModel):
    """Agent 的输出结构"""
    output: Any

@app.get("/")
async def redirect_root_to_docs():
    """将根路径重定向到 API 文档页"""
    return RedirectResponse("/docs")

@app.get("/health")
async def health():
    """健康检查接口"""
    return {"status": "ok"}

# 使用 LangServe 添加路由，将 LangGraph 图暴露为 /chat 接口
add_routes(
    app,
    graph,
    path="/chat",
    input_type=AgentInput,
    output_type=AgentOutput,
)

if __name__ == "__main__":
    import uvicorn

    # 启动 uvicorn 服务器
    uvicorn.run(app, host="0.0.0.0", port=8080)
