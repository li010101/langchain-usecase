# =============================================================================
# DeepAgent 模板 - 基于 LangChain DeepAgents 的高级 Agent
# 适用场景：复杂长任务、规划分解、子Agent委托、文件系统操作、持久化上下文
# =============================================================================

from typing import Any, Optional, List, Dict, Callable
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model

import os
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# 1. 安装与配置
# =============================================================================
"""
# 安装 DeepAgents
pip install deepagents

# 或使用 uv
uv add deepagents
"""


# =============================================================================
# 2. 快速开始 - 最简用法
# =============================================================================
def create_basic_deep_agent():
    """创建基础 Deep Agent"""
    from deepagents import create_deep_agent
    
    agent = create_deep_agent()
    return agent


# =============================================================================
# 3. 完整配置 - DeepAgent 完整选项
# =============================================================================
class DeepAgentConfig(BaseModel):
    """DeepAgent 配置类"""
    model: str = "openai:gpt-4o"  # 模型选择
    tools: Optional[List[Any]] = None  # 自定义工具
    system_prompt: Optional[str] = None  # 系统提示
    max_steps: int = 100  # 最大步数
    max_iterations: int = 10  # 最大迭代次数
    verbose: bool = True  # 详细输出


def create_configured_deep_agent(config: DeepAgentConfig):
    """创建配置化的 Deep Agent"""
    from deepagents import create_deep_agent
    
    agent = create_deep_agent(
        model=init_chat_model(model=config.model),
        tools=config.tools,
        system_prompt=config.system_prompt,
        max_steps=config.max_steps,
        max_iterations=config.max_iterations,
    )
    return agent


# =============================================================================
# 4. 子 Agent 配置 - 委托任务
# =============================================================================
def create_subagent(
    name: str,
    description: str,
    system_prompt: str,
    tools: Optional[List[Any]] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    创建子 Agent 配置
    
    Args:
        name: 子 Agent 名称
        description: 用途描述
        system_prompt: 系统提示
        tools: 工具列表
        model: 指定模型（可选）
    
    Returns:
        子 Agent 配置字典
    """
    subagent = {
        "name": name,
        "description": description,
        "system_prompt": system_prompt,
        "tools": tools or [],
    }
    if model:
        subagent["model"] = model
    return subagent


# =============================================================================
# 5. Skills 配置 - 渐进式披露
# =============================================================================
def create_skill_config(
    skill_path: str,
    description: str = None
) -> Dict[str, str]:
    """
    创建 Skill 配置
    
    Args:
        skill_path: .deepagents/skills/ 下的路径
        description: 描述
    
    Returns:
        Skill 配置
    """
    return {
        "path": skill_path,
        "description": description or ""
    }


# =============================================================================
# 6. Backend 配置 - 持久化存储
# =============================================================================
class BackendConfig:
    """Backend 配置工厂"""
    
    @staticmethod
    def memory():
        """内存存储（默认）"""
        from deepagents.backends import MemoryBackend
        return MemoryBackend()
    
    @staticmethod
    def filesystem(root_dir: str = "."):
        """文件系统存储"""
        from deepagents.backends import FilesystemBackend
        return FilesystemBackend(root_dir=root_dir)
    
    @staticmethod
    def state(thread_id: str):
        """状态存储"""
        from deepagents.backends import StateBackend
        return StateBackend(thread_id=thread_id)
    
    @staticmethod
    def store(namespace: str):
        """Store 存储（持久化）"""
        from deepagents.backends import StoreBackend
        return StoreBackend(namespace=namespace)


# =============================================================================
# 7. Middleware - 中间件扩展
# =============================================================================
class DeepAgentMiddleware:
    """Middleware 基类"""
    
    @staticmethod
    def create_logging_middleware():
        """日志中间件"""
        from deepagents.middleware import LoggingMiddleware
        return LoggingMiddleware()
    
    @staticmethod
    def create_token_counter_middleware():
        """Token 计数中间件"""
        from deepagents.middleware import TokenCounterMiddleware
        return TokenCounterMiddleware()


# =============================================================================
# 8. 工作流编排 - 多 Agent 系统
# =============================================================================
class DeepAgentWorkflow:
    """
    DeepAgent 工作流编排器
    支持：主 Agent + 子 Agent 协同
    """
    
    def __init__(self, config: DeepAgentConfig):
        from deepagents import create_deep_agent
        
        self.config = config
        self.subagents = []
        
        # 创建主 Agent
        self.main_agent = create_deep_agent(
            model=init_chat_model(model=config.model),
            tools=config.tools,
            system_prompt=config.system_prompt,
        )
    
    def add_subagent(self, subagent_config: Dict[str, Any]):
        """添加子 Agent"""
        self.subagents.append(subagent_config)
    
    def create_supervisor_workflow(self) -> Any:
        """
        创建监督者工作流（Supervisor Pattern）
        主 Agent 协调多个子 Agent
        """
        from deepagents import create_deep_agent
        
        # 构建子 Agent 工具
        subagent_tools = []
        for sa in self.subagents:
            # 将子 Agent 作为工具注册
            tool = {
                "name": sa["name"],
                "description": sa["description"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "任务描述"}
                    },
                    "required": ["task"]
                }
            }
            subagent_tools.append(tool)
        
        # 创建监督者 Agent
        supervisor_system_prompt = f"""
        You are a supervisor coordinating multiple specialized agents.
        
        Available subagents:
        {', '.join([sa['name'] for sa in self.subagents])}
        
        Delegate tasks appropriately and synthesize results.
        """
        
        return create_deep_agent(
            model=init_chat_model(model=self.config.model),
            tools=subagent_tools,
            system_prompt=supervisor_system_prompt,
        )
    
    def create_parallel_workflow(self) -> Any:
        """
        创建并行工作流（Parallel Pattern）
        多个 Agent 并行处理，独立汇总
        """
        # 并行工作流通常需要自定义实现
        # 这里返回各个子 Agent 的配置
        return self.subagents


# =============================================================================
# 9. 完整示例：研究 Agent 系统
# =============================================================================
class ResearchAgentSystem:
    """
    研究 Agent 系统示例
    包含：主 Agent、搜索 Agent、写作 Agent、审查 Agent
    """
    
    def __init__(self, model: str = "openai:gpt-4o"):
        self.model = model
        self._build_agents()
    
    def _build_agents(self):
        """构建 Agent 系统"""
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        
        # 子 Agent 1: 搜索 Agent
        search_subagent = {
            "name": "searcher",
            "description": "Research agent for searching information",
            "system_prompt": "You are a research assistant. Search for relevant information.",
            "tools": [],  # 添加搜索工具
        }
        
        # 子 Agent 2: 写作 Agent
        writer_subagent = {
            "name": "writer", 
            "description": "Agent for writing content",
            "system_prompt": "You are a professional writer. Create well-structured content.",
            "tools": [],  # 添加写作工具
        }
        
        # 子 Agent 3: 审查 Agent
        review_subagent = {
            "name": "reviewer",
            "description": "Agent for reviewing and improving content",
            "system_prompt": "You are a critical reviewer. Provide constructive feedback.",
            "tools": [],
        }
        
        # 主 Agent（协调者）
        self.main_agent = create_deep_agent(
            model=init_chat_model(model=self.model),
            subagents=[search_subagent, writer_subagent, review_subagent],
            backend=FilesystemBackend(root_dir="./workspace"),
            system_prompt="""
            You are a research project manager.
            
            Coordinate the following subagents:
            - searcher: Find relevant information
            - writer: Write content based on findings
            - reviewer: Review and improve content
            
            Plan tasks, delegate to appropriate agents, and synthesize results.
            """
        )
    
    def research(self, topic: str, **kwargs) -> Dict[str, Any]:
        """
        执行研究任务
        
        Args:
            topic: 研究主题
            **kwargs: 其他参数
        
        Returns:
            研究结果
        """
        result = self.main_agent.invoke({
            "messages": [HumanMessage(content=topic)]
        }, **kwargs)
        
        return {
            "status": "success",
            "result": result,
            "topic": topic
        }


# =============================================================================
# 10. 常用工具模板
# =============================================================================
class DeepAgentTools:
    """常用工具工厂"""
    
    @staticmethod
    def create_search_tool(search_func: Callable[[str], str]):
        """创建搜索工具"""
        from langchain.tools import Tool
        return Tool(
            name="search",
            description="Search for information on the web",
            func=search_func
        )
    
    @staticmethod
    def create_read_tool():
        """创建读取工具"""
        from deepagents.tools import read_file
        return read_file
    
    @staticmethod
    def create_write_tool():
        """创建写入工具"""
        from deepagents.tools import write_file
        return write_file
    
    @staticmethod
    def create_edit_tool():
        """创建编辑工具"""
        from deepagents.tools import edit_file
        return edit_file


# =============================================================================
# 使用示例
# =============================================================================
if __name__ == "__main__":
    # 方式1: 最简用法
    # agent = create_basic_deep_agent()
    # result = agent.invoke({"messages": [{"role": "user", "content": "Write a summary"}]})
    
    # 方式2: 自定义配置
    # config = DeepAgentConfig(
    #     model="anthropic:claude-sonnet-4-5-20250929",
    #     system_prompt="You are a helpful assistant.",
    #     verbose=True
    # )
    # agent = create_configured_deep_agent(config)
    
    # 方式3: 研究 Agent 系统
    # research_system = ResearchAgentSystem()
    # result = research_system.research("What are the latest AI trends?")
    
    # 方式4: 带 Skills
    # from deepagents import create_deep_agent
    # agent = create_deep_agent(
    #     skills=[".deepagents/skills/deploy"]
    # )
    
    print("DeepAgent template ready!")
    print("See examples above for usage.")
