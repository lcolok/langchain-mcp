import asyncio
import sys
from pathlib import Path

# 添加父目录到 Python 路径
sys.path.append(str(Path(__file__).parent))

from config import load_config
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from models import ModelManager
from tools import run_with_tools

from langchain_mcp import MCPToolkit

DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"


async def main(prompt: str) -> None:
    # 加载配置
    config = load_config()

    # 初始化模型管理器
    model_manager = ModelManager(config["api_key"])

    # 验证默认模型是否可用
    if not model_manager.verify_model(DEFAULT_MODEL):
        raise ValueError(
            f"Preferred model {DEFAULT_MODEL} is not available. "
            f"Available models: {model_manager.get_available_models()}"
        )

    # 设置MCP服务器参数
    server_params = StdioServerParameters(
        command="/opt/miniconda3/bin/uv",
        args=[
            "--project",
            "/Users/lco/GitHub/MCP-searxng/",
            "run",
            "/Users/lco/GitHub/MCP-searxng/mcp-searxng/main.py",
        ],
        env={"SEARXNG_URL": "http://localhost:18081"},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            toolkit = MCPToolkit(session=session)
            await toolkit.initialize()

            response = await run_with_tools(
                tools=toolkit.get_tools(),
                prompt=prompt,
                model_name=DEFAULT_MODEL,
                base_url=config["base_url"],
                api_key=config["api_key"],
            )
            print(response)


if __name__ == "__main__":
    prompt = """
    请你运用搜索能力, 查看高桥一生演出侧耳倾听的时候多少岁？
    首先你应该搜索高桥一生出生年份，然后搜索侧耳倾听发布年份
    """.strip()
    asyncio.run(main(prompt))
