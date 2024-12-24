# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: MIT

import asyncio
import os
import pathlib
import sys
import time
import typing as t

import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp import MCPToolkit

# 检查.env文件是否存在
env_path = pathlib.Path(__file__).parent.parent / ".env"
if not env_path.exists():
    print("\n错误: 未找到 .env 文件!")
    print("请在项目根目录创建 .env 文件，并添加以下配置:")
    print("SILICONFLOW_API_KEY=你的API密钥")
    print("SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1")
    sys.exit(1)

load_dotenv()

# 检查必要的环境变量
if not os.getenv("SILICONFLOW_API_KEY"):
    print("\n错误: 环境变量 SILICONFLOW_API_KEY 未设置!")
    print("请在 .env 文件中添加你的 API 密钥")
    sys.exit(1)

if not os.getenv("SILICONFLOW_BASE_URL"):
    print("\n错误: 环境变量 SILICONFLOW_BASE_URL 未设置!")
    print("请在 .env 文件中添加 API 基础URL")
    sys.exit(1)


# 全局缓存变量
_model_cache = {"models": None, "last_update": None}


def get_available_models():
    # 缓存有效期（24小时）
    CACHE_DURATION = 24 * 60 * 60  # 秒

    current_time = time.time()

    # 如果缓存存在且未过期，直接返回缓存的结果
    if (
        _model_cache["models"] is not None
        and _model_cache["last_update"] is not None
        and current_time - _model_cache["last_update"] < CACHE_DURATION
    ):
        return _model_cache["models"]

    # 如果缓存为空或已过期，进行API请求
    url = "https://api.siliconflow.cn/v1/models"
    querystring = {"type": "text", "sub_type": "chat"}
    headers = {"Authorization": f"Bearer {os.getenv('SILICONFLOW_API_KEY')}"}

    try:
        response = requests.request("GET", url, headers=headers, params=querystring)
        response.raise_for_status()
        models = response.json()
        model_list = [model["id"] for model in models["data"]] if models.get("data") else []

        # 更新缓存
        _model_cache["models"] = model_list
        _model_cache["last_update"] = current_time

        return model_list
    except Exception as e:
        print(f"Error fetching models: {e}")
        # 只有在发生错误且有之前的缓存时才返回缓存
        if _model_cache["models"] is not None:
            print("Using cached model list due to API error")
            return _model_cache["models"]
        # 如果没有缓存且API请求失败，返回空列表
        return []


async def run(tools: list[BaseTool], prompt: str) -> str:
    # 打印可用工具
    print("Available tools:", [tool.name for tool in tools])

    available_models = get_available_models()
    print("Using model: Qwen/Qwen2.5-72B-Instruct")

    preferred_model = "Qwen/Qwen2.5-72B-Instruct"
    if preferred_model not in available_models:
        raise ValueError(f"Preferred model {preferred_model} is not available. Available models: {available_models}")

    model = ChatOpenAI(
        model=preferred_model,
        base_url=os.getenv("SILICONFLOW_BASE_URL"),
        api_key=os.getenv("SILICONFLOW_API_KEY"),
    )
    tools_map = {tool.name: tool for tool in tools}
    tools_model = model.bind_tools(tools)
    messages: list[BaseMessage] = [HumanMessage(prompt)]

    print("\nSending initial prompt to model...")
    ai_message = t.cast(AIMessage, await tools_model.ainvoke(messages))
    messages.append(ai_message)

    print("\nProcessing tool calls...")
    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            print(f"\nTool call: {tool_call}")  # 打印完整的工具调用信息
            tool_name = tool_call["name"].lower()
            selected_tool = tools_map[tool_name]

            # 解析工具调用参数
            tool_args = tool_call["args"]
            if isinstance(tool_args, str):
                import json

                tool_args = json.loads(tool_args)

            # 调用工具
            tool_msg = await selected_tool.ainvoke(tool_args)
            print(f"Tool response: {str(tool_msg)[:200]}...")
            messages.append(tool_msg)
    else:
        print("No tool calls in the response")

    print("\nGenerating final response...")
    return await (tools_model | StrOutputParser()).ainvoke(messages)


async def main(prompt: str) -> None:
    server_params = StdioServerParameters(
        command="/opt/miniconda3/bin/uv",
        args=[
            "--project",
            "/Users/lco/GitHub/MCP-searxng/",
            "run",
            "/Users/lco/GitHub/MCP-searxng/mcp-searxng/main.py",
        ],
        env={"SEARXNG_URL": "http://localhost:18080"},
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            toolkit = MCPToolkit(session=session)
            await toolkit.initialize()
            response = await run(toolkit.get_tools(), prompt)
            print(response)


if __name__ == "__main__":
    prompt = "请你运用搜索能力,请你介绍一下特朗普和马斯克成立的DOGE部门"
    asyncio.run(main(prompt))
