# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: MIT

import asyncio
import json
import pathlib
import sys
import typing as t

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp import MCPToolkit


async def run(tools: list[BaseTool], prompt: str) -> str:
    try:
        print("Starting chat with Groq...")
        model = ChatGroq(model="llama-3.3-70b-versatile", stop_sequences=None)  # requires GROQ_API_KEY
        tools_map = {tool.name: tool for tool in tools}
        tools_model = model.bind_tools(tools)
        messages: list[BaseMessage] = [HumanMessage(prompt)]
        ai_message = t.cast(AIMessage, await tools_model.ainvoke(messages))
        messages.append(ai_message)
        print(f"Received response: {ai_message.content}")

        # Process tool calls
        for tool_call in ai_message.tool_calls:
            print(f"Executing tool: {tool_call['name']}")
            print(f"Tool arguments: {tool_call['arguments']}")
            selected_tool = tools_map[tool_call["name"].lower()]
            # Parse arguments as JSON
            args = json.loads(tool_call["arguments"])
            tool_msg = await selected_tool.ainvoke(**args)
            print(f"Tool result: {tool_msg.content}")
            messages.append(tool_msg)

        final_response = await tools_model.ainvoke(messages)
        return StrOutputParser().invoke(final_response)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


async def main(prompt: str) -> None:
    # Load environment variables from .env file
    load_dotenv()
    print("Environment variables loaded")

    server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            str(pathlib.Path(__file__).parent.parent),
        ],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            toolkit = MCPToolkit(session=session)
            await toolkit.initialize()
            response = await run(toolkit.get_tools(), prompt)
            print(response)


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Read and summarize the file ./LICENSE"
    asyncio.run(main(prompt))
