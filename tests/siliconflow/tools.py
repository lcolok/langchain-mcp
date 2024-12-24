import json
import typing as t

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """你是一个强大的AI助手，能够使用搜索工具来回答问题。

重要提示：
1. 工具调用时，参数必须是严格的JSON格式
2. 每次只能搜索一个简单的查询词
3. 避免在搜索词中使用特殊字符

搜索策略：
1. 使用最简单、最关键的词组
2. 一次只搜索一个信息点
3. 如果搜索结果不理想，尝试更简单的关键词

例如，如果用户问"演员在电影中多大年纪"，你应该：
1. 第一次搜索：只用"演员名字"
2. 第二次搜索：只用"电影名字"
3. 根据搜索结果提取信息，如果信息不足，再用更具体的词搜索

请记住：
1. 保持搜索词简单
2. 避免复杂的查询组合
3. 确保每次工具调用的参数都是有效的JSON"""


def parse_tool_args(args: str | dict) -> dict:
    """解析工具调用参数"""
    if isinstance(args, dict):
        return args

    if isinstance(args, str):
        try:
            # 尝试直接解析
            return json.loads(args)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            try:
                # 如果是字符串，直接构造查询
                return {"query": args.strip("\"'")}
            except Exception as e2:
                print(f"Error creating query: {e2}")
                return {"query": "error"}

    return {"query": str(args)}


async def run_with_tools(tools: list[BaseTool], prompt: str, model_name: str, base_url: str, api_key: str) -> str:
    print("Available tools:", [tool.name for tool in tools])
    print(f"Using model: {model_name}")

    model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=0.3,  # 降低温度，使输出更加稳定
    )

    tools_map = {tool.name: tool for tool in tools}
    tools_model = model.bind_tools(tools)

    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

    error_count = 0
    max_errors = 3
    max_iterations = 10
    iteration = 0
    search_results = []

    while iteration < max_iterations:
        iteration += 1
        print(f"\nIteration {iteration}")

        try:
            ai_message = t.cast(AIMessage, await tools_model.ainvoke(messages))

            # 检查是否有工具调用
            if not (hasattr(ai_message, "tool_calls") and ai_message.tool_calls):
                if ai_message.content:
                    # 如果有搜索结果且回答看起来合理，就结束
                    if search_results and "抱歉" not in ai_message.content.lower():
                        return ai_message.content
                    # 如果没有搜索结果或回答不确定，继续尝试
                    messages.append(HumanMessage(content="请使用更简单的关键词重新搜索。每次只搜索一个简单的词组。"))
                    continue
                break

            for tool_call in ai_message.tool_calls:
                tool_name = tool_call["name"].lower()
                selected_tool = tools_map[tool_name]

                # 解析工具调用参数
                tool_args = parse_tool_args(tool_call["args"])
                print(f"Parsed tool args: {tool_args}")

                try:
                    # 调用工具
                    tool_msg = await selected_tool.ainvoke(tool_args)
                    print(f"Tool response: {str(tool_msg)[:200]}...")

                    # 保存搜索结果
                    search_results.append(str(tool_msg))

                    messages.append(
                        HumanMessage(
                            content=f"""搜索结果: {str(tool_msg)}\n\n请分析这个结果，
                            如果需要更多信息，使用更简单的关键词继续搜索。"""
                        )
                    )
                except Exception as e:
                    print(f"Error invoking tool: {e}")
                    error_count += 1
                    if error_count >= max_errors:
                        if search_results:
                            return "虽然遇到了一些错误，但根据已有信息：\n" + "\n".join(search_results)
                        return "抱歉，在搜索过程中遇到了太多错误。请稍后再试。"
                    messages.append(HumanMessage(content="请使用更简单的关键词重试。"))

        except Exception as e:
            print(f"Error in conversation loop: {e}")
            error_count += 1
            if error_count >= max_errors:
                if search_results:
                    return "虽然遇到了一些错误，但根据已有信息：\n" + "\n".join(search_results)
                return "抱歉，在处理过程中遇到了错误。请稍后再试。"
            messages.append(HumanMessage(content="请使用更简单的搜索词重试。"))
            continue

    # 如果有搜索结果，生成最终总结
    if search_results:
        try:
            final_message = t.cast(
                AIMessage,
                await tools_model.ainvoke(
                    messages + [HumanMessage(content="请根据所有搜索结果，给出最终答案。如果信息不足，请说明。")]
                ),
            )
            return final_message.content
        except Exception as e:
            print(f"Error generating final summary: {e}")
            return "根据搜索结果：\n" + "\n".join(search_results)

    return "抱歉，未能找到相关信息。请换个方式提问。"
