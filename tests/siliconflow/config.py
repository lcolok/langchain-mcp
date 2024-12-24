import os
import pathlib
import sys

from dotenv import load_dotenv


def load_config():
    # 检查.env文件是否存在
    env_path = pathlib.Path(__file__).parent.parent.parent / ".env"
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

    return {"api_key": os.getenv("SILICONFLOW_API_KEY"), "base_url": os.getenv("SILICONFLOW_BASE_URL")}
