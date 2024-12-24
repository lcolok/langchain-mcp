import time

import requests


class ModelManager:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._cache: dict = {"models": None, "last_update": None}
        self.CACHE_DURATION = 24 * 60 * 60  # 24小时缓存

    def get_available_models(self) -> list[str]:
        current_time = time.time()

        # 如果缓存存在且未过期，直接返回缓存的结果
        if (
            self._cache["models"] is not None
            and self._cache["last_update"] is not None
            and current_time - self._cache["last_update"] < self.CACHE_DURATION
        ):
            return self._cache["models"]

        # 如果缓存为空或已过期，进行API请求
        url = "https://api.siliconflow.cn/v1/models"
        querystring = {"type": "text", "sub_type": "chat"}
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            response = requests.request("GET", url, headers=headers, params=querystring)
            response.raise_for_status()
            models = response.json()
            model_list = [model["id"] for model in models["data"]] if models.get("data") else []

            # 更新缓存
            self._cache["models"] = model_list
            self._cache["last_update"] = current_time

            return model_list
        except Exception as e:
            print(f"Error fetching models: {e}")
            # 只有在发生错误且有之前的缓存时才返回缓存
            if self._cache["models"] is not None:
                print("Using cached model list due to API error")
                return self._cache["models"]
            # 如果没有缓存且API请求失败，返回空列表
            return []

    def verify_model(self, model_name: str) -> bool:
        available_models = self.get_available_models()
        return model_name in available_models
