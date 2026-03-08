import os
from typing import Generator, Dict, Optional
from openai import OpenAI
from .base import BaseModel

class ZhipuModel(BaseModel):
    """智谱AI (GLM) 模型适配器，使用 OpenAI 兼容 API"""
    
    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

    def __init__(self, api_key: str, temperature: float = 0.7, system_prompt: str = None,
                 language: str = None, model_name: str = None, api_base_url: str = None):
        self.model_name = model_name or "glm-4v-plus"
        super().__init__(api_key, temperature, system_prompt, language)
        self.api_base_url = api_base_url or self.DEFAULT_BASE_URL

    def get_default_system_prompt(self) -> str:
        return """你是智谱AI助手，擅长文字理解、图像分析和逻辑推理。请根据用户提供的问题或图片：
1. 仔细阅读并理解问题
2. 分析问题的关键组成部分
3. 提供清晰的、逐步的解决方案
4. 如果相关，解释涉及的概念或理论
5. 如果有多种解决方法，先解释最高效的方法"""

    def get_model_identifier(self) -> str:
        return self.model_name

    def analyze_text(self, text: str, proxies: dict = None) -> Generator[dict, None, None]:
        """流式返回智谱模型的文本分析结果"""
        try:
            yield {"status": "started", "content": ""}

            original_env = {
                'http_proxy': os.environ.get('http_proxy'),
                'https_proxy': os.environ.get('https_proxy')
            }

            try:
                if proxies:
                    if 'http' in proxies:
                        os.environ['http_proxy'] = proxies['http']
                    if 'https' in proxies:
                        os.environ['https_proxy'] = proxies['https']

                client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ]

                response = client.chat.completions.create(
                    model=self.get_model_identifier(),
                    messages=messages,
                    temperature=self.temperature,
                    stream=True,
                    max_tokens=4000
                )

                response_buffer = ""
                for chunk in response:
                    if hasattr(chunk.choices[0].delta, 'content'):
                        content = chunk.choices[0].delta.content
                        if content:
                            response_buffer += content
                            if len(content) >= 10 or content.endswith(('.', '!', '?', '。', '！', '？', '\n')):
                                yield {"status": "streaming", "content": response_buffer}

                if response_buffer:
                    yield {"status": "streaming", "content": response_buffer}
                yield {"status": "completed", "content": response_buffer}

            finally:
                for key, value in original_env.items():
                    if value is None:
                        if key in os.environ:
                            del os.environ[key]
                    else:
                        os.environ[key] = value

        except Exception as e:
            yield {"status": "error", "error": str(e)}

    def analyze_image(self, image_data: str, proxies: dict = None) -> Generator[dict, None, None]:
        """流式返回智谱模型的图像分析结果"""
        try:
            yield {"status": "started", "content": ""}

            original_env = {
                'http_proxy': os.environ.get('http_proxy'),
                'https_proxy': os.environ.get('https_proxy')
            }

            try:
                if proxies:
                    if 'http' in proxies:
                        os.environ['http_proxy'] = proxies['http']
                    if 'https' in proxies:
                        os.environ['https_proxy'] = proxies['https']

                client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "请分析这张图片并提供详细的解答。"
                            }
                        ]
                    }
                ]

                response = client.chat.completions.create(
                    model=self.get_model_identifier(),
                    messages=messages,
                    temperature=self.temperature,
                    stream=True,
                    max_tokens=4000
                )

                response_buffer = ""
                for chunk in response:
                    if hasattr(chunk.choices[0].delta, 'content'):
                        content = chunk.choices[0].delta.content
                        if content:
                            response_buffer += content
                            if len(content) >= 10 or content.endswith(('.', '!', '?', '。', '！', '？', '\n')):
                                yield {"status": "streaming", "content": response_buffer}

                if response_buffer:
                    yield {"status": "streaming", "content": response_buffer}
                yield {"status": "completed", "content": response_buffer}

            finally:
                for key, value in original_env.items():
                    if value is None:
                        if key in os.environ:
                            del os.environ[key]
                    else:
                        os.environ[key] = value

        except Exception as e:
            yield {"status": "error", "error": str(e)}
