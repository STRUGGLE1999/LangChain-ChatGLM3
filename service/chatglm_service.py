#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
ChatGLM service
"""

from typing import Dict, Union, Optional
from typing import List

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoModel, AutoTokenizer


class ChatGLMService(LLM):
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        response, history = self.model.chat(
            self.tokenizer,
            prompt,
            history=self.history,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        
        self.history = self.history + [history[-1]]#将问题和答案传到历史记录中
        return response

    def load_model(self,
                   model_name_or_path: str = "chatglm3-6b"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()

