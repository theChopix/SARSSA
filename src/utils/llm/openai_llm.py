from typing import List, Tuple
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

from .llm import ChatLLM

load_dotenv()


class OpenAIChatLLM(ChatLLM):
    def __init__(self, model: str, temperature: float, max_tokens: int = 5000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_ = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                                model=self.model,
                                temperature=self.temperature,
                                max_tokens=self.max_tokens
                        )
        
    def generate_response(self, prompt: str | List[Tuple[str, str]]) -> str:
        message: AIMessage = self.model_.invoke(prompt)
        return message.content