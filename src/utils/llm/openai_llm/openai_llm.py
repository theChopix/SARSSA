import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from utils.llm.llm import ChatLLM

load_dotenv()


class OpenAIChatLLM(ChatLLM):
    def __init__(self, model: str, temperature: float, max_tokens: int = 5000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        api_key = os.getenv("OPENAI_API_KEY") or ""
        self.model_ = ChatOpenAI(
            openai_api_key=api_key,  # type: ignore[arg-type]
            model=self.model,  # type: ignore[call-arg]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def generate_response(self, prompt: str | list[tuple[str, str]]) -> str:
        message: AIMessage = self.model_.invoke(prompt)
        content = message.content
        if isinstance(content, str):
            return content
        return str(content)
