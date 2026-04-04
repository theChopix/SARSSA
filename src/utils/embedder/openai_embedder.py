import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from .embedder import EmbeddingLLM

load_dotenv()


class OpenAIEmbeddingLLM(EmbeddingLLM):
    def __init__(self, model: str):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY") or ""
        self.model_ = OpenAIEmbeddings(openai_api_key=api_key, model=self.model)  # type: ignore[arg-type]

    def generate_embedding(self, text: str) -> list[float]:
        return self.model_.embed_query(text)
