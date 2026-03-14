import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from .embedder import EmbeddingLLM

load_dotenv()
    

class OpenAIEmbeddingLLM(EmbeddingLLM):
    def __init__(self, model: str):
        self.model = model
        self.model_ = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"),
                                      model=self.model)
        
    def generate_embedding(self, text: str) -> list[float]:
        return self.model_.embed_query(text)
    