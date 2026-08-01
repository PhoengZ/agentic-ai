from typing import Literal
from pydantic import Field, BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

class Supervised_response(BaseModel):
    next_agent: Literal["generator_agent", "retriever_agent", "END"] = Field(
        description= "To choose which agent to act next or END if the task is finished."
    )  

llm  = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite', temperature=0.3)

# add node and edges below