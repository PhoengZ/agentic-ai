from typing import TypedDict, List

# for now we are not using reviewer agent so answer field is enough to work

class GraphState(TypedDict):
    question: str
    documents: List[str]
    answer: str