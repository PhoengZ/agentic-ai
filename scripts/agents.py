from state_graph import GraphState
from config import llm, Supervised_response
from langchain_core.prompts import ChatPromptTemplate
def supervisor_agent_node(gs: GraphState):
   prompt =  ChatPromptTemplate.from_messages(
    [
        ("system", """You are a Supervisor orchestrating two workers: 'retriever_agent' and 'generator_agent'.
                   1. If we don't have retrieved documents yet, call 'retriever_agent'.
                   2. If we have documents but the question isn't answer, call 'generator_agent'.
                   3. If the question is answered, output 'END' to terminate the workflow."""),
        ("user", "Question: {question}\n Document Retrieve: {docs}")
    ]
   )
   chain = prompt | llm.with_structured_output(Supervised_response)
   result = chain.invoke({
    "question": gs['question'],
    "docs": True if gs.get('documents') else False,
    "answer": True if gs.get('answer') else False
   })
   return {}

