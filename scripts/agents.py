from scripts.state_graph import GraphState
from scripts.config import llm, Supervised_response
from scripts.tools import search_knowledge
from langchain_core.prompts import ChatPromptTemplate
def supervisor_agent_node(gs: GraphState):
   prompt =  ChatPromptTemplate.from_messages(
    [
        ("system", """You are a Supervisor orchestrating two workers: 'retriever_agent' and 'generator_agent'.
                   1. If we don't have retrieved documents yet, call 'retriever_agent'.
                   2. If we have documents but the question isn't answer, call 'generator_agent'.
                   3. If the answer is ready, output 'END' to terminate the workflow."""),
        ("user", "Question: {question}\n Document Retrieve: {docs}\nAnswer Generated: {answer}")
    ]
   )
   chain = prompt | llm.with_structured_output(Supervised_response)
   docs_text = "\n\n".join(gs.get('documents', [])) if gs.get('documents') else "No documents retrieved."
   result = chain.invoke({
    "question": gs['question'],
    "docs": docs_text,
    "answer": gs.get('answer') or "No answer generated yet."
   })
   return {"next_node": result.next_worker}

def retrieve_agent_node(gs: GraphState):
    llm_tools = llm.bind_tools([search_knowledge])
    result = llm_tools.invoke(f"Understand the {gs['question']} then find context from the question. Please use the 'search_knowledge' tool. Adjust the 'k' parameter based on how complex the question is.")

    documents = []
    if result.tool_calls:
        for idx, call in enumerate(result.tool_calls):
            # print(f"round: {idx+1}")
            # print(f"callargs" , call['args'])
            documents.append(search_knowledge.invoke(call['args']))

    return {"documents": documents}

def generator_agent_node(gs: GraphState):
    prompt = ChatPromptTemplate.from_messages([
        ("system","""
            **Role and Objective:**
            You are a highly accurate and professional Corporate Policy Assistant for Company A. Your primary objective is to answer employee questions and summarize information based STRICTLY on the retrieved context provided by the Retriever Agent. 

            **Instructions and Constraints:**
            1. **Strict Context Reliance:** You must base your answer ONLY on the provided context. Do NOT use outside knowledge, assume any unstated policies, or hallucinate information. 
            2. **Handling Missing Information:** If the provided context does not contain the information needed to answer the user's query, you must state clearly: "I cannot find the answer to this question in the provided company policy." Do not attempt to guess or provide a general industry standard.
            3. **Conciseness and Clarity:** Summarize the information clearly, concisely, and professionally. Use bullet points or numbered lists where appropriate to make the policy easy to read.
            4. **Tone:** Maintain a polite, professional, and objective tone at all times. 
            5. **No Legal Advice:** You are an AI assistant providing summaries of internal documents. Add a brief disclaimer if the question involves severe legal or disciplinary actions, reminding the user to consult Human Resources (HR) for official guidance.

            **Input Format:**
            - User Question: [The question or request from the user]
            - Retrieved Contexts: [The snippets of the company policy document provided by the retriever]

            **Output Generation:**
            Analyze the User Question, scan the Retrieved Context for relevant facts, and generate your response following the constraints above.
        """
        ),
        ("user", "Question: {question}\nContexts: {docs}")
    ])
    chain = prompt | llm
    docs_text = "\n\n".join(gs.get('documents', [])) if gs.get('documents') else "No documents retrieved."
    result = chain.invoke({
        "question": gs['question'],
        "docs": docs_text,
    })
    return {"answer": result.content[1]['text']}

def route_logic(gs: GraphState):
    worker = gs.get('next_node')
    if worker == "END":
        return "__end__"
    return worker