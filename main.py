from dotenv import load_dotenv
import os

load_dotenv()
from scripts.workflow import graph

if __name__ == "__main__":
    print("This is agent to answer the question about policy of Company A. If you want to end the session, please type 'exit'.")
    print("______________________________________________________________")
    while(True):
        question = input("Please ask your question : ")
        if question == "exit":
            break
        init_state = {
            "question" : question,
            "documents" : [],
            "answer" : "",
            "next_node" : "supervisor_agent"
        }
        answer = graph.invoke(init_state, config={"recursion_limit": 30})
        print("=========== Answer ===========")
        print(answer['answer'])