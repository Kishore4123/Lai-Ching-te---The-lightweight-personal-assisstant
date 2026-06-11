from dotenv import load_dotenv
from imap_tools import MailBox
import os

from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import ToolMessage

load_dotenv()

EMAIL_ACCOUNT = os.getenv('EMAIL_ACCOUNT')
APP_PASSWORD = os.getenv('PASSWORD')
emailMsg = """  """

model = ChatOllama(
    model="gemma4:31b", # Note: Updated to standard gemma2 model name syntax
    base_url="http://localhost:11434",
    temperature=0.2
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load the existing database you created in main.py
persist_directory = r"/home/mic-733ao/Email_Agent"
collection_name = "Email_DB"

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name=collection_name
)

retriever = vectorstore.as_retriever(
       search_type = "similarity",
       search_kwargs = {"k":5}
)



@tool
def retriever_tool(query: str) -> str:
    """
       This tool searches and returns the information from all emails stored in the vector database.
    """
    
    docs = retriever.invoke(query)

    if not docs:
       return "I found no relevant information from the list of all the emails stored"
    
    results = []
    for i, doc in enumerate(docs):
         results.append(f"Document {i+1}:\n{doc.page_content}")
    return "\n\n".join(results)

tools = [retriever_tool]

model = model.bind_tools(tools)

class AgentState(TypedDict):
      messages: Annotated[Sequence[BaseMessage], add_messages]        

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state["messages"][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """You are an intelligent AI Email Analyzer assistant. Your primary role is to process, summarize, and extract actionable information from the user's fetched emails using the available vector store retriever tool. 

When answering questions or analyzing content:
1. Always base your answers strictly on the email context provided by the retriever tool.
2. If you need to look up specific emails, sender details, or history before answering or asking a follow-up question, utilize the retriever tool immediately.
3. Be concise and prioritize extracting key metadata such as dates, deadlines, and actionable tasks (especially for calendar scheduling).
4. Please always cite or reference the specific parts of the emails (such as the Sender or Subject) that you use to support your answers so the user can verify them.
5. If the required information cannot be found within the retrieved email data, state clearly that it is not available in the current mailbox logs."""

tools_dict = {our_tool.name: our_tool for our_tool in tools}

def call_llm(state: AgentState) -> AgentState:
    """Function to call LLM with current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content = system_prompt)] + messages
    response = model.invoke(messages)
    print(response)
    return {'messages': [response]}

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""
    tool_calls = state['messages'][-1].tool_calls
    results = []
    
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if t['name'] not in tools_dict:  # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            
        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        
    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

graph = StateGraph(AgentState)

# Define the nodes
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

# Define the entry point
graph.set_entry_point("llm")

# Define the conditional edges (deciding whether to continue or stop)
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "retriever_agent", 
        False: END
    }
)

# Define the normal edge (looping back to the LLM)
graph.add_edge("retriever_agent", "llm")

# Compile the graph into a runnable agent
rag_agent = graph.compile()

def running_agent():
    print("\n=== RAG AGENT ===")

    while True:
        user_input = input("\nWhat is your question: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        # Converts user input to a HumanMessage type and wraps it in a list
        messages = [HumanMessage(content=user_input)]

        # Invoke the graph with the state dictionary
        result = rag_agent.invoke({"messages": messages})

        print("\n=== ANSWER ===")
        print(result["messages"][-1].content)


# Call the function to run the loop
if __name__ == "__main__":
    running_agent()