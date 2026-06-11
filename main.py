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
from langchain_core.documents import Document

import json
import pickle
import pandas as pd
from datetime import datetime
from gcsa.event import Event
from gcsa.google_calendar import GoogleCalendar

load_dotenv()

EMAIL_ACCOUNT = os.getenv('EMAIL_ACCOUNT')
APP_PASSWORD = os.getenv('PASSWORD')
emailMsg = """  """

model = ChatOllama(
    model="gemma4:31b", 
    base_url="http://localhost:11434",
    temperature=0.2
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

text_splitter = RecursiveCharacterTextSplitter(
         chunk_size = 1000,
         chunk_overlap = 200
)

class AgentState(TypedDict):
      messages: Annotated[Sequence[BaseMessage], add_messages]

def email_summarizer(state: AgentState) -> AgentState: 
    print("🤖 Gemma 31B is summarizing the emails... (This takes a moment)")
    system_prompt = SystemMessage(content=
            "You are an AI email summarizer. Analyze the user's emails given as human-message. Then, summarize the entire contents of the email in 5 lines. "
    )
    msgs = [system_prompt] + state["messages"]
    response = model.invoke(msgs)
    return {"messages": [response]}

def calendar_extractor(state: AgentState) -> AgentState:
    print("📅 Gemma 31B is extracting calendar events...") # ADD THIS LINE
    system_prompt = SystemMessage(content=
            """You are an AI calendar dates extractor. Extract events/tasks from the emails.
            You MUST output ONLY a valid JSON list of lists. No conversational text.
            Example: [["event1Name","2026-05-28 10:00:00" ], ["Deadline of task","2026-06-07 12:00:00"]]"""
    )
    response = model.invoke([system_prompt] + state["messages"])
    print(response.content)
    raw_string = response.content
    cleaned_string = raw_string.strip().strip("`").replace("json\n", "").strip()
    
    events_list = []
    try:
        events_list = json.loads(cleaned_string)
    except json.JSONDecodeError:
        print("Error: The LLM did not output valid JSON.")
    
    # Early Exit: If no events were found, stop here so the ML model doesn't crash
    if not events_list:
        print("No events to show")
        return {"messages": []}
        
    MODEL_PATH = "event_classifier.pkl"
    with open(MODEL_PATH, 'rb') as file:
        calendar_model = pickle.load(file)
        
    df = pd.DataFrame(events_list, columns=["event_text", "event_date"])
    df["is_important"] = calendar_model.predict(df["event_text"])
    
    important_df = df[df["is_important"] == 1]
    calendarEvents = important_df[["event_text", "event_date"]].values.tolist()
    
    # Updated: Explicitly pointing to local credentials to prevent FileNotFoundError
    # Update your GoogleCalendar initialization to look exactly like this:
    calendar = GoogleCalendar(
        EMAIL_ACCOUNT,
        credentials_path='credentials.json',
        token_path='token.pickle',
    )
    
    for event_data in calendarEvents:
        event_title = event_data[0]
        raw_date_string = event_data[1]
        parsed_start_time = datetime.strptime(raw_date_string, "%Y-%m-%d %H:%M:%S")
    
        event = Event(
            title=event_title,
            start=parsed_start_time,
        )
        try:
            calendar.add_event(event)
            print(f"✅ Successfully added: '{event_title}' scheduled for {parsed_start_time}")
        except Exception as e:
            print(f"❌ Failed to add '{event_title}': {e}")
            
    return {"messages": []}

graph = StateGraph(AgentState)
graph.add_node("EmailSummarizer", email_summarizer)
graph.add_node("CalendarExtractor", calendar_extractor)
graph.set_entry_point("EmailSummarizer")
graph.add_edge("EmailSummarizer","CalendarExtractor")
graph.add_edge("CalendarExtractor", END)
app = graph.compile()

# --- Execution Block (Only runs if script is executed directly or called via UI button) ---
if __name__ == "__main__":
    raw_docs = []
    with MailBox("imap.gmail.com").login(EMAIL_ACCOUNT, APP_PASSWORD, "Inbox") as mb:
         for i, msg in enumerate(mb.fetch("UNSEEN", limit=5, reverse=True, mark_seen=True)):
             emailMsg += f"--- Email {i+1} ---\nSubject: {msg.subject}\nBody: {msg.text}\n\n"
             raw_docs.append(Document(
                    page_content=f"Subject: {msg.subject}\nBody: {msg.text}",
                    metadata={
                        "sender": msg.from_,      
                        "subject": msg.subject,   
                        "uid": msg.uid            
                    }
                ))
             
    final_docs = text_splitter.split_documents(raw_docs)
    
    # Extract unique UIDs to prevent vector database duplication
    # Extract UIDs and append a chunk index to guarantee uniqueness for Chroma
    doc_ids = [f"{doc.metadata['uid']}_chunk_{i}" for i, doc in enumerate(final_docs)]

    persist_directory = r"/home/mic-733ao/Email_Agent"
    collection_name = "Email_DB"

    if not os.path.exists(persist_directory):
       os.makedirs(persist_directory)

    try:
        # Load the existing DB and add unique documents safely
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        
        if final_docs:
            vector_store.add_documents(documents=final_docs, ids=doc_ids)
        print("Stored in Vector Database (Duplicates Prevented)")

    except Exception as e:
           print(f"Error : {str(e)}") 
           raise 

    inputs = {"messages": [HumanMessage(content=emailMsg)]}

    for chunk in app.stream(inputs):
         print(chunk)