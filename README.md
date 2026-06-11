# Lai-Ching-te---The-lightweight-personal-assisstant
Agentic AI system built to manage your email workflow
# Autonomous Email Intelligence Ecosystem

An advanced, AI-driven personal email assistant that transforms unstructured mailbox data into an active, self-optimizing knowledge base. 

This project goes beyond simple email fetching; it employs a dual-pipeline architecture utilizing Large Language Models (LLMs), RAG (Retrieval-Augmented Generation), predictive machine learning, and reinforcement learning to automate schedule management and optimize digital communication.

## 🌟 Core Features

* **Intelligent Data Intake & Summarization:** Automatically fetches unread emails via IMAP, parses them, and generates concise 5-line summaries using a local **Gemma 4 (31B)** model orchestrated by LangGraph.
* **Predictive Inbox Hygiene:** A trained **Random Forest** classification module acts as a gatekeeper, performing real-time identification to automatically delete spam/ads and extract only high-value commitments (e.g., project deadlines, meetings).
* **Adaptive Q-Learning (RL Feedback Loop):** The RAG Agent includes a specialized reinforcement learning module. Users can provide direct feedback on the agent's actions via the chat interface. The Q-learning algorithm processes this feedback as a reward/penalty to continuously adjust its behavioral policy for future responses and scheduling logic.
* **Semantic RAG Memory:** Email chunks are embedded and indexed into a local **Chroma Vector Database**. Duplicate UIDs are strictly tracked to prevent data redundancy, allowing the user to seamlessly "chat with their inbox" over time.
* **Headless Calendar Automation:** Validated events are pushed silently to Google Calendar using the **GCSA** library and a Google Cloud Service Account, bypassing the need for manual OAuth browser popups.
* **Real-Time Responsive UI:** A sleek, threaded **CustomTkinter** dashboard that allows for interactive chat queries while running the heavy backend pipeline in an unbuffered subprocess, streaming terminal logs directly to the user interface.

---

## 🏗️ Architecture Setup

### 1. Prerequisites
* **Python 3.10+** (Ubuntu/Linux recommended)
* **Ollama** installed locally with the `gemma4:31b` (or equivalent) model pulled.
* A **Google Cloud Project** with the Google Calendar API enabled.

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/yourusername/Email_Agent.git](https://github.com/yourusername/Email_Agent.git)
cd Email_Agent
pip install -r requirements.txt
(Ensure customtkinter, langgraph, langchain-ollama, langchain-chroma, imap-tools, pandas, scikit-learn, and gcsa are included in your environment).

3. Configuration & Authentication
You need two authentication layers to run the agent:

A. Email IMAP Credentials
Create a .env file in the root directory and add your email app password (do not use your standard email password):
EMAIL_ACCOUNT=your_email@gmail.com
PASSWORD=your_app_specific_password

B. Google Calendar Service Account

Navigate to the Google Cloud Console.

Create a Service Account and download the JSON key.

Rename the downloaded file exactly to service_account.json and place it in the root directory.

Go to your personal Google Calendar settings and Share your calendar with the Service Account's email address, granting it "Make changes to events" permissions.

4. Machine Learning Models
Ensure your pre-trained Random Forest model (event_classifier.pkl) is placed in the root directory. This model is critical for the main.py pipeline to classify events correctly.

🚀 Usage
The entire ecosystem is unified under the graphical interface.

To launch the dashboard, run:

Bash
python3 ui_app.py
Dashboard Controls:
📥 Collect Emails: Triggers the background data pipeline (main.py). This fetches emails, runs the summarization, processes the Random Forest classification, syncs with Google Calendar, and updates the Chroma vector database. The -u subprocess flag ensures real-time progress updates are streamed to the UI.

💬 Chat Interface: Interact with the RAG_Agent.py. You can ask questions about your schedule, request specific sender histories, or provide direct behavioral feedback which the Q-Learning tool will process to optimize future agent operations.

📁 File Structure
Plaintext
/Email_Agent/
├── ui_app.py               # Main CustomTkinter Dashboard 
├── main.py                 # Intake, ML Pipeline, & Calendar Sync
├── RAG_Agent.py            # LangGraph Chat Agent & Q-Learning Tools
├── credentials.json        # Google Service Account Key (Keep Private!)
├── event_classifier.pkl    # Pickled Random Forest Model
├── .env                    # IMAP Passwords (Keep Private!)
└── /Email_DB/              # Auto-generated Chroma Vector Store
├── DockerFile                   
└── Docker.yaml


3. Execution Instructions
To bring your ecosystem to life inside the container, open your Ubuntu terminal and run these exact commands:

Step 1: Grant UI Permissions
Because Docker is technically a separate user, you must grant it permission to draw windows on your current screen. Run this once per session:

Bash
xhost +local:docker
Step 2: Build and Launch
Navigate to your project directory and start the compose cluster:

Bash
docker-compose up --build
