import customtkinter as ctk
import threading
import subprocess
from langchain_core.messages import HumanMessage

# Import the compiled LangGraph agent from your script
# This allows the UI to talk directly to the LLM loaded in memory
from RAG_Agent import rag_agent

class RoyalEmailAgentUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("Lai Ching-te (賴清德)")
        self.geometry("900x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # --- Layout Grid ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar (Controls) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Agent Controls", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.collect_btn = ctk.CTkButton(self.sidebar_frame, text="📥 Collect Emails", command=self.start_email_collection)
        self.collect_btn.grid(row=1, column=0, padx=20, pady=10)

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Status: Ready", text_color="gray")
        self.status_label.grid(row=2, column=0, padx=20, pady=10)

        # --- Main Chat Area ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Chat Display (Read-Only)
        self.chat_display = ctk.CTkTextbox(self.main_frame, font=ctk.CTkFont(size=14), wrap="word")
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 10), sticky="nsew")
        self.chat_display.insert("0.0", "Welcome, King John. The AI is online and ready.\n\n")
        self.chat_display.configure(state="disabled")

        # User Input Box
        self.entry_box = ctk.CTkEntry(self.main_frame, placeholder_text="Ask about your schedule or emails...")
        self.entry_box.grid(row=1, column=0, padx=(10, 10), pady=(0, 10), sticky="ew")
        self.entry_box.bind("<Return>", lambda event: self.start_chat_query())

        # Send Button
        self.send_btn = ctk.CTkButton(self.main_frame, text="Send", width=80, command=self.start_chat_query)
        self.send_btn.grid(row=1, column=1, padx=(0, 10), pady=(0, 10))

    # --- UI Helper Functions ---
    def append_to_chat(self, text, sender="System"):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"[{sender}]\n{text}\n\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def update_status(self, text, color="gray"):
        self.status_label.configure(text=f"Status: {text}", text_color=color)

    # --- Chat RAG Threading ---
    def start_chat_query(self):
        user_input = self.entry_box.get()
        if not user_input.strip():
            return
            
        self.entry_box.delete(0, "end")
        self.append_to_chat(user_input, "Kishor John")
        
        self.send_btn.configure(state="disabled")
        self.update_status("Thinking...", "yellow")
        
        # Run the LLM in a separate thread so UI doesn't freeze
        threading.Thread(target=self.process_chat_query, args=(user_input,), daemon=True).start()

    def process_chat_query(self, user_input):
        try:
            messages = [HumanMessage(content=user_input)]
            result = rag_agent.invoke({"messages": messages})
            answer = result["messages"][-1].content
            
            # Update UI back on the main thread
            self.after(0, self.append_to_chat, answer, "AI Agent")
            self.after(0, self.update_status, "Ready", "gray")
        except Exception as e:
            self.after(0, self.append_to_chat, f"Error: {str(e)}", "System")
            self.after(0, self.update_status, "Error", "red")
        finally:
            self.after(0, lambda: self.send_btn.configure(state="normal"))

    # --- Email Pipeline Threading ---
    def start_email_collection(self):
        self.collect_btn.configure(state="disabled")
        self.update_status("Fetching Emails...", "yellow")
        self.append_to_chat("Initiating email pipeline. Fetching, summarizing, and updating calendar...", "System")
        
        # Run the heavy main.py script in a background thread
        threading.Thread(target=self.run_email_pipeline, daemon=True).start()

    def run_email_pipeline(self):
        try:
            # We run main.py and pipe the output
            process = subprocess.Popen(
                ["python3", "-u", "main.py"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, # Merge errors into standard output
                text=True,
                bufsize=1 # Line buffered
            )
            
            # Read the output line by line as it generates
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Send each line to the chat window immediately
                    self.after(0, self.append_to_chat, line.strip(), "Pipeline")
            
            process.stdout.close()
            return_code = process.wait()
            
            if return_code == 0:
                self.after(0, self.update_status, "Ready", "green")
                self.after(0, self.append_to_chat, "✅ Pipeline Completed Successfully.", "System")
            else:
                self.after(0, self.update_status, "Error", "red")
                self.after(0, self.append_to_chat, "❌ Pipeline Encountered an Error.", "System")
                
        except Exception as e:
            self.after(0, self.append_to_chat, f"Failed to execute main.py: {str(e)}", "System")
            self.after(0, self.update_status, "Error", "red")
        finally:
            self.after(0, lambda: self.collect_btn.configure(state="normal"))

if __name__ == "__main__":
    app = RoyalEmailAgentUI()
    app.mainloop()