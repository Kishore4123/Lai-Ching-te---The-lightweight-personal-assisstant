from gcsa.google_calendar import GoogleCalendar
import os
from dotenv import load_dotenv

# Load your email from the .env file
load_dotenv()
EMAIL_ACCOUNT = os.getenv('EMAIL_ACCOUNT')

print("👑 Opening browser for King John's Royal Authentication...")

# This runs purely on its own, with no LangGraph retries to interrupt it!
calendar = GoogleCalendar(
    EMAIL_ACCOUNT,
    credentials_path='credentials.json',
    token_path='token.pickle'
)

print("✅ Success! token.pickle has been generated. You may now delete this auth.py file.")