import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import pickle

# 1. Your historical training data
# 1 = Important (Add to Calendar), 0 = Ignore

training_data = [
    # --- CLASS 1: Important Events to Mark (30 rows) ---
    ("Global AI Hackathon Opening Ceremony", 1),
    ("Startup Pitch Deck Presentation", 1),
    ("Annual Developer Conference 2026", 1),
    ("Tech Founder Networking Dinner", 1),
    ("Machine Learning Bootcamp Kickoff", 1),
    ("Product Launch: NextGen Analytics", 1),
    ("In-person Workshop: Advanced Python", 1),
    ("Keynote Address: The Future of AI", 1),
    ("Investor Meet and Greet", 1),
    ("Final Demo Day and Award Ceremony", 1),
    ("Hackathon Winner Announcement", 1),
    ("Cybersecurity Summit 2026", 1),
    ("Blockchain Developer Meetup", 1),
    ("Regional Tech Expo", 1),
    ("Innovation Lab Grand Opening", 1),
    ("Cloud Computing Symposium", 1),
    ("Data Science Panel Discussion", 1),
    ("Tech Job Fair and Recruitment Drive", 1),
    ("Robotics Competition Finals", 1),
    ("Web3 Innovators Retreat", 1),
    ("Women in Tech Leadership Summit", 1),
    ("Open Source Contributors Gala", 1),
    ("AI Agent Development Masterclass", 1),
    ("Venture Capital Pitch Session", 1),
    ("Tech Incubator Cohort 5 Orientation", 1),
    ("Fintech Disruption Conference", 1),
    ("IoT Makerspace Collaboration Event", 1),
    ("Augmented Reality App Showcase", 1),
    ("Software Engineering Leaders Summit", 1),
    ("Generative AI Hackathon Day 1", 1),

    # --- CLASS 0: Routine/Mundane Events to Ignore (30 rows) ---
    ("Project milestone deadline", 0),
    ("Weekly team online meeting", 0),
    ("Mathematics Assignment 3 due", 0),
    ("Fall semester Class schedule overview", 0),
    ("Online meeting with design team", 0),
    ("Submission deadline for InnoVent", 0),
    ("Data Structures Assignment 4", 0),
    ("Marketing sync online meeting", 0),
    ("Daily standup online meeting", 0),
    ("Algorithms Class schedule update", 0),
    ("Final project code submission deadline", 0),
    ("HR compliance training online meeting", 0),
    ("Physics lab Assignment", 0),
    ("Q3 taxes filing deadline", 0),
    ("Client check-in online meeting", 0),
    ("Literature review Assignment due", 0),
    ("Computer Architecture Class schedule", 0),
    ("Sprint planning online meeting", 0),
    ("Grant application deadline", 0),
    ("Weekly one-on-one online meeting", 0),
    ("Midterm essay Assignment", 0),
    ("Yoga Class schedule", 0),
    ("Quarterly performance review online meeting", 0),
    ("Draft proposal deadline", 0),
    ("Database architecture Assignment", 0),
    ("Vendor negotiation online meeting", 0),
    ("Chemistry 101 Class schedule", 0),
    ("Software license renewal deadline", 0),
    ("Machine learning model Assignment", 0),
    ("Project post-mortem online meeting", 0)
]

df_train = pd.DataFrame(training_data, columns=["event_text", "is_important"])

# 2. Build and train the pipeline
print("Training the Royal Event Classifier...")
model = make_pipeline(
    TfidfVectorizer(stop_words="english"),
    RandomForestClassifier(n_estimators=100, random_state=42)
)
model.fit(df_train["event_text"], df_train["is_important"])

# 3. Export the model using pickle
pickle_filename = "event_classifier.pkl"
with open(pickle_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Success! Model saved securely as '{pickle_filename}'.")
