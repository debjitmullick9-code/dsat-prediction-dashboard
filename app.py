import streamlit as st
import pandas as pd
import numpy as np
import os

from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

from textblob import TextBlob

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="DSAT Intelligence", layout="wide")
st.title("🚀 DSAT Intelligence Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
file_path = "bpo_customer_experience_dataset.csv"

df = pd.read_csv(file_path, encoding="latin1")
df.columns = df.columns.str.strip()

df['Week'] = pd.to_datetime(df['Week'], errors='coerce')
df = df.dropna(subset=['Week'])

df['DSAT'] = df['Customer_Effortless'].apply(lambda x: 1 if str(x).lower() == "no" else 0)

# -------------------------------
# 🔥 SENTIMENT MODEL (ML)
# -------------------------------
def get_sentiment_label(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df['Sentiment_Label'] = df['Customer_Comment'].apply(get_sentiment_label)

sent_vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
X_sent = sent_vectorizer.fit_transform(df['Customer_Comment'])
y_sent = df['Sentiment_Label']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sent, y_sent, test_size=0.3, random_state=42
)

sent_model = LogisticRegression(max_iter=200)
sent_model.fit(X_train_s, y_train_s)

df['Predicted_Sentiment'] = sent_model.predict(
    sent_vectorizer.transform(df['Customer_Comment'])
)

# -------------------------------
# ISSUE MODEL
# -------------------------------
def label_issue(comment):
    c = str(comment).lower()
    if any(w in c for w in ["rude","bad","angry","unhelpful"]):
        return "Communication"
    elif any(w in c for w in ["delay","wait","slow","long"]):
        return "Process"
    elif any(w in c for w in ["error","bug","failed","not working"]):
        return "Product"
    else:
        return "Other"

df['Issue_Label'] = df['Customer_Comment'].apply(label_issue)

df_clean = df[df['Issue_Label'] != "Other"].copy()

noise_idx = np.random.choice(df_clean.index, int(len(df_clean)*0.2), replace=False)
df_clean.loc[noise_idx, 'Issue_Label'] = np.random.choice(
    ["Communication","Process","Product"], len(noise_idx)
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)

X_train = vectorizer.fit_transform(df_clean['Customer_Comment'])
y_train = df_clean['Issue_Label']

issue_model = LogisticRegression(max_iter=200)
issue_model.fit(X_train, y_train)

# -------------------------------
# 🔥 XGBOOST DSAT MODEL
# -------------------------------
weekly_df = df.groupby(['Agent_Name','Week']).agg({
    'DSAT': 'sum',
    'Ticket_ID': 'count'
}).reset_index()

weekly_df.rename(columns={'DSAT':'DSAT_Count','Ticket_ID':'Total_Tickets'}, inplace=True)

for i in range(1,5):
    weekly_df[f'DSAT_lag_{i}'] = weekly_df.groupby('Agent_Name')['DSAT_Count'].shift(i)
    weekly_df[f'Tickets_lag_{i}'] = weekly_df.groupby('Agent_Name')['Total_Tickets'].shift(i)

weekly_df = weekly_df.dropna()

features = [f'DSAT_lag_{i}' for i in range(1,5)] + [f'Tickets_lag_{i}' for i in range(1,5)]

X = weekly_df[features]
y = weekly_df['DSAT_Count']

model = XGBRegressor(n_estimators=200, learning_rate=0.05)
model.fit(X,y)

# -------------------------------
# SELECT AGENT
# -------------------------------
agent = st.selectbox("Select Agent", weekly_df['Agent_Name'].unique())

agent_data = weekly_df[weekly_df['Agent_Name']==agent].sort_values('Week')
latest = agent_data.iloc[-1]

prediction = model.predict([latest[features]])[0]
risk = (prediction - y.mean())/y.std()*10 + 50

# -------------------------------
# MODEL ACCURACY
# -------------------------------
agent_actual = agent_data['DSAT_Count']
agent_pred = model.predict(agent_data[features])

mae = mean_absolute_error(agent_actual, agent_pred)
r2 = r2_score(agent_actual, agent_pred)

st.subheader("📊 Model Accuracy")

c1,c2,c3 = st.columns(3)
c1.metric("Prediction Reliability (R²)", round(r2,2))
c2.metric("Avg Prediction Error", round(mae,2))
c3.metric("Sentiment Model Accuracy", "ML Enabled")

# -------------------------------
# RISK
# -------------------------------
st.subheader("🚨 Risk Segmentation")

agent_summary = weekly_df.groupby('Agent_Name')['DSAT_Count'].mean().reset_index()
agent_summary['Risk'] = (agent_summary['DSAT_Count'] - y.mean())/y.std()*10 + 50

st.dataframe(agent_summary.sort_values("Risk", ascending=False).head(10))

# -------------------------------
# TREND
# -------------------------------
st.line_chart(agent_data.set_index('Week')['DSAT_Count'])

# -------------------------------
# ISSUE BREAKDOWN
# -------------------------------
agent_comments = df[df['Agent_Name']==agent]

pred_issues = issue_model.predict(
    vectorizer.transform(agent_comments['Customer_Comment'])
)

issue_df = pd.DataFrame(pred_issues, columns=["Issue"])
issue_df = issue_df.value_counts().reset_index(name="Count")

st.subheader("📊 Issue Breakdown")
st.dataframe(issue_df)

# -------------------------------
# AI INSIGHT
# -------------------------------
st.subheader("🤖 AI Insight")

top_issue = issue_df.sort_values("Count", ascending=False).iloc[0]["Issue"]

sent_counts = agent_comments['Predicted_Sentiment'].value_counts()

sentiment = sent_counts.idxmax() if len(sent_counts) > 0 else "Neutral"

if risk < 45:
    st.success(f"{agent} is performing strongly with positive sentiment.")
else:
    st.warning(f"Performance risk detected. Dominant issue: {top_issue}")
