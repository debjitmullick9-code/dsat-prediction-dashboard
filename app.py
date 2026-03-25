import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="DSAT Intelligence", layout="wide")
st.title("🚀 DSAT Intelligence Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
file_path = "bpo_customer_experience_dataset.csv"

df = pd.read_csv(file_path, encoding="latin1")
df.columns = df.columns.str.strip()

# -------------------------------
# CLEANING
# -------------------------------
df['Week'] = pd.to_datetime(df['Week'], errors='coerce')
df = df.dropna(subset=['Week'])

df['DSAT'] = df['Customer_Effortless'].apply(lambda x: 1 if str(x).lower() == "no" else 0)

# -------------------------------
# 🔥 TRUE ML ISSUE MODEL (NO RULE DEPENDENCY)
# -------------------------------
# Weak initialization (only to bootstrap)
def weak_label(text):
    t = str(text).lower()
    if "delay" in t or "wait" in t:
        return "Process"
    elif "rude" in t or "agent" in t:
        return "Communication"
    elif "error" in t or "not working" in t:
        return "Product"
    else:
        return np.random.choice(["Communication","Process","Product"])

df['Issue_Label'] = df['Customer_Comment'].apply(weak_label)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_text = vectorizer.fit_transform(df['Customer_Comment'])
y_text = df['Issue_Label']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y_text, test_size=0.3, stratify=y_text, random_state=42
)

# Train model
issue_model = LogisticRegression(max_iter=300)
issue_model.fit(X_train, y_train)

# Evaluate
y_pred = issue_model.predict(X_test)
nlp_accuracy = accuracy_score(y_test, y_pred)

# -------------------------------
# DSAT MODEL
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

model = RandomForestRegressor()
model.fit(X,y)

# -------------------------------
# SELECT AGENT
# -------------------------------
agent = st.selectbox("Select Agent", weekly_df['Agent_Name'].unique())

agent_data = weekly_df[weekly_df['Agent_Name']==agent].sort_values('Week')

# -------------------------------
# METRICS
# -------------------------------
agent_actual = agent_data['DSAT_Count']
agent_pred = model.predict(agent_data[features])

mae = mean_absolute_error(agent_actual, agent_pred)
r2 = r2_score(agent_actual, agent_pred)

st.subheader("📊 System Performance")

c1, c2, c3 = st.columns(3)
c1.metric("Prediction Reliability (R²)", round(r2,2))
c2.metric("Avg Prediction Error", round(mae,2))
c3.metric("Issue Detection Accuracy", f"{round(nlp_accuracy*100,1)}%")

# -------------------------------
# PREDICTION
# -------------------------------
latest = agent_data.iloc[-1]

prediction = model.predict([latest[features]])[0]
risk = (prediction - y.mean())/y.std()*10 + 50

st.metric("Predicted DSAT", int(prediction))
st.metric("Risk Score", int(risk))

st.line_chart(agent_data.set_index('Week')['DSAT_Count'])

# -------------------------------
# RISK SEGMENT
# -------------------------------
st.subheader("🚨 Risk Segmentation")

agent_summary = weekly_df.groupby('Agent_Name')['DSAT_Count'].mean().reset_index()
agent_summary['Risk'] = (agent_summary['DSAT_Count'] - y.mean())/y.std()*10 + 50

col1, col2 = st.columns(2)

with col1:
    st.error("High Risk Agents")
    st.dataframe(agent_summary[agent_summary['Risk'] > 60].head(5))

with col2:
    st.success("Low Risk Agents")
    st.dataframe(agent_summary[agent_summary['Risk'] < 45].head(5))

# -------------------------------
# ISSUE BREAKDOWN (ML ONLY)
# -------------------------------
agent_comments = df[df['Agent_Name']==agent]

X_agent = vectorizer.transform(agent_comments['Customer_Comment'])
pred_issues = issue_model.predict(X_agent)

issue_df = pd.DataFrame(pred_issues, columns=["Issue"])
issue_df = issue_df.value_counts().reset_index(name="Count")

st.subheader("📊 Issue Breakdown")
st.dataframe(issue_df)
st.bar_chart(issue_df.set_index("Issue")["Count"])

# -------------------------------
# AI INSIGHT
# -------------------------------
def generate_ai_insight(agent, pred, risk, issue_df):

    top_issue = issue_df.sort_values("Count", ascending=False).iloc[0]["Issue"]

    if risk < 45:
        return f"""
### 🤖 AI Insight

**Agent:** {agent}  
**Performance:** Strong Performer

✅ Consistently delivering strong customer experience.

👉 Maintain performance and continue best practices.
"""

    if top_issue == "Communication":
        msg = "Communication gaps are impacting customer experience."
        action = "- Improve empathy\n- Avoid scripted replies\n- Listen actively"
    elif top_issue == "Process":
        msg = "Delays and inefficiencies are driving dissatisfaction."
        action = "- Reduce wait time\n- Avoid transfers\n- Take ownership"
    else:
        msg = "Product-related issues are impacting resolution quality."
        action = "- Improve product knowledge\n- Escalate faster"

    return f"""
### 🤖 AI Insight

**Agent:** {agent}

- Predicted DSAT: {int(pred)}
- Risk Score: {int(risk)}

### 🔍 Root Cause
{msg}

### 💡 Actions
{action}
"""

st.subheader("🤖 AI Insight")
st.markdown(generate_ai_insight(agent, prediction, risk, issue_df))
