import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

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

df = None
for encoding in ["utf-8", "latin1", "utf-16"]:
    for sep in [",", "|", ";", "\t"]:
        try:
            temp = pd.read_csv(file_path, encoding=encoding, sep=sep)
            if temp.shape[1] > 1:
                df = temp
                break
        except:
            continue
    if df is not None:
        break

if df is None or df.empty:
    st.error("❌ Failed to load dataset")
    st.stop()

df.columns = df.columns.str.strip()

# -------------------------------
# CLEANING
# -------------------------------
df['Week'] = pd.to_datetime(df['Week'], errors='coerce')
df = df.dropna(subset=['Week'])

df['DSAT'] = df['Customer_Effortless'].apply(lambda x: 1 if str(x).lower() == "no" else 0)
df['Sentiment'] = df['Customer_Comment'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# -------------------------------
# ISSUE LABEL
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

# -------------------------------
# NLP MODEL
# -------------------------------
df_clean = df[df['Issue_Label'] != "Other"].copy()

vectorizer = None
issue_model = None
nlp_accuracy = 0

if len(df_clean) > 50:
    train_df, test_df = train_test_split(df_clean, test_size=0.3, stratify=df_clean['Issue_Label'])

    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    X_train = vectorizer.fit_transform(train_df['Customer_Comment'])
    y_train = train_df['Issue_Label']

    X_test = vectorizer.transform(test_df['Customer_Comment'])
    y_test = test_df['Issue_Label']

    issue_model = LogisticRegression(max_iter=200)
    issue_model.fit(X_train, y_train)

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
# AGENT SELECT
# -------------------------------
agent = st.selectbox("Select Agent", weekly_df['Agent_Name'].unique())

agent_data = weekly_df[weekly_df['Agent_Name']==agent].sort_values('Week')
latest = agent_data.iloc[-1]

prediction = model.predict([latest[features]])[0]
risk = (prediction - y.mean())/y.std()*10 + 50

# -------------------------------
# METRICS
# -------------------------------
agent_actual = agent_data['DSAT_Count']
agent_pred = model.predict(agent_data[features])

mae = mean_absolute_error(agent_actual, agent_pred)

st.subheader("📊 Performance Metrics")
st.metric("Predicted DSAT", int(prediction))
st.metric("Risk Score", int(risk))
st.metric("Avg Prediction Error", round(mae,2))

# -------------------------------
# 🚨 RISK SEGMENT (RESTORED)
# -------------------------------
st.subheader("🚨 Risk Segmentation")

agent_summary = weekly_df.groupby('Agent_Name')['DSAT_Count'].mean().reset_index()
agent_summary['Risk'] = (agent_summary['DSAT_Count'] - y.mean())/y.std()*10 + 50

high_risk = agent_summary[agent_summary['Risk'] > 60]
low_risk = agent_summary[agent_summary['Risk'] < 45]

col1, col2 = st.columns(2)

with col1:
    st.error("High Risk Agents")
    st.dataframe(high_risk.head(5))

with col2:
    st.success("Low Risk Agents")
    st.dataframe(low_risk.head(5))

# -------------------------------
# ISSUE BREAKDOWN
# -------------------------------
agent_comments = df[df['Agent_Name']==agent]
dsat_comments = agent_comments[agent_comments['DSAT']==1]['Customer_Comment']

issues = ["Communication","Process","Product"]

if len(dsat_comments) > 0 and issue_model:
    X_test_agent = vectorizer.transform(dsat_comments)
    pred_issues = issue_model.predict(X_test_agent)

    issue_df = pd.DataFrame(pred_issues, columns=["Issue"])
    issue_df = issue_df.value_counts().reset_index(name="Count")

    for i in issues:
        if i not in issue_df["Issue"].values:
            issue_df = pd.concat([issue_df, pd.DataFrame({"Issue":[i],"Count":[0]})])
else:
    issue_df = pd.DataFrame({"Issue": issues, "Count": [0,0,0]})

st.subheader("📊 Issue Breakdown")
st.dataframe(issue_df)

# -------------------------------
# 🔥 FINAL AI INSIGHT (LOGIC FIXED)
# -------------------------------
def generate_ai_insight(agent, pred, risk, issue_df, trend):

    top_issue = issue_df.sort_values("Count", ascending=False).iloc[0]["Issue"]

    if risk < 45:
        return f"""
### 🤖 AI Insight

**Agent:** {agent}  
**Performance:** Strong Performer

- Predicted DSAT: {int(pred)}
- Risk Score: {int(risk)}

✅ The agent is consistently delivering a good customer experience.

### 💡 Strengths
- Low DSAT trend
- Stable performance
- Good handling across key areas

👉 Continue current approach and maintain consistency.
"""

    # -------------------------------
    # LOW PERFORMANCE LOGIC
    # -------------------------------
    if top_issue == "Communication":
        problem = "Customer dissatisfaction is driven by communication gaps."
        coaching = "- Improve empathy\n- Avoid scripted responses\n- Listen actively"
    elif top_issue == "Process":
        problem = "Customer dissatisfaction is driven by delays and inefficient handling."
        coaching = "- Reduce wait time\n- Avoid transfers\n- Take ownership"
    else:
        problem = "Customer dissatisfaction is driven by product-related issues."
        coaching = "- Improve product knowledge\n- Escalate issues faster"

    trend_msg = "DSAT is rising, indicating declining performance." if trend > 0 else "Performance needs monitoring."

    return f"""
### 🤖 AI Insight

**Agent:** {agent}  
**Performance:** Needs Attention

- Predicted DSAT: {int(pred)}
- Risk Score: {int(risk)}

{trend_msg}

### 🔍 Root Cause
{problem}

### 💡 Recommended Actions
{coaching}
"""

trend = latest['DSAT_lag_1'] - latest['DSAT_lag_4']

st.subheader("🤖 AI Insight")
st.markdown(generate_ai_insight(agent, prediction, risk, issue_df, trend))
