import streamlit as st
import pandas as pd
import numpy as np

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
df = pd.read_csv("bpo_customer_experience_dataset.csv", encoding="latin1")
df.columns = df.columns.str.strip()

df['Week'] = pd.to_datetime(df['Week'], errors='coerce')
df = df.dropna(subset=['Week'])

df['DSAT'] = df['Customer_Effortless'].apply(lambda x: 1 if str(x).lower() == "no" else 0)

# -------------------------------
# ISSUE MODEL (ML)
# -------------------------------
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

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_text = vectorizer.fit_transform(df['Customer_Comment'])
y_text = df['Issue_Label']

X_train, X_test, y_train, y_test = train_test_split(
    X_text, y_text, test_size=0.3, stratify=y_text, random_state=42
)

issue_model = LogisticRegression(max_iter=300)
issue_model.fit(X_train, y_train)

nlp_accuracy = accuracy_score(y_test, issue_model.predict(X_test))

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

model = RandomForestRegressor()
model.fit(weekly_df[features], weekly_df['DSAT_Count'])

# -------------------------------
# 📊 MODEL ACCURACY (TOP)
# -------------------------------
st.subheader("📊 Model Accuracy")

sample = weekly_df.sample(min(100, len(weekly_df)))
mae = mean_absolute_error(sample['DSAT_Count'], model.predict(sample[features]))
r2 = r2_score(sample['DSAT_Count'], model.predict(sample[features]))

c1, c2, c3 = st.columns(3)

# 🔥 FIXED ORDER
c1.metric("Issue Detection Accuracy", f"{round(nlp_accuracy*100,1)}%")
c2.metric("Prediction Reliability (R²)", round(r2,2))
c3.metric("Avg Prediction Error", round(mae,2))

# -------------------------------
# 🚨 RISK SEGMENTATION
# -------------------------------
st.subheader("🚨 Risk Segmentation")

agent_summary = weekly_df.groupby('Agent_Name')['DSAT_Count'].mean().reset_index()
mean = weekly_df['DSAT_Count'].mean()
std = weekly_df['DSAT_Count'].std()

agent_summary['Risk'] = (agent_summary['DSAT_Count'] - mean)/std*10 + 50

col1, col2 = st.columns(2)

with col1:
    st.error("High Risk Agents")
    st.dataframe(agent_summary[agent_summary['Risk'] > 60])

with col2:
    st.success("Low Risk Agents")
    st.dataframe(agent_summary[agent_summary['Risk'] < 45])

# -------------------------------
# 👤 AGENT SELECT (CORRECT POSITION)
# -------------------------------
agent = st.selectbox("Select Agent", weekly_df['Agent_Name'].unique())

agent_data = weekly_df[weekly_df['Agent_Name']==agent].sort_values('Week')
latest = agent_data.iloc[-1]

# -------------------------------
# 📈 WEEKLY TREND
# -------------------------------
st.subheader("📈 Weekly DSAT Trend")
st.line_chart(agent_data.set_index('Week')['DSAT_Count'])

# -------------------------------
# 🎯 PREDICTION
# -------------------------------
prediction = model.predict([latest[features]])[0]
risk = (prediction - mean)/std*10 + 50

st.subheader("🎯 Case Handling & Prediction")

col4, col5 = st.columns(2)
col4.metric("Predicted DSAT (Next Week)", int(prediction))
col5.metric("Risk Score", int(risk))

# -------------------------------
# 📊 ISSUE BREAKDOWN
# -------------------------------
agent_comments = df[df['Agent_Name']==agent]

pred_issues = issue_model.predict(
    vectorizer.transform(agent_comments['Customer_Comment'])
)

issue_df = pd.DataFrame(pred_issues, columns=["Issue"])
issue_df = issue_df.value_counts().reset_index(name="Count")

st.subheader("📊 Issue Breakdown")
st.dataframe(issue_df)
st.bar_chart(issue_df.set_index("Issue")["Count"])

# -------------------------------
# 🤖 AI INSIGHT + ROOT + ACTION
# -------------------------------
st.subheader("🤖 AI Insight")

top_issue = issue_df.sort_values("Count", ascending=False).iloc[0]["Issue"]

if risk < 45:

    st.markdown(f"""
### ✅ Performance Summary

Agent **{agent}** is performing strongly.

- Predicted DSAT: **{int(prediction)}**
- Risk Score: **{int(risk)}**

""")

    st.markdown("""
### 💡 Strengths
- Strong communication
- Stable performance
- Low customer dissatisfaction
""")

else:

    if top_issue == "Communication":
        root = "Poor communication quality"
        action = "- Improve empathy\n- Avoid scripted replies\n- Listen actively"
    elif top_issue == "Process":
        root = "Delays and inefficiencies"
        action = "- Reduce wait time\n- Avoid transfers\n- Take ownership"
    else:
        root = "Product issues"
        action = "- Improve product knowledge\n- Escalate faster"

    st.markdown(f"""
### ⚠️ Performance Needs Attention

- Predicted DSAT: **{int(prediction)}**
- Risk Score: **{int(risk)}**

### 🔍 Root Cause
{root}

### 💡 Actions
{action}
""")
