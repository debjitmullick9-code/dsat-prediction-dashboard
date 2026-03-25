import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, r2_score

from textblob import TextBlob

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="DSAT Intelligence", layout="wide")
st.title("🚀 DSAT Intelligence Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("bpo_customer_experience_dataset.csv")
df['Week'] = pd.to_datetime(df['Week'])

df['DSAT'] = df['Customer_Effortless'].apply(lambda x: 1 if x == "No" else 0)
df['Sentiment'] = df['Customer_Comment'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# -------------------------------
# ISSUE LABEL FUNCTION
# -------------------------------
def label_issue(comment):
    c = str(comment).lower()
    if any(w in c for w in ["rude","unhelpful","confusing","agent"]):
        return "Communication"
    elif any(w in c for w in ["delay","long","wait","slow","time"]):
        return "Process"
    elif any(w in c for w in ["not working","bug","broken","glitch"]):
        return "Product"
    else:
        return "Other"

# -------------------------------
# NLP MODEL (FIXED)
# -------------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df['Issue_Label'] = train_df['Customer_Comment'].apply(label_issue)
test_df['Issue_Label'] = test_df['Customer_Comment'].apply(label_issue)

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)

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

preds = model.predict(X)

mae = mean_absolute_error(y,preds)
r2 = r2_score(y,preds)

# -------------------------------
# ACCURACY DISPLAY
# -------------------------------
st.subheader("📊 System Accuracy Overview")

col1,col2,col3 = st.columns(3)
col1.metric("Issue Detection Accuracy", f"{round(nlp_accuracy*100,1)}%")
col2.metric("Prediction Reliability (R²)", round(r2,2))
col3.metric("Avg Prediction Error", round(mae,2))

st.caption("Accuracy is evaluated on unseen test data")

# -------------------------------
# ALERT SYSTEM (NEW 🔥)
# -------------------------------
st.subheader("🚨 Alerts & Risk Monitoring")

agent_summary = weekly_df.groupby('Agent_Name')['DSAT_Count'].mean().reset_index()
agent_summary['Risk'] = (agent_summary['DSAT_Count'] - y.mean())/y.std()*10 + 50

high_risk_agents = agent_summary[agent_summary['Risk'] > 60]
moderate_agents = agent_summary[(agent_summary['Risk'] > 45) & (agent_summary['Risk'] <= 60)]

colA, colB = st.columns(2)

with colA:
    st.error(f"🔴 High Risk Agents: {len(high_risk_agents)}")
    st.dataframe(high_risk_agents.sort_values("Risk", ascending=False).head(5))

with colB:
    st.warning(f"🟡 Moderate Risk Agents: {len(moderate_agents)}")
    st.dataframe(moderate_agents.sort_values("Risk", ascending=False).head(5))

# -------------------------------
# AGENT SELECTION
# -------------------------------
agent = st.selectbox("Select Agent", weekly_df['Agent_Name'].unique())

agent_data = weekly_df[weekly_df['Agent_Name']==agent].sort_values('Week')
latest = agent_data.iloc[-1]

prediction = model.predict([latest[features]])[0]
risk = (prediction - y.mean())/y.std()*10 + 50

# -------------------------------
# AGENT ALERT
# -------------------------------
if risk > 60:
    st.error("🚨 ALERT: High DSAT Risk – Immediate action required")
elif risk > 45:
    st.warning("⚠️ Warning: Moderate Risk – Monitor closely")
else:
    st.success("✅ Stable Performance")

# -------------------------------
# KPI
# -------------------------------
st.metric("Predicted DSAT", int(prediction))
st.metric("Risk Score", int(risk))

# -------------------------------
# TREND ALERT
# -------------------------------
trend = latest['DSAT_lag_1'] - latest['DSAT_lag_4']

if trend > 0:
    st.warning("📈 DSAT is increasing")
elif trend < 0:
    st.success("📉 DSAT is improving")
else:
    st.info("➖ DSAT is stable")

st.line_chart(agent_data.set_index('Week')['DSAT_Count'])

# -------------------------------
# ISSUE ANALYSIS
# -------------------------------
agent_comments = df[df['Agent_Name']==agent]
dsat_comments = agent_comments[agent_comments['DSAT']==1]['Customer_Comment']

X_test_agent = vectorizer.transform(dsat_comments)
predicted_issues = issue_model.predict(X_test_agent)

issue_df = pd.DataFrame(predicted_issues, columns=["Issue"])
issue_df = issue_df.value_counts().reset_index(name="Count")

st.subheader("📊 Issue Breakdown")
st.dataframe(issue_df)

# -------------------------------
# AI INSIGHT
# -------------------------------
def generate_insight(agent, pred, risk, issue_df, trend):

    top_issue = issue_df.iloc[0]["Issue"]

    if risk < 45:
        level = "strong"
    elif risk < 60:
        level = "moderate"
    else:
        level = "critical"

    return f"""
### 🤖 AI Insight

Agent **{agent}** performance is **{level}**.

- Predicted DSAT: **{int(pred)}**
- Risk Score: **{int(risk)}**

### 🔍 Key Issue: {top_issue}

### 💡 Recommendation:
Focus on improving **{top_issue}** related interactions to reduce DSAT.
"""

st.subheader("🤖 AI Insight")

st.markdown(generate_insight(agent,prediction,risk,issue_df,trend))
