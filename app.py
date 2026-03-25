import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from textblob import TextBlob

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="DSAT Intelligence", layout="wide")
st.title("🚀 DSAT Intelligence Dashboard")

# -------------------------------
# SAFE DATA LOADING (FIXED)
# -------------------------------
file_path = "bpo_customer_experience_dataset.csv"

if not os.path.exists(file_path):
    st.error("❌ Dataset file not found. Upload correct CSV.")
    st.stop()

try:
    df = pd.read_csv(file_path)
except:
    try:
        df = pd.read_csv(file_path, sep="|")
    except:
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except Exception as e:
            st.error("❌ Unable to read dataset")
            st.text(str(e))
            st.stop()

df.columns = df.columns.str.strip()

if df.empty:
    st.error("❌ Dataset is empty")
    st.stop()

# Preview (for debug)
st.success(f"✅ Dataset Loaded: {df.shape[0]} rows")

# -------------------------------
# BASIC CLEANING
# -------------------------------
df['Week'] = pd.to_datetime(df['Week'], errors='coerce')
df = df.dropna(subset=['Week'])

df['DSAT'] = df['Customer_Effortless'].apply(lambda x: 1 if str(x).lower() == "no" else 0)
df['Sentiment'] = df['Customer_Comment'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# -------------------------------
# LABEL FUNCTION
# -------------------------------
def label_issue(comment):
    c = str(comment).lower()

    if any(w in c for w in ["rude","unhelpful","confusing","bad"]):
        return "Communication"
    elif any(w in c for w in ["delay","wait","slow","long"]):
        return "Process"
    elif any(w in c for w in ["error","bug","failed","not working"]):
        return "Product"
    else:
        return "Other"

df['Issue_Label'] = df['Customer_Comment'].apply(label_issue)

# -------------------------------
# NLP MODEL (STABLE VERSION)
# -------------------------------
df_clean = df[df['Issue_Label'] != "Other"].copy()

if len(df_clean) < 50:
    st.warning("⚠️ Not enough data for NLP model")
    nlp_accuracy = 0.0
else:
    df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)

    split = int(len(df_clean) * 0.6)
    train_sample = df_clean.iloc[:split]
    test_sample = df_clean.iloc[split:]

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1,2),
        max_features=3000
    )

    X_train = vectorizer.fit_transform(train_sample['Customer_Comment'])
    y_train = train_sample['Issue_Label']

    X_test = vectorizer.transform(test_sample['Customer_Comment'])
    y_test = test_sample['Issue_Label']

    if len(set(y_train)) < 2:
        nlp_accuracy = 0.0
    else:
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
# ACCURACY UI
# -------------------------------
st.subheader("📊 System Accuracy Overview")

c1,c2,c3 = st.columns(3)
c1.metric("Issue Detection Accuracy", f"{round(nlp_accuracy*100,1)}%")
c2.metric("Prediction Reliability (R²)", round(r2,2))
c3.metric("Avg Prediction Error", round(mae,2))

st.caption("Accuracy depends on dataset quality")

# -------------------------------
# ALERT SYSTEM
# -------------------------------
st.subheader("🚨 Alerts & Risk Monitoring")

agent_summary = weekly_df.groupby('Agent_Name')['DSAT_Count'].mean().reset_index()
agent_summary['Risk'] = (agent_summary['DSAT_Count'] - y.mean())/y.std()*10 + 50

high = agent_summary[agent_summary['Risk'] > 60]
moderate = agent_summary[(agent_summary['Risk'] > 45) & (agent_summary['Risk'] <= 60)]

colA,colB = st.columns(2)

with colA:
    st.error(f"High Risk Agents: {len(high)}")
    st.dataframe(high.head())

with colB:
    st.warning(f"Moderate Risk Agents: {len(moderate)}")
    st.dataframe(moderate.head())

# -------------------------------
# LOW RISK
# -------------------------------
st.subheader("🟢 Low Risk Agents")
st.dataframe(agent_summary.sort_values("Risk").head(10))

# -------------------------------
# AGENT VIEW
# -------------------------------
agent = st.selectbox("Select Agent", weekly_df['Agent_Name'].unique())

agent_data = weekly_df[weekly_df['Agent_Name']==agent].sort_values('Week')
latest = agent_data.iloc[-1]

prediction = model.predict([latest[features]])[0]
risk = (prediction - y.mean())/y.std()*10 + 50

st.metric("Predicted DSAT", int(prediction))
st.metric("Risk Score", int(risk))

st.line_chart(agent_data.set_index('Week')['DSAT_Count'])

# -------------------------------
# ISSUE ANALYSIS
# -------------------------------
agent_comments = df[df['Agent_Name']==agent]
dsat_comments = agent_comments[agent_comments['DSAT']==1]['Customer_Comment']

if len(dsat_comments) > 0 and 'issue_model' in globals():
    X_test_agent = vectorizer.transform(dsat_comments)
    pred_issues = issue_model.predict(X_test_agent)

    issue_df = pd.DataFrame(pred_issues, columns=["Issue"])
    issue_df = issue_df.value_counts().reset_index(name="Count")
else:
    issue_df = pd.DataFrame({"Issue":["No Data"], "Count":[0]})

st.subheader("📊 Issue Breakdown")
st.dataframe(issue_df)

# -------------------------------
# AI INSIGHT (ONLY ONE)
# -------------------------------
def generate_insight(agent, pred, risk, issue_df):

    top_issue = issue_df.iloc[0]["Issue"]

    if risk < 45:
        level = "Good"
    elif risk < 60:
        level = "Moderate"
    else:
        level = "High Risk"

    return f"""
### 🤖 AI Insight

Agent **{agent}** is at **{level} performance**.

- Predicted DSAT: **{int(pred)}**
- Risk Score: **{int(risk)}**

### Key Issue:
**{top_issue}**

### Recommendation:
Focus on improving {top_issue} to reduce DSAT.
"""

st.subheader("🤖 AI Insight")
st.markdown(generate_insight(agent, prediction, risk, issue_df))
