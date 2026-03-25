import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_absolute_error, r2_score

from textblob import TextBlob

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="DSAT Intelligence", layout="wide")
st.title("🚀 DSAT Intelligence Dashboard (ML Powered)")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("bpo_customer_experience_dataset.csv")
df['Week'] = pd.to_datetime(df['Week'])

df['DSAT'] = df['Customer_Effortless'].apply(lambda x: 1 if x == "No" else 0)
df['Sentiment'] = df['Customer_Comment'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# -------------------------------
# AUTO LABEL FOR NLP MODEL
# -------------------------------
def label_issue(comment):
    c = str(comment).lower()
    if any(w in c for w in ["rude","unhelpful","confusing","agent"]):
        return "Communication"
    elif any(w in c for w in ["delay","long","wait","slow","time"]):
        return "Delay"
    elif any(w in c for w in ["not working","bug","broken","glitch"]):
        return "Product"
    else:
        return "Other"

df['Issue_Label'] = df['Customer_Comment'].apply(label_issue)

# -------------------------------
# NLP MODEL
# -------------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    max_features=5000
)

X_text = vectorizer.fit_transform(df['Customer_Comment'])
y_text = df['Issue_Label']

X_train, X_test, y_train, y_test = train_test_split(
    X_text, y_text, test_size=0.2, random_state=42
)

issue_model = LogisticRegression(max_iter=200)
issue_model.fit(X_train, y_train)

y_pred = issue_model.predict(X_test)

nlp_accuracy = accuracy_score(y_test, y_pred)

# -------------------------------
# TEAM SIMULATION
# -------------------------------
teams = ["Team A", "Team B", "Team C"]
df["Team"] = np.random.choice(teams, len(df))

# -------------------------------
# FILTERS
# -------------------------------
st.sidebar.header("🎛 Filters")

team = st.sidebar.selectbox("Team", ["All"] + teams)
date_range = st.sidebar.date_input(
    "Date Range",
    [df['Week'].min(), df['Week'].max()]
)

if team != "All":
    df = df[df["Team"] == team]

df = df[(df['Week'] >= pd.to_datetime(date_range[0])) &
        (df['Week'] <= pd.to_datetime(date_range[1]))]

# -------------------------------
# WEEKLY AGG
# -------------------------------
weekly_df = df.groupby(['Agent_Name','Week']).agg({
    'DSAT': 'sum',
    'Ticket_ID': 'count'
}).reset_index()

weekly_df.rename(columns={
    'DSAT': 'DSAT_Count',
    'Ticket_ID': 'Total_Tickets'
}, inplace=True)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
for i in range(1, 5):
    weekly_df[f'DSAT_lag_{i}'] = weekly_df.groupby('Agent_Name')['DSAT_Count'].shift(i)
    weekly_df[f'Tickets_lag_{i}'] = weekly_df.groupby('Agent_Name')['Total_Tickets'].shift(i)

weekly_df = weekly_df.dropna()

# -------------------------------
# DSAT MODEL
# -------------------------------
features = [f'DSAT_lag_{i}' for i in range(1,5)] + [f'Tickets_lag_{i}' for i in range(1,5)]

X = weekly_df[features]
y = weekly_df['DSAT_Count']

model = RandomForestRegressor()
model.fit(X, y)

preds = model.predict(X)

mae = mean_absolute_error(y, preds)
r2 = r2_score(y, preds)

# -------------------------------
# MODEL PERFORMANCE
# -------------------------------
st.subheader("📊 Model Performance")

st.write(f"NLP Accuracy: {round(nlp_accuracy,2)}")
st.write(f"DSAT Model R2 Score: {round(r2,2)}")
st.write(f"DSAT MAE: {round(mae,2)}")

# -------------------------------
# LEADERBOARD
# -------------------------------
agent_summary = weekly_df.groupby('Agent_Name')['DSAT_Count'].mean().reset_index()
agent_summary.rename(columns={'DSAT_Count': 'Avg_DSAT'}, inplace=True)

agent_summary['Risk'] = (agent_summary['Avg_DSAT'] - y.mean()) / y.std() * 10 + 50

st.subheader("🚨 High Risk Agents")
st.dataframe(agent_summary.sort_values("Risk", ascending=False).head(10))

st.subheader("🟢 Low Risk Agents")
st.dataframe(agent_summary.sort_values("Risk").head(10))

# -------------------------------
# AGENT SELECTION
# -------------------------------
agent = st.selectbox("Select Agent", weekly_df['Agent_Name'].unique())

agent_data = weekly_df[weekly_df['Agent_Name'] == agent].sort_values('Week')
latest = agent_data.iloc[-1]

prediction = model.predict([latest[features]])[0]
risk = (prediction - y.mean()) / y.std() * 10 + 50

# -------------------------------
# KPI
# -------------------------------
col1, col2 = st.columns(2)

col1.metric("Predicted DSAT", int(prediction))
col2.metric("Risk Score", int(risk))

# -------------------------------
# TREND
# -------------------------------
st.line_chart(agent_data.set_index('Week')['DSAT_Count'])

# -------------------------------
# SIMULATION
# -------------------------------
st.subheader("🔮 Simulation")

sim_dsat = st.slider("Last Week DSAT", 0, 20, int(latest['DSAT_lag_1']))
sim_tickets = st.slider("Last Week Tickets", 10, 150, int(latest['Tickets_lag_1']))

sim_input = latest[features].copy()
sim_input.iloc[0] = sim_dsat
sim_input.iloc[4] = sim_tickets

sim_pred = model.predict([sim_input])[0]

st.metric("Simulated DSAT", int(sim_pred))

# -------------------------------
# CX ANALYSIS (ML BASED)
# -------------------------------
agent_comments = df[df['Agent_Name'] == agent]
dsat_comments = agent_comments[agent_comments['DSAT'] == 1]['Customer_Comment']

X_test_agent = vectorizer.transform(dsat_comments)
predicted_issues = issue_model.predict(X_test_agent)

issue_df = pd.DataFrame(predicted_issues, columns=["Issue"])
issue_df = issue_df.value_counts().reset_index(name="Count")

total = issue_df["Count"].sum()
issue_df["Percentage"] = (issue_df["Count"]/total*100).round(1)

st.subheader("📊 Issue Breakdown (ML)")
st.dataframe(issue_df)
st.bar_chart(issue_df.set_index("Issue")["Count"])

# -------------------------------
# SMART INSIGHT
# -------------------------------
def generate_insight(agent, pred, risk, sentiment, issue_df, trend):

    top_issue = issue_df.iloc[0]["Issue"]

    if trend > 0:
        trend_msg = "increasing"
    elif trend < 0:
        trend_msg = "improving"
    else:
        trend_msg = "stable"

    if risk < 45:
        level = "strong"
    elif risk < 60:
        level = "moderate"
    else:
        level = "critical"

    return f"""
### 🤖 AI Insight

Agent **{agent}** performance is **{level}**.

- DSAT trend is **{trend_msg}**
- Predicted DSAT: **{int(pred)}**
- Risk Score: **{int(risk)}**

### 🔍 Root Cause:
Primary issue: **{top_issue}**

### 💡 Recommendation:
- Improve **{top_issue}**
- Monitor DSAT trend closely
- Reduce repeated complaints

### 🎯 Impact:
Fixing this can significantly reduce DSAT.
"""

trend = latest['DSAT_lag_1'] - latest['DSAT_lag_4']
sentiment = agent_comments['Sentiment'].mean()

st.subheader("🤖 AI Insight")

st.markdown(generate_insight(
    agent,
    prediction,
    risk,
    sentiment,
    issue_df,
    trend
))