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
# ISSUE MODEL (PURE ML)
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
# 📊 MODEL ACCURACY (RENAMED)
# -------------------------------
agent_actual = agent_data['DSAT_Count']
agent_pred = model.predict(agent_data[features])

mae = mean_absolute_error(agent_actual, agent_pred)
r2 = r2_score(agent_actual, agent_pred)

st.subheader("📊 Model Accuracy")

c1, c2, c3 = st.columns(3)
c1.metric("Issue Detection Accuracy", f"{round(nlp_accuracy*100,1)}%")
c2.metric("Prediction Reliability (R²)", round(r2,2))
c3.metric("Avg Prediction Error", round(mae,2))

# -------------------------------
# 🚨 RISK SEGMENTATION
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
# 📈 WEEKLY DSAT TREND
# -------------------------------
st.subheader("📈 Weekly DSAT Trend")
st.line_chart(agent_data.set_index('Week')['DSAT_Count'])

# -------------------------------
# 🎯 CASE HANDLING + PREDICTION
# -------------------------------
st.subheader("🎯 Case Handling & Prediction")

col4, col5 = st.columns(2)
col4.metric("Predicted DSAT (Next Week)", int(prediction))
col5.metric("Risk Score", int(risk))

# -------------------------------
# 📊 ISSUE BREAKDOWN
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
# 🤖 AI INSIGHT + ROOT CAUSE + ACTIONS
# -------------------------------
top_issue = issue_df.sort_values("Count", ascending=False).iloc[0]["Issue"]

st.subheader("🤖 AI Insight")

if risk < 45:
    st.markdown(f"""
### ✅ Strong Performance

Agent **{agent}** is performing well with low DSAT risk.

### 💡 Strengths
- Consistent delivery
- Good customer handling

👉 Maintain current performance.
""")

else:

    if top_issue == "Communication":
        root = "Customer dissatisfaction is driven by communication issues."
        action = """
- Improve empathy and tone  
- Avoid scripted responses  
- Listen actively  
"""
    elif top_issue == "Process":
        root = "Delays and inefficient handling are driving DSAT."
        action = """
- Reduce wait time  
- Avoid multiple transfers  
- Take ownership  
"""
    else:
        root = "Product-related issues are impacting resolution quality."
        action = """
- Improve product knowledge  
- Escalate faster  
"""

    st.markdown(f"""
### ⚠️ Performance Needs Attention

- Predicted DSAT: **{int(prediction)}**
- Risk Score: **{int(risk)}**

---

### 🔍 Root Cause
{root}

---

### 💡 Actions
{action}
""")
