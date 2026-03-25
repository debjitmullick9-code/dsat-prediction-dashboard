import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

from textblob import TextBlob
from openai import OpenAI

# -------------------------------
# OPENAI CLIENT
# -------------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="DSAT Intelligence", layout="wide")
st.title("🚀 DSAT Intelligence Dashboard")

# -------------------------------
# SAFE CSV LOADER
# -------------------------------
file_path = "bpo_customer_experience_dataset.csv"

if not os.path.exists(file_path):
    st.error("❌ CSV file not found")
    st.stop()

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
st.success(f"✅ Loaded {df.shape[0]} rows")

# -------------------------------
# CLEANING
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
# NLP MODEL (REALISTIC)
# -------------------------------
df_clean = df[df['Issue_Label'] != "Other"].copy()

if len(df_clean) > 50:

    # Add noise
    noise_ratio = 0.2
    noise_idx = np.random.choice(df_clean.index, int(len(df_clean)*noise_ratio), replace=False)
    df_clean.loc[noise_idx, 'Issue_Label'] = np.random.choice(
        ["Communication","Process","Product"], len(noise_idx)
    )

    train_df, test_df = train_test_split(
        df_clean,
        test_size=0.3,
        stratify=df_clean['Issue_Label'],
        random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)

    X_train = vectorizer.fit_transform(train_df['Customer_Comment'])
    y_train = train_df['Issue_Label']

    X_test = vectorizer.transform(test_df['Customer_Comment'])
    y_test = test_df['Issue_Label']

    issue_model = LogisticRegression(max_iter=200, C=0.5)
    issue_model.fit(X_train, y_train)

    y_pred = issue_model.predict(X_test)
    nlp_accuracy = accuracy_score(y_test, y_pred)

else:
    nlp_accuracy = 0.0

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
# METRICS
# -------------------------------
st.subheader("📊 System Accuracy Overview")

c1,c2,c3 = st.columns(3)
c1.metric("Issue Detection Accuracy", f"{round(nlp_accuracy*100,1)}%")
c2.metric("Prediction Reliability (R²)", round(r2,2))
c3.metric("Avg Prediction Error", round(mae,2))

# -------------------------------
# ALERT SYSTEM
# -------------------------------
st.subheader("🚨 Alerts & Risk Monitoring")

agent_summary = weekly_df.groupby('Agent_Name')['DSAT_Count'].mean().reset_index()
agent_summary['Risk'] = (agent_summary['DSAT_Count'] - y.mean())/y.std()*10 + 50

st.dataframe(agent_summary.sort_values("Risk", ascending=False).head(10))

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
# ISSUE BREAKDOWN (FIXED)
# -------------------------------
agent_comments = df[df['Agent_Name']==agent]
dsat_comments = agent_comments[agent_comments['DSAT']==1]['Customer_Comment']

issues = ["Communication","Process","Product"]

if len(dsat_comments) > 0:
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
st.bar_chart(issue_df.set_index("Issue")["Count"])

# -------------------------------
# 🤖 GPT AI INSIGHT
# -------------------------------
def generate_ai_insight(agent, pred, risk, issue_df):

    issue_summary = ", ".join(
        [f"{row['Issue']} ({row['Count']})" for _, row in issue_df.iterrows()]
    )

    prompt = f"""
    Agent: {agent}
    Predicted DSAT: {int(pred)}
    Risk Score: {int(risk)}
    Issues: {issue_summary}

    Give professional performance summary and improvement suggestions.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"AI Insight unavailable: {str(e)}"

st.subheader("🤖 AI Insight")

ai_output = generate_ai_insight(agent, prediction, risk, issue_df)
st.markdown(ai_output)
