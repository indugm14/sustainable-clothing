
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.title("Sustainable Fashion Analytics")

df = pd.read_csv("data.csv")

st.write("Dataset Preview", df.head())

# Encode
df_enc = df.copy()
le = LabelEncoder()
for col in ['age_group','gender','income']:
    df_enc[col] = le.fit_transform(df_enc[col])

X = df_enc.drop('interested', axis=1)
y = df_enc['interested']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestClassifier()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

st.subheader("Model Performance")
st.write("Accuracy:", accuracy_score(y_test,y_pred))
st.write("Precision:", precision_score(y_test,y_pred))
st.write("Recall:", recall_score(y_test,y_pred))
st.write("F1 Score:", f1_score(y_test,y_pred))

# Upload new data
st.subheader("Upload New Data")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    new_df = pd.read_csv(uploaded)
    for col in ['age_group','gender','income']:
        new_df[col] = le.fit_transform(new_df[col])
    preds = model.predict(new_df)
    new_df['prediction'] = preds
    st.write(new_df)
