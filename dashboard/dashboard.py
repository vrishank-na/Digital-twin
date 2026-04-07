import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🚀 Astronaut Health Digital Twin")

# Load dataset
data = pd.read_csv("astronaut_health.csv")

st.subheader("Astronaut Health Data")
st.write(data)

# Heart Rate Graph
st.subheader("Heart Rate Over Time")
fig1 = px.line(data, x="time", y="heart_rate", title="Heart Rate")
st.plotly_chart(fig1)

# Fatigue Graph
st.subheader("Fatigue Index")
fig2 = px.line(data, x="time", y="fatigue", title="Fatigue")
st.plotly_chart(fig2)

# Sleep Quality Graph
st.subheader("Sleep Quality")
fig3 = px.line(data, x="time", y="sleep_quality", title="Sleep Quality")
st.plotly_chart(fig3)

# Event markers
st.subheader("Mission Events")

events = data[data["event"] != "None"]

st.write(events)