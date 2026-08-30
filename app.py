import streamlit as st
from src.generate_data import *  # noqa
from src.agent import FinSightAgent
from src.tools import variance_analysis, anomaly_detection, revenue_forecast, variance_chart

st.set_page_config(page_title='FinSight AI', layout='wide')
st.title('FinSight AI — Agentic FP&A & ERP Analytics Copilot')
st.caption('Synthetic data • SQL + Python + ML + visualization + optional LLM')

if not __import__('pathlib').Path('data/finsight.db').exists():
    import src.generate_data as gd

agent=FinSightAgent()
question=st.text_input('Ask a finance / management question', 'Which cost drivers need management attention?')
if st.button('Run agent', type='primary'):
    with st.spinner('Planning → querying → analyzing → synthesizing...'):
        result=agent.run(question)
    st.subheader('Agent plan')
    st.write(' → '.join(result['plan']))
    st.subheader('Executive summary')
    st.write(result['summary'])
    if result['actions'] is not None:
        st.subheader('PM action backlog')
        st.dataframe(result['actions'], use_container_width=True)
    with st.expander('Evidence used by the agent'):
        for e in result['evidence']: st.code(e)

st.divider(); st.subheader('Analytics dashboard')
col1,col2=st.columns(2)
with col1: st.plotly_chart(variance_chart(), use_container_width=True)
with col2:
    st.write('Detected anomalies')
    st.dataframe(anomaly_detection().head(10), use_container_width=True)
st.write('3-month revenue forecast')
st.dataframe(revenue_forecast(3), use_container_width=True)
