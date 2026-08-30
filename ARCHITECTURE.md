# System Architecture

```text
User question
     |
     v
Agent planner / intent router
     |
     +--> SQL KPI tool ----------> SQLite ERP-style data
     |
     +--> Variance tool ----------> budget vs actual
     |
     +--> Anomaly tool -----------> Isolation Forest
     |
     +--> Forecast tool ----------> Linear Regression
     |
     +--> Visualization tool -----> Plotly
     |
     v
Evidence bundle
     |
     v
LLM adapter (Gemini / Ollama / fallback)
     |
     +--> Executive summary
     +--> PM action backlog
     |
     v
Streamlit dashboard
```

## Agent design principle
The model is not trusted to invent financial calculations. Deterministic Python/SQL tools compute the metrics; the LLM only synthesizes the returned evidence into business language and suggested actions.
