from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import plotly.express as px

DB = Path(__file__).resolve().parents[1] / 'data' / 'finsight.db'

def query_df(sql: str, params=None):
    con = sqlite3.connect(DB)
    try:
        return pd.read_sql_query(sql, con, params=params or {})
    finally:
        con.close()

def kpi_summary(month=None):
    where = ''
    params = {}
    if month:
        where = 'WHERE month = :month'
        params['month'] = month
    return query_df(f'''
        SELECT month, business_unit,
               SUM(revenue) AS revenue,
               SUM(budget) AS budget,
               SUM(actual) AS actual,
               SUM(actual-budget) AS variance,
               AVG((actual-budget)/NULLIF(budget,0)) AS avg_variance_pct
        FROM finance_transactions {where}
        GROUP BY month, business_unit
        ORDER BY month, business_unit
    ''', params)

def variance_analysis(limit=10):
    return query_df('''
        SELECT business_unit, cost_category,
               ROUND(SUM(budget),2) AS budget,
               ROUND(SUM(actual),2) AS actual,
               ROUND(SUM(actual-budget),2) AS variance,
               ROUND(100.0*SUM(actual-budget)/NULLIF(SUM(budget),0),2) AS variance_pct
        FROM finance_transactions
        GROUP BY business_unit, cost_category
        ORDER BY ABS(SUM(actual-budget)) DESC
        LIMIT ?
    ''', [limit])

def anomaly_detection():
    df = query_df('''
        SELECT month, business_unit, cost_category,
               SUM(actual) AS actual, SUM(budget) AS budget,
               SUM(actual-budget) AS variance
        FROM finance_transactions
        GROUP BY month, business_unit, cost_category
    ''')
    X = df[['actual','budget','variance']].fillna(0)
    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = model.fit_predict(X)
    return df[df['anomaly'] == -1].sort_values('variance', ascending=False)

def revenue_forecast(months_ahead=3):
    df = query_df('''
        SELECT month, business_unit, SUM(revenue) AS revenue
        FROM finance_transactions
        GROUP BY month, business_unit
        ORDER BY month
    ''')
    outputs=[]
    for bu, g in df.groupby('business_unit'):
        g=g.copy(); g['t']=np.arange(len(g))
        model=LinearRegression().fit(g[['t']],g['revenue'])
        future=np.arange(len(g),len(g)+months_ahead)
        pred=model.predict(future.reshape(-1,1))
        for i,p in zip(range(1,months_ahead+1),pred): outputs.append([bu,i,float(p)])
    return pd.DataFrame(outputs, columns=['business_unit','months_ahead','forecast_revenue'])

def variance_chart():
    df=variance_analysis(20)
    return px.bar(df.head(12), x='variance', y='cost_category', color='business_unit', orientation='h', title='Largest budget variances')
