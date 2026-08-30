from .tools import variance_analysis, anomaly_detection, revenue_forecast, kpi_summary
from .llm import generate_summary
from .pm_actions import create_actions

class FinSightAgent:
    """Lightweight tool-using agent: classify intent, call tools, synthesize, create actions."""
    def plan(self, question):
        q=question.lower()
        steps=[]
        if any(x in q for x in ['forecast','predict','next quarter','future']): steps.append('forecast')
        if any(x in q for x in ['anomal','unusual','outlier','spike']): steps.append('anomaly')
        if any(x in q for x in ['variance','budget','overspend','underspend','cost driver']): steps.append('variance')
        if not steps: steps.append('kpi')
        return steps

    def run(self, question):
        plan=self.plan(question); evidence=[]
        variance=None
        if 'variance' in plan:
            variance=variance_analysis(10); evidence.append('TOP VARIANCES\n'+variance.to_string(index=False))
        if 'anomaly' in plan:
            anomalies=anomaly_detection(); evidence.append('ANOMALIES\n'+anomalies.head(10).to_string(index=False))
        if 'forecast' in plan:
            forecast=revenue_forecast(3); evidence.append('FORECAST\n'+forecast.to_string(index=False))
        if 'kpi' in plan:
            kpi=kpi_summary(); evidence.append('KPI SAMPLE\n'+kpi.tail(12).to_string(index=False))
        context='\n\n'.join(evidence)
        summary=generate_summary(context,question)
        actions=create_actions(variance,5) if variance is not None else None
        return {'plan':plan,'evidence':evidence,'summary':summary,'actions':actions}
