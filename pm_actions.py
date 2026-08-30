import pandas as pd

def create_actions(variance_df: pd.DataFrame, max_actions=5):
    actions=[]
    for _,r in variance_df.head(max_actions).iterrows():
        pct=abs(float(r['variance_pct']))
        severity='P0' if pct>=0.20 else ('P1' if pct>=0.10 else 'P2')
        direction='overspend' if r['variance']>0 else 'underspend'
        actions.append({
            'priority':severity,
            'action':f"Investigate {r['cost_category']} {direction} in {r['business_unit']} and validate root cause.",
            'owner':'Finance Business Partner',
            'impact':f"₹/USD-equivalent {abs(float(r['variance'])):,.0f} variance",
            'due_in_days':3 if severity=='P0' else 7 if severity=='P1' else 14
        })
    return pd.DataFrame(actions)
