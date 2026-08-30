import os, requests
from dotenv import load_dotenv
load_dotenv()

def generate_summary(context: str, question: str) -> str:
    provider=os.getenv('LLM_PROVIDER','fallback').lower()
    if provider == 'gemini' and os.getenv('GEMINI_API_KEY'):
        return _gemini(context, question)
    if provider == 'ollama':
        return _ollama(context, question)
    return fallback_summary(context, question)

def fallback_summary(context: str, question: str) -> str:
    return (
        f"Executive analysis for: {question}\n\n"
        "The agent reviewed the available financial evidence using SQL/ML tools. "
        "Use the ranked variance and anomaly results below to focus management attention on the largest cost drivers, "
        "then convert material issues into owners and due dates.\n\n" + context[:5000]
    )

def _gemini(context, question):
    key=os.environ['GEMINI_API_KEY']; model=os.getenv('GEMINI_MODEL','gemini-3.7-flash')
    url=f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}'
    prompt=f"You are an FP&A manager. Answer with evidence, caveats, and 3 prioritized actions. Question: {question}\nData: {context}"
    r=requests.post(url,json={'contents':[{'parts':[{'text':prompt}]}]},timeout=30)
    r.raise_for_status()
    return r.json()['candidates'][0]['content']['parts'][0]['text']

def _ollama(context, question):
    url=os.getenv('OLLAMA_URL','http://localhost:11434/api/generate')
    model=os.getenv('OLLAMA_MODEL','llama3.2')
    prompt=f"You are an FP&A manager. Answer with evidence and 3 prioritized actions. Question: {question}\nData: {context}"
    r=requests.post(url,json={'model':model,'prompt':prompt,'stream':False},timeout=60)
    r.raise_for_status(); return r.json().get('response','')
