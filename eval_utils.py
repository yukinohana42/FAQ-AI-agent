import csv
from datetime import datetime
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.evaluation.qa import QAEvalChain

def log_user_question(query: str):
    with open("user_questions_log.csv", "a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().isoformat(), query])

def extract_frequent_questions(filepath="user_questions_log.csv", top_n=10):
    df = pd.read_csv(filepath, header=None, names=["timestamp", "query"])
    freq_df = df["query"].value_counts().head(top_n).reset_index()
    freq_df.columns = ["query", "count"]
    freq_df.to_csv("frequent_questions.csv", index=False)
    return freq_df

def llm_self_eval(ai_answer):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
    prompt = f"次の回答は信頼できる正確な医療情報ですか？ 回答:「{ai_answer}」 含まれる場合はYes、含まれない場合はNoとだけ答えてください。"
    resp = llm.invoke(prompt)
    decision = resp.content.strip().lower()
    print(f"[LLM Self Eval] {decision}")
    return decision
