import csv
from datetime import datetime
from gold_standard import GOLD_STANDARD
from eval_utils import extract_frequent_questions, llm_self_eval
from graph_agent import chat_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.evaluation.qa import QAEvalChain
import pandas as pd

def main():
    frequent_df = extract_frequent_questions()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
    qa_eval_chain = QAEvalChain.from_llm(llm)

    with open("eval_results.csv", mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "query", "expected_answer", "ai_answer", "eval_result", "comment", "self_eval"])

        for _, row in frequent_df.iterrows():
            query = row["query"]
            expected = next((g["answer"] for g in GOLD_STANDARD if g["query"] == query), "N/A")
            ai_answer = chat_agent.invoke({"query": query})["answer"]

            if expected != "N/A":
                eval_res = qa_eval_chain.evaluate(
                    [{"query": query, "answer": expected}],
                    [{"query": query, "answer": ai_answer}]
                )
                result = eval_res[0].get("results", ["UNCERTAIN"])[0]
                comment = eval_res[0].get("reasoning", "")
            else:
                result = "NO_GOLD"
                comment = "ゴールドスタンダードが存在しない質問"

            self_eval = llm_self_eval(ai_answer) if result == "NO_GOLD" else ""

            writer.writerow([
                datetime.now().isoformat(),
                query,
                expected,
                ai_answer,
                result,
                comment,
                self_eval
            ])

if __name__ == "__main__":
    main()
