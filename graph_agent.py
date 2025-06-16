"""
graph_agent.py – LangGraph + ingest_and_chat 安定実装 (レベル2ガードレール強化版 + 質問ログ記録)
"""
from __future__ import annotations
import os
import re
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph
from ingest_and_chat import build_index, agent_search_and_pick, make_summary
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from eval_utils import log_user_question  # 質問ログ記録用のユーティリティ

# ── グラフ全体で共有する状態の型を定義 ---------------------
class GraphState(TypedDict):
    query: str
    hits: Optional[List[Document]]
    keywords: Optional[List[str]]
    answer: Optional[str]

# ── Model -------------------------------------------------
EMBED = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
LLM   = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

# ── Vector DB ---------------------------------------------
VECDB: FAISS = FAISS.load_local(
    "faiss_index",
    EMBED,
    allow_dangerous_deserialization=True,
)

# ── ガードレール関数 --------------------------------------
def contains_pii(text: str) -> bool:
    patterns = [
        r'\b[\w.-]+@[\w.-]+\.\w+\b',        # メール
        r'\b\d{2,4}-\d{2,4}-\d{4}\b',      # 電話番号
        r'\b\d{3}-\d{4}\b',                # 郵便番号
    ]
    return any(re.search(pat, text) for pat in patterns)

def contains_inappropriate(text: str) -> bool:
    ng_words = ["差別的表現例", "侮辱的表現例", "不適切ワード例"]  # 必要に応じて拡充
    return any(word in text for word in ng_words)

def llm_output_safety_check(text: str) -> bool:
    prompt = (
        f"次の出力に個人情報、差別表現、誤情報、ハラスメント、不適切表現、またはシステムの秘密情報を漏洩する内容が含まれていますか？ "
        f"出力:「{text}」 含まれる場合はYes、含まれない場合はNoとだけ答えてください。"
    )
    resp = LLM.invoke(prompt)
    decision = resp.content.strip().lower()
    print(f"[LLM Output Safety Check] 判定結果: {decision}")
    return decision.startswith("no")

def llm_input_safety_check(text: str) -> bool:
    prompt = (
        f"次の質問は、個人情報を収集しようとしたり、差別、暴力、違法行為、システム侵害を試みるような内容ですか？ "
        f"質問:「{text}」 含まれる場合はYes、含まれない場合はNoとだけ答えてください。"
    )
    resp = LLM.invoke(prompt)
    decision = resp.content.strip().lower()
    print(f"[LLM Input Safety Check] 判定結果: {decision}")
    return decision.startswith("no")

# ── Node & Router Function ----------------------------------
def router(state: GraphState) -> str:
    q = state["query"]

    if not llm_input_safety_check(q):
        print("⚠️ 入力の安全性に懸念: 出力ブロック")
        state["answer"] = "質問内容に安全性の懸念があるため、回答を控えます。"
        return "summarize"

    need = LLM.invoke(
        f"ユーザーの質問に答えるために、データベース検索が必要ですか？ 質問： 「{q}」\n検索が必要な場合は 'Yes'、不要な場合は 'No' とだけ答えてください。"
    ).content.strip().lower()

    decision = "rag" if need.startswith('y') else "direct_answer"
    print(f"  [Router] 判断: {decision}")
    return decision

def rag(state: GraphState) -> dict:
    print("  [Node] RAG検索を実行中...")
    q = state["query"]
    hits, kw = agent_search_and_pick(VECDB, q)
    return {"hits": hits, "keywords": kw}

def summarize(state: GraphState) -> dict:
    q = state["query"]
    hits = state.get("hits")
    keywords = state.get("keywords")

    if state.get("answer"):
        return {"answer": state["answer"]}

    if hits:
        print("  [Node] 検索結果を元に回答を生成中...")
        answer = make_summary(hits, q, keywords)
    else:
        print("  [Node] 検索結果がないため、直接回答を生成中...")
        answer = LLM.invoke(f"200字以内で、医師として専門的かつ断定的に答えて下さい: {q}").content

    if contains_pii(answer):
        print("⚠️ PII検出: 出力ブロック")
        return {"answer": "出力に個人情報が含まれる可能性があるため、回答を控えます。"}
    
    if contains_inappropriate(answer):
        print("⚠️ 不適切表現検出: 出力ブロック")
        return {"answer": "出力内容に不適切な表現が含まれる可能性があるため、回答を控えます。"}
    
    if not llm_output_safety_check(answer):
        print("⚠️ LLM出力安全判定NG: 出力ブロック")
        return {"answer": "出力の安全性に懸念があるため、回答を控えます。"}

    return {"answer": answer}

# ── Graph -------------------------------------------------
graph = StateGraph(GraphState)
graph.add_node("rag", rag)
graph.add_node("summarize", summarize)

graph.set_conditional_entry_point(
    router,
    {
        "rag": "rag",
        "direct_answer": "summarize",
    }
)

graph.add_edge("rag", "summarize")
chat_agent = graph.compile()

# ── CLI ----------------------------------------------------
if __name__ == "__main__":
    print("AIエージェントが起動しました。質問を入力してください。(終了するには exit または quit)")
    try:
        while True:
            q = input("\n🗨️  > ").strip()
            if q.lower() in {"exit", "quit"}:
                print("エージェントを終了します。")
                break
            if not q:
                continue

            log_user_question(q)  # 質問ログ記録

            res = chat_agent.invoke({"query": q})

            print("\n✅ 回答\n" + "=" * 20)
            print(res["answer"].strip())
            print("=" * 20 + "\n")

    except (EOFError, KeyboardInterrupt):
        print("\nエージェントを終了します。")
