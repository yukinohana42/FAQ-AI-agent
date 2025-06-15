"""
graph_agent.py – LangGraph + ingest_and_chat 安定実装 (最終完成版)
"""
from __future__ import annotations
import os
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph
from ingest_and_chat import build_index, agent_search_and_pick, make_summary
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

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
if not os.path.exists("faiss_index"):
    print("インデックスが見つからないため、新規に作成します...")
    build_index()
    print("インデックスの作成が完了しました。")

VECDB: FAISS = FAISS.load_local(
    "faiss_index",
    EMBED,
    allow_dangerous_deserialization=True,
)

# ── Node & Router Function ----------------------------------
def router(state: GraphState) -> str:
    """ユーザーの質問内容に応じて、RAG検索を行うか、直接LLMに回答させるかを判断する"""
    q = state["query"]
    need = LLM.invoke(
        f"ユーザーの質問に答えるために、データベース検索が必要ですか？ 質問： 「{q}」\n検索が必要な場合は 'Yes'、不要な場合は 'No' とだけ答えてください。"
    ).content.strip().lower()
    
    decision = "rag" if need.startswith('y') else "direct_answer"
    print(f"  [Router] 判断: {decision}")
    return decision

def rag(state: GraphState) -> dict:
    """データベースを検索し、関連チャンクを取得する"""
    print("  [Node] RAG検索を実行中...")
    q = state["query"]
    hits, kw = agent_search_and_pick(VECDB, q)
    return {"hits": hits, "keywords": kw}

def summarize(state: GraphState) -> dict:
    """取得した情報または質問を元に、最終的な回答を生成する"""
    q = state["query"]
    hits     = state.get("hits")
    keywords = state.get("keywords")
    
    if hits:
        print("  [Node] 検索結果を元に回答を生成中...")
        answer = make_summary(hits, q, keywords)
    else:
        print("  [Node] 検索結果がないため、直接回答を生成中...")
        answer = LLM.invoke(f"200字以内で、医師として専門的かつ断定的に答えて下さい: {q}").content
        
    return {"answer": answer}

# ── Graph -------------------------------------------------
# グラフの状態として、上で定義したGraphStateクラスを指定
graph = StateGraph(GraphState)

graph.add_node("rag",       rag)
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
            
            # invokeに渡す辞書のキーをGraphStateのキーと一致させる
            res = chat_agent.invoke({"query": q})
            
            print("\n✅ 回答\n" + "="*20)
            print(res["answer"].strip())
            print("="*20 + "\n")
            
    except (EOFError, KeyboardInterrupt):
        print("\nエージェントを終了します。")
        pass