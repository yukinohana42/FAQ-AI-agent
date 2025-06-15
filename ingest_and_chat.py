# ingest_and_chat.py  ── “回答” だけを返す簡潔版（LangChain/LangGraph最新系対応）
from dotenv import load_dotenv
load_dotenv()

import os, glob, re, textwrap
from collections import Counter

try:
    from termcolor import colored
except ModuleNotFoundError:
    def colored(t, *_a, **_k): return t

# 最新: RecursiveCharacterTextSplitter は専用パッケージに分離
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

# 設定
INDEX_PATH   = "faiss_index"
DOCS_DIR     = "docs"
CHUNK_SIZE   = 500
CHUNK_OVERLP = 350
NEARBY_RANGE = 2

EMBED_MODEL  = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
LLM          = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
DEBUG = False

MED_SYNONYMS = {
    "ノルアドレナリン": ["ノルアドレナリン", "ノルアド", "NAd", "noradrenaline", "NA"],
}

def auto_expand_keywords(keywords: list[str]) -> list[str]:
    joined = ", ".join(keywords)
    prompt = (
        f"ユーザーの検索キーワード: {joined}\n"
        "同義語・医薬品名・略語・関連語を日本語/英語で最大10個カンマ区切りで列挙してください。"
    )
    resp = LLM.invoke(prompt)
    return list({kw.strip() for kw in (keywords + resp.content.split(",")) if kw.strip()})

def extract_main_keywords(query: str) -> list[str]:
    tokens = re.findall(r"[一-龥ぁ-んァ-ンa-zA-Z0-9\-＋・.]+", query)
    kws: list[str] = []
    for w in tokens:
        hit = next((syns for syns in MED_SYNONYMS.values() if w in syns), None)
        kws.extend(hit if hit else [w])
    kws = list(set(kws))
    expanded = auto_expand_keywords(kws)
    if DEBUG:
        print(colored(f"🔑 展開キーワード: {expanded}", "yellow"))
    return expanded

def build_index() -> None:
    docx_files = glob.glob(os.path.join(DOCS_DIR, "*.docx"))
    if not docx_files:
        raise FileNotFoundError("./docs に .docx が見つかりません")

    raw_docs = []
    for p in docx_files:
        for i, d in enumerate(UnstructuredWordDocumentLoader(p).load()):
            d.metadata.update(source=os.path.basename(p), chunk_id=i)
            raw_docs.append(d)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLP
    )
    chunks = splitter.split_documents(raw_docs)
    if DEBUG:
        print(colored(f"🔹 チャンク数: {len(chunks)}", "cyan"))

    vecdb = None
    batch: list = []
    for idx, doc in enumerate(chunks, 1):
        batch.append(doc)
        if len(batch) == 100 or idx == len(chunks):
            vecdb = FAISS.from_documents(batch, EMBED_MODEL) if vecdb is None else vecdb.add_documents(batch)
            batch = []
    vecdb.save_local(INDEX_PATH)
    if DEBUG:
        print(colored(f"✅ インデックス保存 ({INDEX_PATH})", "green"))

def embedding_search(vecdb, query, k=30):
    return vecdb.similarity_search(query, k=k)

def keyword_score(doc, keywords):
    return sum(len(re.findall(re.escape(kw), doc.page_content, re.I)) for kw in keywords)

def keyword_search_multi(vecdb, keywords, k=30):
    docs = vecdb.docstore._dict.values()
    hits = [
        d for d in docs if any(re.search(re.escape(kw), d.page_content, re.I) for kw in keywords)
    ]
    hits.sort(key=lambda d: keyword_score(d, keywords), reverse=True)
    return hits[:k]

def filter_by_keywords(docs, keywords):
    if len(keywords) >= 2:
        all_hit = [d for d in docs if all(kw.lower() in d.page_content.lower() for kw in keywords)]
        if all_hit:
            return all_hit
    return [d for d in docs if any(kw.lower() in d.page_content.lower() for kw in keywords)]

def add_nearby_chunks(vecdb, selected):
    results = list(selected)
    done = {(d.metadata["source"], d.metadata["chunk_id"]) for d in selected}

    for d in selected:
        sid, cid = d.metadata["source"], d.metadata["chunk_id"]
        for nid in range(cid - NEARBY_RANGE, cid + NEARBY_RANGE + 1):
            if nid < 0 or (sid, nid) in done:
                continue
            maybe = next(
                (
                    cand
                    for cand in vecdb.docstore._dict.values()
                    if cand.metadata.get("source") == sid and cand.metadata.get("chunk_id") == nid
                ),
                None,
            )
            if maybe:
                results.append(maybe)
                done.add((sid, nid))
    return results

def make_summary(results, query, keywords):
    uniq_text = "\n---\n".join({d.page_content for d in results})
    prompt = f"""
あなたは救急・総合内科のベテラン医師です。
ユーザー質問: 「{query}」

## 執筆指示
1. 以下チャンクから関連情報を抽出し、臨床的推論で補完して **断定的** にまとめる。
2. 薬剤名・用量・投与経路・タイミングなど必須ポイントは **明確に列挙**。
3. 情報が不足していても、標準治療を根拠に”医師としての最適解”を示す。
4. 不要な前置きや一般論は書かない。

--- 参考 ---
{uniq_text}
--- ここまで ---
"""
    return LLM.invoke(prompt).content.strip()

def agent_search_and_pick(vecdb, query, max_retry=3, k=40):
    keywords = extract_main_keywords(query)

    for _ in range(max_retry):
        emb = embedding_search(vecdb, query, k=k)
        cand = add_nearby_chunks(vecdb, filter_by_keywords(emb, keywords))
        if cand:
            return cand, keywords
        query = LLM.invoke(f"「{query}」を日本語で別表現 1 つ").content.strip()

    kw_hits = add_nearby_chunks(vecdb, keyword_search_multi(vecdb, keywords, k=k))
    return kw_hits, keywords
