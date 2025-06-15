import streamlit as st
from graph_agent import chat_agent  # 私たちが作ったエージェントの頭脳をインポート

# --- ページの基本設定 ---
st.set_page_config(
    page_title="AI Q&A Agent",
    page_icon="🤖"
)

st.title("みんほす！Q&A powerd by AIエージェント 🤖")
st.caption("みんほす！のこれまでの全資料のみに基づいて、AIが質問に回答します。")

# --- チャット履歴の初期化 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- チャット履歴の表示 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- ユーザーからの入力を受け取る ---
if prompt := st.chat_input("質問を入力してください..."):
    # ユーザーのメッセージを履歴に追加して表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # アシスタント（AI）の応答を生成
    with st.chat_message("assistant"):
        # クルクル回る「考え中」アイコンを表示
        with st.spinner("AIが考えています..."):
            # ここで、これまで作ってきたエージェントを呼び出す！
            response = chat_agent.invoke({"query": prompt})
            answer = response.get("answer", "申し訳ありません、回答を生成できませんでした。")
        
        # AIの回答を表示
        st.markdown(answer)
    
    # AIのメッセージを履歴に追加
    st.session_state.messages.append({"role": "assistant", "content": answer})