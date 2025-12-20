import json
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8080"

st.set_page_config(page_title="SEC Filings Multi-Agent Chat (Demo)", layout="wide")
st.title("📄 SEC Filings Multi-Agent Chat (Demo)")

with st.expander("What this UI shows"):
    st.markdown("""
- Chat interface for SEC filings Q&A
- Shows **sources**, **routing (agents used)**, and **cost report**
- Uses your FastAPI backend `/ask`
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("meta"):
            st.code(json.dumps(m["meta"], indent=2), language="json")

question = st.chat_input("Ask about the filings… (e.g., 'Compare gross margin Q1 2024 vs Q1 2025')")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Agents working..."):
            try:
                # resp = requests.post(
                #     f"{BACKEND_URL}/ask",
                #     json={"question": question, "top_k": 10},
                #     timeout=300
                resp = requests.get(
                    f"{BACKEND_URL}/ask",
                    params={"q": question},
                    timeout=300000
                )
                # )
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                st.error(f"Backend error: {e}")
                st.stop()

        # Show answer
        st.markdown(result.get("answer", ""))

        # Side panels
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sources")
            sources = result.get("sources", [])
            if sources:
                for s in sources:
                    st.write(f"- {s}")
            else:
                st.write("No sources returned.")

        with col2:
            st.subheader("Diagnostics")
            meta = {
                "routing": result.get("routing", {}),
                "cost": result.get("cost", {}),
                "latency_seconds": result.get("latency_seconds", None),
            }
            st.code(json.dumps(meta, indent=2), language="json")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("answer", ""),
        "meta": {
            "sources": result.get("sources", []),
            "routing": result.get("routing", {}),
            "cost": result.get("cost", {}),
            "latency_seconds": result.get("latency_seconds", None),
        }
    })
