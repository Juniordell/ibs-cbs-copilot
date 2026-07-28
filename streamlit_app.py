import os
import httpx
import streamlit as st

API_URL = os.getenv("COPILOT_API_URL", "https://ibs-cbs-copilot.fly.dev")

st.set_page_config(page_title="IBS/CBS Copilot", page_icon="⚖️", layout="centered")

st.title("⚖️ IBS/CBS Copilot")
st.caption("RAG grounded on LC 214/2025, EC 132/2023, Decreto 12.955/2026")

question = st.text_input(
    "Pergunta em português:",
    placeholder="Qual a alíquota do IBS?",
)

top_k = st.slider("Chunks a recuperar (top_k)", 3, 15, 5)

if st.button("Perguntar", type="primary", disabled=not question):
    with st.spinner("Consultando a legislação..."):
        try:
            r = httpx.post(
                f"{API_URL}/v1/ask",
                json={"question": question, "top_k": top_k},
                timeout=90,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            st.error(f"Erro: {e}")
            st.stop()

    answer = data["answer"]

    confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}[answer["confidence"]]
    st.markdown(f"### Resposta {confidence_color}")
    st.write(answer["answer"])

    if answer["citations"]:
        st.markdown("### Citações")
        for c in answer["citations"]:
            st.markdown(f"**{c['article']}** · _{c['source']}_")
            st.caption(f"> {c['quote']}")

    if answer.get("gaps"):
        st.info(f"**Lacunas:** {answer['gaps']}")

    with st.expander("Debug"):
        st.json({
            "tokens_in": data["input_tokens"],
            "tokens_out": data["output_tokens"],
            "model": data["model"],
            "articles_retrieved": data["retrieved_articles"],
        })