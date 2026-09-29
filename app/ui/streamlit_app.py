import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Agentic AI Knowledge Assistant", layout="centered")

st.title("Agentic AI Knowledge Assistant")
st.write("Ask questions about the Agentic AI eBook.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "context" in message:
            st.caption(f"Confidence: {message['confidence']:.2f}")
            with st.expander("Retrieved Context"):
                for chunk in message["context"]:
                    st.markdown(f"**Page {chunk['page']}**: {chunk['text']} (Score: {chunk['score']:.2f})")

query = st.chat_input("What would you like to know?")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post("http://127.0.0.1:8000/chat", json={"query": query})
                response.raise_for_status()
                data = response.json()
                
                answer = data["final_answer"]
                confidence = data["confidence_score"]
                chunks = data["retrieved_context_chunks"]
                
                st.markdown(answer)
                st.caption(f"Confidence: {confidence:.2f}")
                
                if chunks:
                    with st.expander("Retrieved Context"):
                        for chunk in chunks:
                            st.markdown(f"**Page {chunk['page']}**: {chunk['text']} (Score: {chunk['score']:.2f})")
                else:
                    st.info("System refused because information is unavailable.")
                    
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "confidence": confidence,
                    "context": chunks
                })
            except requests.exceptions.RequestException as e:
                error_msg = str(e)
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_msg = e.response.json().get("detail", str(e))
                    except:
                        pass
                st.error(f"API Error: {error_msg}")
