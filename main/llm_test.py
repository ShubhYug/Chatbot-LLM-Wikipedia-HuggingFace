import helpper as hlp
import rag_chain  # your file rag_chain.py must be saved as wiki.py or import as needed
import streamlit as st

st.title("LLM Wikipedia Chatbot")

# Empty default input
question = st.text_input("Ask something:")

# Button to trigger the search
if st.button("Get Answer") and question.strip():
    st.write("Question:", question)
    with st.spinner("Thinking..."):
        answer = rag_chain.rag_with_wikipedia(question)
    st.write("Answer:", answer["result"])
