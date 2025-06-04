# 🧠 LLM Wikipedia Chatbot (Chatbot-LLM-Wikipedia-HuggingFace)

A simple chatbot that answers questions using a language model (LLM) and real-time Wikipedia search. Built with **Streamlit**, **LangChain**, and **HuggingFace Transformers**.

---

## 🚀 Features

- Ask any general knowledge question
- Searches Wikipedia for relevant info
- Uses LLM (e.g., FLAN-T5) to generate answers
- Simple web interface with Streamlit

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – UI
- [LangChain](https://www.langchain.com/) – RAG pipeline
- [Transformers (Hugging Face)](https://huggingface.co/) – LLM model
- [WikipediaRetriever](https://python.langchain.com/docs/modules/data_connection/retrievers/wikipedia) – Retrieves relevant content

---

## 📦 Installation

### Create a virtual environment:
```
python3 -m venv env
source env/bin/activate
```
### Install dependencies:

```
pip install -r requirements.txt
```

### Run the App
```
streamlit run llm_test.py
```
