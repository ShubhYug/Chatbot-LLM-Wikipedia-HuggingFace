import os

from langchain.chains import RetrievalQA
from langchain_community.retrievers import WikipediaRetriever
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline


def rag_with_wikipedia(question: str):
    # Use local pipeline instead of HuggingFaceHub to avoid InferenceClient errors
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model_name,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.5,
        repetition_penalty=1.2,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    retriever = WikipediaRetriever(lang="en", load_max_docs=1)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=False
    )

    return qa_chain.invoke({"query": question})
