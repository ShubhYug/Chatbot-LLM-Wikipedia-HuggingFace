# from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


def hf_test(question: str):
    """
    Test HuggingFace model with LangChain.
    This function demonstrates how to use a HuggingFace model with LangChain
    for text generation tasks.
    """
    # Load a local HuggingFace model
    model_name = "google/flan-t5-base"  # You can also try 'tiiuae/falcon-7b-instruct' if you have enough RAM
    # pipe = pipeline("text2text-generation", model=model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model_name,
        tokenizer=model_name,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.7,
        repetition_penalty=1.2,
    )

    # Wrap the pipeline in LangChain
    llm = HuggingFacePipeline(pipeline=pipe)

    template = """
        You are a knowledgeable assistant that explains technical concepts clearly.

        Question: {question}
        Answer:
        """
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # Create LangChain chain
    chain = LLMChain(llm=llm, prompt=prompt)
    print("Chain created successfully.: ", chain)
    # Ask a question
    # question = "What is the capital of India?"
    answer = chain.run(question=question)
    return answer
