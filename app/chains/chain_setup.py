import os
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

def setup_chain():
    """configuring chain and prompt template for the chain"""

    information = """
    A person who is a software engineer and loves to play basketball. They are also a fan of the Golden State Warriors.
    """
    summary_template = """
    Given the information {information} about a person, I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=os.environ["OPENAI_API_KEY"]
    )

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    return chain, information

def invoke_chain(chain, information):
    result = chain.invoke(input={"information": information})
    return result
