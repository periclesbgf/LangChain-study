from chains.chain_setup import LanguageChain, SQLChain, AnswerChain
from database.query import execute_query, prepare_query

from utils import OPENAI_API_KEY, CODE

def code_confirmation(code):
    if code == CODE:
        return True
    else:
        return False

def build_chain(text):
    chain = LanguageChain(api_key=OPENAI_API_KEY)
    response = chain.setup_chain(text=text)
    return response

def build_sql_chain(text):
    chain = SQLChain(api_key=OPENAI_API_KEY)
    response = chain.setup_sql_chain(text=text)
    query_prepared = prepare_query(response)
    print(query_prepared)
    data = execute_query(query_prepared)
    print("data: ", data)
    # if len(data) >= 300:
    #     return "Data too large to process"
    awnser = AnswerChain(api_key=OPENAI_API_KEY)
    response = awnser.setup_chain(user_question=text, data=data, query=query_prepared)

    return response, query_prepared
