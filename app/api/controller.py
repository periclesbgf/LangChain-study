from chains.chain_setup import LanguageChain

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