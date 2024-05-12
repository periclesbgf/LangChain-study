from dotenv import load_dotenv
import os
from chains.chain_setup import setup_chain, invoke_chain
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

password = os.getenv("CODE")

def code_confirmation(code):
    if code == password:
        return True
    else:
        return False

def build_chain(text):
    chain, information = setup_chain(text)
    response = invoke_chain(chain, information)
    return response