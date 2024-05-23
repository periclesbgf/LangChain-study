import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from PIL import Image
from langchain.schema import HumanMessage, AIMessage
import json
import requests

load_dotenv()

CODE = os.getenv("CODE")

def post_message(prompt):
    payload = json.dumps({'question': prompt, 'code': CODE})
    print(payload)
    url = "http://localhost:8000/sql"
    r = requests.post(url, data=payload)
    print(r)
    return r.content

def decode_response(response_content):
    try:
        api_response_str = response_content.decode('utf-8')
        response_list = json.loads(api_response_str)

        print("response_list:", response_list)

        if isinstance(response_list, list) and len(response_list) == 2:
            text = response_list[0]
            sql_query = response_list[1]
            return text, sql_query
        else:
            raise ValueError("Formato inesperado de resposta: {}".format(response_list))
    except Exception as e:
        print(f"Erro ao decodificar a resposta: {e}")
        raise

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0.5)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Sou Éden, seu assistente virtual. Como posso ajudar hoje?"}]

image_file = "/Users/peric/Downloads/example.ico"
img_form = Image.open(image_file)

img_form = img_form.resize((70, 70))

st.set_page_config(
    page_title="Your AI Assistant",
    layout="centered",
    initial_sidebar_state="expanded",
)

col1, col2 = st.columns([1, 8])

with col1:
    st.image(img_form)

with col2:
    st.subheader("VAI Assistant")
    st.caption("Grupo 6")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Digite sua mensagem aqui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Pensando..."):
        response_content = post_message(prompt)
        try:
            text, sql_query = decode_response(response_content)

            st.session_state.messages.append({"role": "assistant", "content": text + "\n \n" + sql_query})

            with st.chat_message("assistant"):
                st.markdown(text + "\n \n" + sql_query)

        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Erro ao processar a resposta: {e}"})
            with st.chat_message("assistant"):
                st.markdown(f"Erro ao processar a resposta: {e}")
