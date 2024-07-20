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
URL = os.getenv("URL")

def post_message(prompt, file=None):
    url = URL
    payload = {'question': prompt, 'code': CODE}
    r = requests.post(url, data=payload)
    return r.content


def decode_response(response_content):
    try:
        api_response_str = response_content.decode('utf-8')
        response_list = json.loads(api_response_str)

        print("response_list:", response_list)

        if isinstance(response_list, list) and len(response_list) == 2:
            text = response_list[1]
            return text
        else:
            raise ValueError("Formato inesperado de resposta: {}".format(response_list))
    except Exception as e:
        print(f"Erro ao decodificar a resposta: {e}")
        raise

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Sou Éden, seu assistente virtual. Como posso ajudar hoje?"}]

image_file = "/Users/peric/projects/LangChain-study/front/eden.jpeg"
img_form = Image.open(image_file)

img_form = img_form.resize((200, 200))

file = None

st.set_page_config(
    page_title="Your AI Assistant",
    layout="centered",
    initial_sidebar_state="expanded",
)

col1, col2 = st.columns([1, 8])


with col1:
    st.image(img_form)

with col2:
    st.subheader("Eden")
    st.caption("eden-ai")

if "uploader_visible" not in st.session_state:
    st.session_state["uploader_visible"] = False

def show_upload(state:bool):
    st.session_state["uploader_visible"] = state

with st.chat_message("system"):
    cols= st.columns((3,1,1))
    cols[0].write("Quer submeter algum arquivo?")
    cols[1].button("Sim", use_container_width=True, on_click=show_upload, args=[True])
    cols[2].button("Não", use_container_width=True, on_click=show_upload, args=[False])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state["uploader_visible"]:
    with st.chat_message("assistant"):
        file = st.file_uploader("Coloque aqui seu arquivo PDF")


if prompt := st.chat_input("Digite sua mensagem aqui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if file is not None:
        with st.spinner("Pensando..."):
            response_content = post_message(prompt, file=file)
            try:
                text, sql_query = decode_response(response_content)
                if sql_query == None:
                    st.session_state.messages.append({"role": "assistant", "content": text})

                    with st.chat_message("assistant"):
                        st.markdown(text)
                else:
                    st.session_state.messages.append({"role": "assistant", "content": text + "\n \n" + sql_query})

                    with st.chat_message("assistant"):
                        st.markdown(text + "\n \n" + sql_query)

            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Erro ao processar a resposta: {e}"})
                with st.chat_message("assistant"):
                    st.markdown(f"Erro ao processar a resposta: {e}")
    else:
        with st.spinner("Pensando..."):
            response_content = post_message(prompt)
            try:
                text, sql_query = decode_response(response_content)
                if sql_query == None:
                    st.session_state.messages.append({"role": "assistant", "content": text})

                    with st.chat_message("assistant"):
                        st.markdown(text)
                else:
                    st.session_state.messages.append({"role": "assistant", "content": text + "\n \n" + sql_query})

                    with st.chat_message("assistant"):
                        st.markdown(text + "\n \n" + sql_query)

            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Erro ao processar a resposta: {e}"})
                with st.chat_message("assistant"):
                    st.markdown(f"Erro ao processar a resposta: {e}")


