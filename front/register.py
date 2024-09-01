import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
import json
import requests

load_dotenv()

# Variáveis de ambiente
CODE = os.getenv("CODE")
URL = os.getenv("URL")

# Função para enviar mensagem para o chatbot
def post_message(prompt, file=None):
    url = URL
    payload = {'question': prompt, 'code': CODE}
    r = requests.post(url, data=payload)
    return r.content

def post_student_register_credentials(nome, email, senha):
    url = "http://localhost:8000/create_account"
    payload = {
        'nome': nome,
        'email': email,
        'senha': senha,
        'tipo_usuario': 'student'
    }
    r = requests.post(url, data=payload)
    return r.content

# Função para decodificar a resposta do chatbot
def decode_response(response_content):
    try:
        api_response_str = response_content.decode('utf-8')
        response_list = json.loads(api_response_str)

        if isinstance(response_list, list) and len(response_list) == 2:
            text = response_list[1]
            return text
        else:
            raise ValueError("Formato inesperado de resposta: {}".format(response_list))
    except Exception as e:
        print(f"Erro ao decodificar a resposta: {e}")
        raise

# Configuração inicial da página
st.set_page_config(
    page_title="Your AI Assistant",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Inicialização da sessão
if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "is_logged_in" not in st.session_state:
    st.session_state["is_logged_in"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Sou Éden, seu assistente virtual. Como posso ajudar hoje?"}]
if "uploader_visible" not in st.session_state:
    st.session_state["uploader_visible"] = False

# Função para definir a página atual
def set_page(page_name):
    st.session_state["page"] = page_name

# Página Home
def home():
    st.title("Bem-vindo ao Assistente Educacional Éden")
    st.write("Este é o seu assistente virtual para ajudar nos estudos!")
    st.button("Cadastro de Aluno", on_click=set_page, args=["cadastro_aluno"])
    st.button("Cadastro de Professor", on_click=set_page, args=["cadastro_professor"])
    st.button("Login", on_click=set_page, args=["login"])

# Página de Cadastro de Aluno
def cadastro_aluno():
    st.title("Cadastro de Aluno")
    st.write("Preencha os dados abaixo para se cadastrar.")
    nome = st.text_input("Nome")
    email = st.text_input("Email")
    senha = st.text_input("Senha", type="password")
    if st.button("Cadastrar"):
        # Implementar lógica de cadastro
        if post_student_register_credentials(nome, email, senha) == b'{"message":"Conta criada com sucesso"}':
            st.success("Aluno cadastrado com sucesso!")
            set_page("login")
        else:
            st.error("Erro ao cadastrar aluno.")
    
    # Botão para voltar para a Home
    st.button("Voltar", on_click=set_page, args=["home"])

# Página de Cadastro de Professor
def cadastro_professor():
    st.title("Cadastro de Professor")
    st.write("Preencha os dados abaixo para se cadastrar.")
    nome = st.text_input("Nome")
    email = st.text_input("Email")
    senha = st.text_input("Senha", type="password")
    if st.button("Cadastrar"):
        # Implementar lógica de cadastro
        st.success("Professor cadastrado com sucesso!")
        set_page("login")
    
    # Botão para voltar para a Home
    st.button("Voltar", on_click=set_page, args=["home"])

# Página de Login
def login():
    st.title("Login")
    st.write("Insira suas credenciais para acessar a plataforma.")
    email = st.text_input("Email")
    senha = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        # Implementar lógica de autenticação
        st.session_state["is_logged_in"] = True
        st.success("Login realizado com sucesso!")
        set_page("home_usuario_logado")
    
    # Botão para voltar para a Home
    st.button("Voltar", on_click=set_page, args=["home"])

# Página Home do Usuário Logado
def home_usuario_logado():
    st.title("Home - Área Restrita")
    st.write("Bem-vindo, você está logado!")
    st.button("Ir para Sessão de Estudo", on_click=set_page, args=["sessao_estudo"])
    
    # Botão para voltar para a Home
    st.button("Voltar", on_click=set_page, args=["home"])

# Página de Sessão de Estudo (Chat)
def sessao_estudo():
    st.title("Sessão de Estudo")
    st.write("Aqui você pode interagir com o assistente de estudo.")

    # Código do chat implementado
    image_file = "/Users/peric/projects/LangChain-study/front/eden.jpeg"
    img_form = Image.open(image_file)
    img_form = img_form.resize((200, 200))

    file = None

    col1, col2 = st.columns([1, 8])
    with col1:
        st.image(img_form)
    with col2:
        st.subheader("Eden")
        st.caption("eden-ai")

    def show_upload(state: bool):
        st.session_state["uploader_visible"] = state

    with st.chat_message("system"):
        cols = st.columns((3, 1, 1))
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

        with st.spinner("Pensando..."):
            response_content = post_message(prompt, file=file)
            try:
                text = decode_response(response_content)
                st.session_state.messages.append({"role": "assistant", "content": text})
                with st.chat_message("assistant"):
                    st.markdown(text)
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Erro ao processar a resposta: {e}"})
                with st.chat_message("assistant"):
                    st.markdown(f"Erro ao processar a resposta: {e}")

    # Botão para voltar para a Home após a sessão de estudo
    st.button("Voltar", on_click=set_page, args=["home"])

# Navegação de páginas
if st.session_state["page"] == "home":
    home()
elif st.session_state["page"] == "cadastro_aluno":
    cadastro_aluno()
elif st.session_state["page"] == "cadastro_professor":
    cadastro_professor()
elif st.session_state["page"] == "login":
    login()
elif st.session_state["page"] == "home_usuario_logado" and st.session_state["is_logged_in"]:
    home_usuario_logado()
elif st.session_state["page"] == "sessao_estudo" and st.session_state["is_logged_in"]:
    sessao_estudo()
else:
    st.write("Você precisa fazer login para acessar essa página.")
