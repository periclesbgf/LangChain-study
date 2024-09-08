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


def post_message(prompt, file=None):
    url = f"{URL}/prompt"  # Certifique-se de que o endpoint está correto
    payload = {'question': prompt, 'code': CODE}

    headers = {
        "Authorization": f"Bearer {st.session_state['access_token']}"
    }

    r = requests.post(url, data=payload, headers=headers)
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

def post_login_credentials(email, senha):
    url = f"{URL}/login"
    payload = {
        'email': email,
        'senha': senha
    }
    r = requests.post(url, data=payload)
    return r

# Função para decodificar a resposta do chatbot
def decode_response(response_content):
    try:
        api_response_str = response_content.decode('utf-8')
        # Carregar a string como um dicionário JSON
        response_data = json.loads(api_response_str)

        # Verificar se o campo 'response' existe
        if 'response' in response_data:
            response_text = response_data['response']
            audio_data = response_data.get('audio')  # Pega o campo 'audio', se existir
            return response_text, audio_data
        else:
            raise ValueError(f"Formato inesperado de resposta: {response_data}")
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

# Função para exibir a barra lateral nas páginas restritas
def show_sidebar():
    st.sidebar.title("Menu")
    st.sidebar.button("Home Restrita", on_click=set_page, args=["home_usuario_logado"])
    st.sidebar.button("Dashboard", on_click=set_page, args=["dashboard"])
    st.sidebar.button("Sessão de Estudo", on_click=set_page, args=["sessao_estudo"])
    st.sidebar.markdown("---")
    st.sidebar.button("Logout", on_click=set_page, args=["login"])

# Página Home
def home():
    st.title("Bem-vindo ao Assistente Educacional Eden")
    st.write("""
        O Eden AI é uma plataforma que conecta alunos e professores para um aprendizado mais eficiente e personalizado.
        Utilize nossos serviços para melhorar a interação educacional e facilitar o processo de ensino-aprendizagem.
        Você pode se cadastrar como aluno ou professor e começar a aproveitar os recursos que oferecemos!
    """)
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

def send_authenticated_request(endpoint, method="GET", data=None):
    if "access_token" not in st.session_state:
        st.error("Você precisa fazer login.")
        set_page("login")
        return None

    headers = {
        "Authorization": f"Bearer {st.session_state['access_token']}"
    }

    url = f"{URL}{endpoint}"

    if method == "POST":
        response = requests.post(url, headers=headers, data=data)
    else:
        response = requests.get(url, headers=headers)

    if response.status_code == 401:
        st.error("Sua sessão expirou. Faça login novamente.")
        set_page("login")
        return None

    return response

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

# Função para o login no frontend
def login():
    st.title("Login")
    st.write("Insira suas credenciais para acessar a plataforma.")

    email = st.text_input("Email")
    senha = st.text_input("Senha", type="password")

    if st.button("Entrar"):
        response = post_login_credentials(email, senha)

        if response.status_code == 200:
            data = response.json()
            access_token = data.get("access_token")
            token_type = data.get("token_type")

            st.success("Login realizado com sucesso!")

            # Armazena o token no estado da sessão
            st.session_state["access_token"] = access_token
            st.session_state["token_type"] = token_type
            st.session_state["is_logged_in"] = True

            # Redireciona automaticamente para a home do usuário logado
            set_page("home_usuario_logado")
        else:
            st.error(f"Falha no login: {response.status_code} - {response.text}")

    st.button("Voltar", on_click=set_page, args=["home"])

# Página Home do Usuário Logado
def home_usuario_logado():
    show_sidebar()  # Exibir a barra lateral nas telas restritas
    st.title("Home - Área Restrita")
    st.write("Bem-vindo, você está logado!")

# Página de Sessão de Estudo (Chat)
def sessao_estudo():
    show_sidebar()  # Exibir a barra lateral nas telas restritas
    st.title("Sessão de Estudo")
    st.write("Aqui você pode interagir com o assistente de estudo.")

    # Código do chat implementado
    image_file = "front/eden.jpeg"
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
            print(response_content)
            try:
                # Decodifica a resposta (texto e áudio)
                response_text, audio_data = decode_response(response_content)

                # Exibe apenas o texto da resposta
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                with st.chat_message("assistant"):
                    st.markdown(response_text)

                # Se houver áudio, exibe o reprodutor de áudio
                if audio_data:
                    st.audio(audio_data, format="audio/wav")

            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Erro ao processar a resposta: {e}"})
                with st.chat_message("assistant"):
                    st.markdown(f"Erro ao processar a resposta: {e}")

# Página de Dashboard (nova)
def dashboard():
    show_sidebar()  # Exibir a barra lateral nas telas restritas
    st.title("Dashboard")
    st.write("Bem-vindo ao Dashboard. Aqui você pode visualizar as estatísticas e progresso.")

# Navegação de páginas
if st.session_state["is_logged_in"]:
    if st.session_state["page"] == "home_usuario_logado":
        home_usuario_logado()
    elif st.session_state["page"] == "sessao_estudo":
        sessao_estudo()
    elif st.session_state["page"] == "dashboard":
        dashboard()
else:
    if st.session_state["page"] == "home":
        home()
    elif st.session_state["page"] == "cadastro_aluno":
        cadastro_aluno()
    elif st.session_state["page"] == "cadastro_professor":
        cadastro_professor()
    elif st.session_state["page"] == "login":
        login()
    else:
        st.write("Você precisa fazer login para acessar essa página.")
