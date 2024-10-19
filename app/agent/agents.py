from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from database.mongo_database_manager import CustomMongoDBChatMessageHistory
from utils import OPENAI_API_KEY
from database.vector_db import QdrantHandler, Embeddings
from langchain.memory.chat_message_histories import MongoDBChatMessageHistory
from langchain.schema import BaseMessage, AIMessage, message_to_dict, messages_from_dict
from pymongo import errors
from datetime import datetime, timezone
import json
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from database.vector_db import QdrantHandler, Embeddings
from langchain.tools import tool
from youtubesearchpython import VideosSearch
import wikipediaapi
from serpapi import GoogleSearch
from utils import OPENAI_API_KEY


RETRIEVAL_PROMPT = """
Você é o Agente de Recuperador de Conteúdo. Sua função é sugerir recursos e materiais relevantes com base na necessidade do estudante.

### Responsabilidades:
- **Buscar Recursos**: Acesse o banco de dados vetorial e o banco de recursos para encontrar materiais relevantes.
- **Fornecer Conteúdo Personalizado**: Sugira vídeos, artigos e exemplos alinhados ao plano de execução do estudante.

### Entrada:
- **Consulta**: "{consulta_usuario}"
- **Perfil do Estudante**: {perfil_do_estudante}
- **Plano de Execução**: {plano_de_execucao}

### Tarefas:
1. **Pesquisar Recursos Relevantes**: Use a consulta do usuário para encontrar materiais apropriados.
2. **Personalizar Sugestões**: Adapte os recursos às preferências e necessidades do estudante.
3. **Fornecer Recomendações Claras**: Apresente os recursos de forma organizada e acessível.
"""

class RetrievalAgent:
    def __init__(
            self,
            qdrant_handler: QdrantHandler,
            embeddings: Embeddings,
            model_name: str = "gpt-4o-mini",
            ):
        self.model = ChatOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)
        self.qdrant_handler = qdrant_handler
        self.embeddings = embeddings

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
                ("system", RETRIEVAL_PROMPT),
            ]
        )
        self.agent = create_tool_calling_agent(self.model, [self.retrieve_context, self.search_youtube, self.search_wikipedia, self.search_google], self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=[self.retrieve_context, self.search_youtube, self.search_wikipedia, self.search_google])

    @tool
    def retrieve_context(self, query: str):
        """Recupera recursos relevantes do banco vetorial com base na consulta."""
        print(f"Retrieving context for query: {query}")
        embedding = self.embeddings.embed_query(query)
        results = self.qdrant_handler.similarity_search(
            embedding=embedding,
            k=5,
        )
        print(f"Retrieved {len(results)} relevant documents")
        context = [result['content'] for result in results]
        return "\n".join(context)

    @tool
    def search_youtube(query: str) -> str:
        """
        Realiza uma pesquisa no YouTube e retorna o link do vídeo mais relevante.
        """
        videos_search = VideosSearch(query, limit=1)
        results = videos_search.result()

        if results['result']:
            video_info = results['result'][0]
            return f"Título: {video_info['title']}\nLink: {video_info['link']}\nDescrição: {video_info.get('descriptionSnippet', 'Sem descrição')}"
        else:
            return "Nenhum vídeo encontrado."

    @tool
    def search_wikipedia(query: str) -> str:
        """
        Realiza uma pesquisa no Wikipedia e retorna o resumo da página.
        """
        wiki_wiki = wikipediaapi.Wikipedia('pt')  # Português
        page = wiki_wiki.page(query)

        if page.exists():
            return f"Título: {page.title}\nResumo: {page.summary[:500]}...\nLink: {page.fullurl}"
        else:
            return "Página não encontrada."

    @tool
    def search_google(query: str) -> str:
        """
        Realiza uma pesquisa no Google e retorna o primeiro link relevante.
        """
        params = {
            "q": query,
            "hl": "pt",  # Português
            "gl": "br",  # Região Brasil
            "api_key": "YOUR_SERPAPI_KEY"  # Substitua com sua chave da API SerpAPI
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        if "organic_results" in results:
            top_result = results["organic_results"][0]
            return f"Título: {top_result['title']}\nLink: {top_result['link']}\nDescrição: {top_result.get('snippet', 'Sem descrição')}"
        else:
            return "Nenhum resultado encontrado no Google."

    def invoke(self, query: str, student_profile, execution_plan):
        """Invoca o agente para sugerir recursos."""
        try:
            response = self.agent_executor.invoke({
                "input": query,
                "perfil_do_estudante": student_profile,
                "plano_de_execucao": execution_plan,
                "consulta_usuario": query
            })
            print(f"Retrieval Agent Response: {response}")
            return response
        except Exception as e:
            print(f"Error in RetrievalAgent: {e}")
            return "An error occurred while retrieving resources."

PROMPT_AGENTE_ANALISE_PROGRESSO = """
Você é o Agente de Análise de Progresso. Sua responsabilidade é avaliar o desempenho do estudante e fornecer feedback corretivo, se necessário.

### Responsabilidades:
- **Avaliar o Progresso**: Verifique se o estudante está avançando conforme o plano de execução.
- **Fornecer Feedback**: Identifique áreas de dificuldade e sugira melhorias.
- **Ajustar o Plano**: Sinalize se o plano precisa ser revisado.

### Entrada:
- **Histórico de Interações**: {historico_de_interacoes}
- **Progresso Atual da Sessão**: {session_progress}
- **Plano de Execução**: {plano_de_execucao}

### Tarefas:
1. **Analisar o Histórico**: Examine as interações para identificar padrões de dificuldade.
2. **Comparar com o Plano**: Verifique se o progresso está alinhado com os objetivos definidos.
3. **Fornecer Feedback**: Prepare um relatório com sugestões e observações.

**Exemplo de Feedback**:
"O estudante tem demonstrado dificuldade com conceitos fundamentais de álgebra linear. Recomendo focar em exercícios básicos antes de avançar para tópicos mais complexos."
"""

class ProgressAnalysisAgent:
    def __init__(self, student_profile, execution_plan, mongo_uri, database_name, session_id, user_email, disciplina, model_name="gpt-4o"):
        self.student_profile = student_profile
        self.execution_plan = execution_plan
        self.session_id = session_id
        self.user_email = user_email
        self.disciplina = disciplina
        self.model = ChatOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)
        self.history = CustomMongoDBChatMessageHistory(
            user_email=self.user_email,
            disciplina=self.disciplina,
            connection_string=mongo_uri,
            session_id=self.session_id,
            database_name=database_name,
            collection_name="chat_history",
            session_id_key="session_id",
            history_key="history",
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PROMPT_AGENTE_ANALISE_PROGRESSO),
                MessagesPlaceholder(variable_name="history"),
            ]
        )
        self.agent = create_tool_calling_agent(self.model, [self.analyze_progress], self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=[self.analyze_progress])

    @tool
    def analyze_progress(self):
        """Analisa o progresso do estudante."""
        return "Analisando o progresso do estudante."

    def invoke(self, session_progress: str):
        # Obter o histórico de mensagens
        history_messages = self.history.messages
        # Formatar o prompt
        formatted_prompt = self.prompt.format(
            historico_de_interacoes="\n".join([msg.content for msg in history_messages]),
            session_progress=session_progress,
            plano_de_execucao=self.execution_plan
        )
        # Invocar o agente
        response = self.agent_executor.invoke({"history": history_messages})
        return response
    


PROMPT_AGENTE_CONVERSASIONAL = """
Você é o Agente de Pensamento Crítico, o tutor principal responsável por ensinar o estudante, se comunicar de forma eficaz e promover o desenvolvimento do pensamento crítico.

### Responsabilidades:
- **Ensino do Conteúdo**: Apresente conceitos de forma clara e adaptada ao nível do estudante.
- **Comunicação Eficaz**: Use exemplos personalizados e linguagem apropriada ao perfil do estudante.
- **Desenvolvimento do Pensamento Crítico**: Incentive o estudante a refletir e encontrar respostas por conta própria.

### Entrada:
- **Perfil do Estudante**: {perfil_do_estudante}
- **Plano de Execução**: {plano_de_execucao}
- **Histórico de Interações**: {historico_de_interacoes}

### Tarefas:
1. **Responda de forma personalizada**: Use o perfil e plano do estudante para adaptar sua resposta.
2. **Inicie perguntas reflexivas**: Ajude o estudante a desenvolver habilidades críticas e resolver problemas.
3. **Verifique o alinhamento com o plano**: Certifique-se de que sua resposta está de acordo com o plano de execução.

**Exemplo de Interação**:
*Entrada*: "Não entendo como resolver essa equação diferencial."
*Resposta*: "Vamos resolver isso juntos. O que você já sabe sobre integrais? Talvez possamos começar por aí."

**Formato de Saída**:
- Uma resposta clara e relevante para o estudante.
"""

class ChatAgent:
    def __init__(self, student_profile, execution_plan, mongo_uri, database_name, session_id, user_email, disciplina, model_name="gpt-4o-mini"):
        self.student_profile = student_profile
        self.execution_plan = execution_plan
        self.session_id = session_id
        self.user_email = user_email
        self.disciplina = disciplina
        self.model = ChatOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)
        self.history = CustomMongoDBChatMessageHistory(
            user_email=self.user_email,
            disciplina=self.disciplina,
            connection_string=mongo_uri,
            session_id=self.session_id,
            database_name=database_name,
            collection_name="chat_history",
            session_id_key="session_id",
            history_key="history",
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PROMPT_AGENTE_CONVERSASIONAL),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ]
        )
        self.agent = create_tool_calling_agent(self.model, [self.respond_to_student], self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=[self.respond_to_student])

    @tool
    def respond_to_student(self, query: str):
        """Processa a entrada do estudante e gera uma resposta reflexiva."""
        # Aqui você pode adicionar lógica adicional se necessário
        return query

    def invoke(self, user_input: str):
        # Obter o histórico de mensagens
        history_messages = self.history.messages
        # Formatar o prompt
        formatted_prompt = self.prompt.format(
            perfil_do_estudante=self.student_profile,
            plano_de_execucao=self.execution_plan,
            historico_de_interacoes="\n".join([msg.content for msg in history_messages]),
            input=user_input
        )
        # Invocar o agente
        response = self.agent_executor.invoke({"input": user_input, "history": history_messages})
        # Salvar a interação no histórico
        self.history.add_message(BaseMessage(content=user_input, role="human"))
        self.history.add_message(AIMessage(content=response, role="assistant"))
        return response