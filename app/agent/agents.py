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
from typing import Callable
from langchain_core.runnables import RunnableWithMessageHistory
import json
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.tools import tool, StructuredTool, BaseTool
from youtubesearchpython import VideosSearch
import wikipediaapi
from langchain.agents.react.output_parser import ReActOutputParser
from serpapi import GoogleSearch
from utils import OPENAI_API_KEY, MONGO_URI, MONGO_DB_NAME
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool, initialize_agent, create_react_agent
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.sequential import SequentialChain
from langgraph.checkpoint.memory import MemorySaver
from api.controllers.student_profile_controller import StudentProfileController
from database.mongo_database_manager import MongoDatabaseManager


REACT_PROMP = """
You are a personalized tutor. Your goal is to look at the student progile the way he learns
and the current state of conversation and what is his necessities.

Students can have different learning preferences according to the Felder-Silverman model. The dimensions and their characteristics are:

1. **Sensing vs. Intuitive**:
   - Sensing: Prefers concrete facts, details, and practical applications.
   - Intuitive: Enjoys abstract concepts, innovation, and theories.

2. **Visual vs. Verbal**:
   - Visual: Learns better through images, diagrams, and videos.
   - Verbal: Prefers oral explanations or reading texts.

3. **Active vs. Reflective**:
   - Active: Learns through hands-on activities and group collaboration.
   - Reflective: Prefers to think and work alone before acting.

4. **Sequential vs. Global**:
   - Sequential: Processes knowledge linearly and in an organized manner.
   - Global: Understands better with a big-picture view and makes broad connections.

Based on these characteristics, generate a personalized study plan that considers:
1. The student's dominant dimension in each category.
2. Strategies and study materials aligned with these preferences.
3. Suggestions for developing less dominant skills, if necessary.

Given the student profile:
{perfil_do_estudante}

Given the execution answer plan:
{plano_de_execucao}

Do NOT answer his question, teach him the critical thinking about the question and how to solve it as best you can. 
ALWAYS RETRIEVE CONTEXT.
Try to ask questions that promote deeper understanding.
prioritize the tool `retrieve_context` to provide relevant context.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer

Thought: you should always think about what to do

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the step by step to teach the student how to solve the question

Final Answer: the step by step to the original input question translated to pt-br language.

Begin!

Question: {input}

Thought: {agent_scratchpad}
"""

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

### Nota Importante:
- Sempre priorize a ferramenta `retrieve_context` para fornecer contexto educativo relevante antes de utilizar Wikipedia.
- Garanta que todas as respostas sejam apresentadas de forma clara e direcionada ao objetivo do estudante.
"""
EDUCATIONAL_PROMPT = """
You are an educational guide focused on developing critical thinking. Your role is to help students learn how to solve problems on their own, NOT to provide direct answers.

Given the student's learning style based on the Felder-Silverman model:
{perfil_do_estudante}

And the current execution plan:
{plano_de_execucao}

Your responsibilities:
1. Guide students through problem-solving steps
2. Ask thought-provoking questions
3. Provide hints and suggestions rather than answers
4. Adapt explanations to match their learning style
5. Encourage self-discovery and critical thinking

When responding:
- Break down complex problems into smaller steps
- Use analogies relevant to the student's interests
- Provide scaffolding that matches their learning style
- Ask questions that promote deeper understanding
- Give constructive feedback on their thinking process

You have access to these tools:
{tools}

Use this format:

Question: the student's question you must help them solve

Thought: your analysis of how to guide the student (not visible to them)

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know how to guide the student

Final Answer: A step-by-step guidance plan that:
1. Asks leading questions
2. Provides relevant hints
3. Suggests learning resources
4. Encourages critical thinking
Do NOT provide the direct answer.

Begin!

Question: {input}

Thought: {agent_scratchpad}
"""


class RetrievalAgent:
    def __init__(
            self,
            qdrant_handler: QdrantHandler,
            embeddings: Embeddings,
            disciplina: str,
            student_email: str,
            session_id: str,
            model_name: str = "gpt-4o",
    ):
        self.model = ChatOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)
        self.qdrant_handler = qdrant_handler
        self.embeddings = embeddings
        self.student_email = student_email
        self.disciplina = disciplina
        self.session_id = session_id
        self.mongo_database_manager = MongoDatabaseManager()
        # Initialize chat history
        self.chat_history = CustomMongoDBChatMessageHistory(
            user_email=self.student_email,
            disciplina=self.disciplina,
            connection_string=MONGO_URI,
            session_id=session_id,
            database_name=MONGO_DB_NAME,
            collection_name="chat_history",
            session_id_key="session_id",
            history_key="history",
        )

        # Initialize tools
        self.tools = [
            Tool(
                name="retrieve_context",
                func=self.retrieve_context,
                description="Search the knowledge base for relevant materials and examples"
            ),
            # Tool(
            #     name="search_youtube",
            #     func=self.search_youtube,
            #     description="Search YouTube for relevant educational videos"
            # ),
            # Tool(
            #     name="search_wikipedia",
            #     func=self.search_wikipedia,
            #     description="Search Wikipedia for detailed information and explanations"
            # )
        ]


        self.prompt = ChatPromptTemplate.from_template(REACT_PROMP)

        # Initialize agent
        self.agent = create_react_agent(
            llm=self.model,
            tools=self.tools,
            prompt=self.prompt
        )

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )

    def get_student_profile(self) -> dict:
        """Retrieve the student's profile from the database."""
        student_profile = self.mongo_database_manager.get_student_profile(
            email=self.student_email,
            collection_name="student_learn_preference"
        )
        return student_profile

    def retrieve_context(self, query: str) -> str:
        print(f"Retrieving context for query: {query}")

        try:
            # Depurar metadados
            print("Query: ",query)
            self.qdrant_handler.debug_metadata()
            semfiltro = self.qdrant_handler.similarity_search_without_filter(query, k=5)
            print(f"Sem filtro: {len(semfiltro)}")
            # Busca sem filtro
            #print("🔍 Buscando sem filtro...")
            #no_filter_results = self.qdrant_handler.similarity_search_without_filter(query, k=5)

            # Busca com filtro
            print("🔍 Buscando com filtro...")
            filter_results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=str(self.student_email),
                disciplina=str(self.disciplina),  # Garantir que é string
                k=5
            )

            print(f"Com filtro: {len(filter_results)}")

            if filter_results:
                context = "\n".join([doc.page_content for doc in filter_results])
            else:
                context = "Nenhum contexto relevante encontrado."

            return context

        except Exception as e:
            print(f"Erro ao recuperar contexto: {e}")
            return "Ocorreu um erro ao tentar recuperar o contexto."


    def search_youtube(self, query: str) -> str:
        """
        Realiza uma pesquisa no YouTube e retorna o link do vídeo mais relevante.
        """
        try:
            videos_search = VideosSearch(query, limit=1)
            results = videos_search.result()

            if results['result']:
                video_info = results['result'][0]
                return f"Título: {video_info['title']}\nLink: {video_info['link']}\nDescrição: {video_info.get('descriptionSnippet', 'Sem descrição')}"
            else:
                return "Nenhum vídeo encontrado."
        except Exception as e:
            print(f"Erro ao buscar no YouTube: {e}")
            return "Ocorreu um erro ao buscar no YouTube."

    def search_wikipedia(self, query: str) -> str:
        """
        Realiza uma pesquisa no Wikipedia e retorna o resumo da página.
        """
        try:
            wiki_wiki = wikipediaapi.Wikipedia('pt')  # Português
            page = wiki_wiki.page(query)

            if page.exists():
                return f"Título: {page.title}\nResumo: {page.summary[:500]}...\nLink: {page.fullurl}"
            else:
                return "Página não encontrada."
        except Exception as e:
            print(f"Erro ao buscar no Wikipedia: {e}")
            return "Ocorreu um erro ao buscar no Wikipedia."

    async def invoke(self, query: str, student_profile: dict, execution_plan: dict, config: dict) -> dict:
        """Process a query and provide educational guidance"""
        try:
            print(f"RetrievalAgent processing query: {query}")
            
            # Prepare the input with intermediate_steps for agent_scratchpad
            agent_input = {
                "input": query,
                "perfil_do_estudante": student_profile,
                "plano_de_execucao": execution_plan,
            }

            # Execute the agent
            response = await self.agent_executor.ainvoke(
                agent_input,
                config=config
            )

            # Process the response
            if isinstance(response, dict):
                output = response.get("output", "")
                steps = response.get("intermediate_steps", [])
                
                # Save response to chat history
                self.chat_history.add_message(AIMessage(content=output))
                
                return {
                    "guidance": output,
                    "steps": steps,
                    "context_used": bool(steps),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Save response to chat history
                self.chat_history.add_message(AIMessage(content=str(response)))
                
                return {
                    "guidance": str(response),
                    "steps": [],
                    "context_used": False,
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            print(f"Error in RetrievalAgent: {e}")
            import traceback
            traceback.print_exc()
            
            error_message = (
                "I encountered an issue while processing your question. "
                "Let's try breaking it down into smaller steps. "
                "What specific aspect would you like to understand first?"
            )
            
            # Save error response to chat history
            self.chat_history.add_message(AIMessage(content=error_message))
            
            return {
                "guidance": error_message,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

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


PROMPT_COMBINADO = """
Você é um agente de aprendizado que realiza duas tarefas principais:
1. **Análise da Pergunta**: Entender a pergunta do estudante e criar um plano de resposta personalizado com base no perfil e no histórico do aluno.
2. **Execução da Resposta**: Ensinar o estudante de forma eficaz, seguindo o plano de resposta gerado e o plano de execução existente.

### Entrada:
- **Pergunta do Usuário**
- **Perfil do Estudante**
- **Plano de Execução**
- **Histórico da Conversa**

### Tarefas:
1. **Compreender a Pergunta**:
   - Identifique o que o estudante quer saber.
2. **Criar um Plano de Resposta**:
   - Com base no perfil e nas necessidades do estudante, defina os passos necessários para ensinar o conceito.
3. **Executar o Plano de Resposta**:
   - Siga o plano passo a passo.
   - Promova o pensamento crítico, incentivando o estudante a pensar e encontrar soluções por conta própria.

### Exemplo de Saída:
- **Plano de Resposta**:
  1. Revisar conceito X.
  2. Aplicar exemplo Y.
  3. Fazer perguntas reflexivas para verificar a compreensão.
- **Resposta Final**:
  "Vamos revisar o conceito X, aplicando o exemplo Y para entender melhor. Em seguida, farei algumas perguntas para verificar sua compreensão."

### Nota:
- NUNCA DE A RESPOSTA. Sempre guie o estudante para a solução.
"""

class ChatAgent:
    def __init__(self, student_profile, execution_plan, mongo_uri, database_name, session_id, user_email, disciplina, model_name="gpt-4o-mini"):
        self.student_profile = student_profile
        self.execution_plan = execution_plan  # Plano de execução mantido no __init__
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

        self.chain = self._setup_chain()

    def _setup_chain(self):
        """Configura a cadeia de prompts para análise e execução da resposta."""

        # Primeira etapa: Análise da pergunta e geração do plano
        plan_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="history"),
            ("ai", "perfil do estudante: {perfil_do_estudante}"),
            ("ai", "plano de execução: {plano_de_execucao}"),
            ("system", PROMPT_COMBINADO),
            ("human", "{input}"),
        ])

        # Configuração das cadeias individuais
        chain = plan_prompt | self.model | StrOutputParser()

        return chain

    async def invoke(self, user_input: str, history):
        """Executa a cadeia configurada para fornecer a resposta final."""
        try:
            print(f"Received message: {user_input}")

            # Prepara as entradas para o fluxo
            inputs = {
                "input": user_input,
                "perfil_do_estudante": self.student_profile,
                "plano_de_execucao": self.execution_plan,
                "history": history,
                "agent_scratchpad": [BaseMessage],
            }

            # Invoca a cadeia sequencial
            result = await self.chain.ainvoke(inputs)

            # Acessa a resposta final

            # (Opcional) Salvar no histórico da conversa
            # self.history.add_message(BaseMessage(content=user_input, role="human"))
            # self.history.add_message(AIMessage(content=final_response, role="assistant"))

            return result

        except Exception as e:
            print(f"Erro ao processar a mensagem: {e}")
            return "Ocorreu um erro ao processar sua mensagem."
        



        
# PROMPT_AGENTE_CONVERSASIONAL = """
# Você é o Agente de Pensamento Crítico, responsável por ensinar o estudante e promover o desenvolvimento do pensamento crítico.

# ## Responsabilidades:
# - **Ensino do Conteúdo**: Apresente conceitos claros e adaptados ao nível do estudante.
# - **Comunicação Eficaz**: Use exemplos e uma linguagem apropriada ao perfil do estudante.
# - **Desenvolvimento do Pensamento Crítico**: Incentive a reflexão e a solução independente de problemas.
# - **Uso de Histórico**: Acompanhe o histórico do estudante para personalizar a resposta e garantir continuidade.

# ## Entrada:
# - **Perfil do Estudante**: {perfil_do_estudante}
# - **Plano de Execução**: {plano_de_execucao}
# - **Histórico de Interações**: {historico_de_interacoes}

# ## Exemplo de Fluxo:
# *Entrada*: "Como resolver uma equação diferencial?"
# *Resposta*: "Vamos resolver juntos! O que você já sabe sobre integrais? Podemos começar por aí."

# ---

# ## Regras:
# 1. **Personalize a resposta**: Utilize o perfil e histórico do estudante.
# 2. **Inicie perguntas reflexivas**: Encoraje o aluno a pensar e encontrar soluções.
# 3. **Garanta alinhamento**: Verifique se a resposta segue o plano de execução.

# ## Nota, 
# utilize o formato abaixo para fazer o registro de cada interação:
# Thought: [Sua análise do que fazer]
# Action: [Nome da ação a ser realizada]
# Action Input: [Parâmetros da ação]
# Observation: [Resultado da ação]
# Thought: [Análise após observar o resultado]
# ... (Repita o ciclo, se necessário)
# Final Answer: [Resposta final ao usuário]
# """


# class ChatAgent:
#     def __init__(self, student_profile, execution_plan, mongo_uri, database_name, session_id, user_email, disciplina, model_name="gpt-4o-mini"):
#         self.student_profile = student_profile
#         self.execution_plan = execution_plan
#         self.session_id = session_id
#         self.user_email = user_email
#         self.disciplina = disciplina

#         # Inicializa o modelo de linguagem
#         self.model = ChatOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)

#         # Histórico de conversas no MongoDB
#         self.mongo_client = MongoClient(mongo_uri)
#         self.history_collection = self.mongo_client[database_name]['chat_history']
#         self.long_term_memory_collection = self.mongo_client[database_name]['long_term_memory']

#         # Prompt do agente
#         self.prompt = ChatPromptTemplate.from_messages([
#             ("system", PROMPT_AGENTE_CONVERSASIONAL),
#             MessagesPlaceholder(variable_name="history"),
#             ("human", "{input}")
#         ])

#         # Parser ReAct
#         self.output_parser = ReActOutputParser()

#         # Define tools para o agente
#         self.tools = [
#             Tool(name="retrieve_chat_history", func=self.retrieve_chat_history, description="Resgata o histórico do chat."),
#             Tool(name="insert_long_term_memory", func=self.insert_long_term_memory, description="Insere memória de longo prazo."),
#             Tool(name="add_long_term_memory_to_agent", func=self.add_long_term_memory_to_agent, description="Adiciona memória ao agente.")
#         ]

#         # Cria o agente ReAct
#         self.agent = create_react_agent(
#             llm=self.model,
#             prompt=self.prompt,
#             output_parser=self.output_parser
#         )

#         # Executor do agente
#         self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

#     def retrieve_chat_history(self, session_id: str) -> str:
#         """Resgata o histórico de chat do MongoDB."""
#         try:
#             history_cursor = self.history_collection.find({"session_id": session_id}).sort("timestamp", 1)
#             history = [json.loads(doc.get('history', '{}')) for doc in history_cursor]
#             return json.dumps(history)
#         except Exception as e:
#             print(f"Erro ao resgatar o histórico de chat: {e}")
#             return "Erro ao resgatar o histórico de chat."

#     def insert_long_term_memory(self, memory: dict) -> str:
#         """Insere memória de longo prazo no MongoDB."""
#         try:
#             memory['timestamp'] = datetime.now(timezone.utc)
#             self.long_term_memory_collection.insert_one(memory)
#             return "Memória de longo prazo inserida com sucesso."
#         except Exception as e:
#             print(f"Erro ao inserir memória de longo prazo: {e}")
#             return "Erro ao inserir memória de longo prazo."

#     def add_long_term_memory_to_agent(self, memory: dict) -> str:
#         """Adiciona memória de longo prazo ao agente."""
#         try:
#             # Adiciona a memória ao histórico
#             self.history_collection.insert_one({
#                 "session_id": self.session_id,
#                 "user_email": self.user_email,
#                 "disciplina": self.disciplina,
#                 "memory": memory,
#                 "timestamp": datetime.now(timezone.utc)
#             })
#             return "Memória adicionada ao agente com sucesso."
#         except Exception as e:
#             print(f"Erro ao adicionar memória ao agente: {e}")
#             return "Erro ao adicionar memória ao agente."

#     async def invoke(self, user_input: str) -> str:
#         """Processa a entrada do usuário e retorna a resposta do agente."""
#         try:
#             # Obter o histórico de mensagens
#             history_messages = self.history_collection.find({"session_id": self.session_id})

#             # Formatar o prompt
#             formatted_prompt = self.prompt.format(
#                 perfil_do_estudante=self.student_profile,
#                 plano_de_execucao=self.execution_plan,
#                 historico_de_interacoes="\n".join([msg['content'] for msg in history_messages]),
#                 input=user_input
#             )

#             # Invocar o agente para gerar a resposta
#             response = await self.agent_executor.ainvoke({"input": user_input, "history": list(history_messages)})

#             # Processar a resposta
#             parsed_response = self.output_parser.parse(response)
#             print(f"Resposta Parseada: {parsed_response}")

#             # Salvar a interação no histórico
#             self.history_collection.insert_one({
#                 "session_id": self.session_id,
#                 "user_email": self.user_email,
#                 "disciplina": self.disciplina,
#                 "history": parsed_response,
#                 "timestamp": datetime.now(timezone.utc)
#             })

#             return parsed_response
#         except Exception as e:
#             print(f"Erro ao processar a mensagem: {e}")
#             return "Ocorreu um erro ao processar sua mensagem."