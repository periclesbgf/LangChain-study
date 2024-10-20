# # app/agent/agents.py

# from langchain_openai import ChatOpenAI
# from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool, initialize_agent, create_react_agent
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.tools import tool
# from database.mongo_database_manager import CustomMongoDBChatMessageHistory
# from utils import OPENAI_API_KEY
# from database.vector_db import QdrantHandler, Embeddings
# from langchain.memory.chat_message_histories import MongoDBChatMessageHistory
# from langchain.schema import BaseMessage, AIMessage, message_to_dict, messages_from_dict
# from pymongo import errors
# from datetime import datetime, timezone
# import json
# from langchain_openai import ChatOpenAI
# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.tools import tool
# from langchain.tools import tool
# from youtubesearchpython import VideosSearch
# import wikipediaapi
# from serpapi import GoogleSearch
# from pydantic import BaseModel
# from langchain.agents.react.output_parser import ReActOutputParser
# from functools import partial
# from langchain.agents import (
#     Agent,
#     AgentExecutor,
#     AgentOutputParser,
#     BaseMultiActionAgent,
#     BaseSingleActionAgent,
#     LLMSingleActionAgent,
# )

# from utils import OPENAI_API_KEY

# class CustomOutputParser:
#     """Parser personalizado para lidar com saídas do agente."""
#     def parse(self, output: str):
#         if "Observation:" in output:
#             # Se a resposta contiver "Observation:", continue o fluxo
#             return {"action": "continue", "observation": output}
#         # Caso contrário, retorne a saída final
#         return {"final_output": output}

# RETRIEVAL_PROMPT = """
# ### Contexto:
# Você é o **Agente de Recuperação de Conteúdo**, especializado em fornecer respostas precisas e personalizadas com base nas necessidades e preferências do estudante. Seu objetivo é identificar e sugerir recursos educativos úteis e alinhados ao contexto fornecido, utilizando as ferramentas disponíveis.

# ### Suas Responsabilidades:
# 1. **Buscar Recursos Relevantes**:
#    - Consulte o banco vetorial e outros recursos para localizar os materiais mais adequados.
#    - Priorize sempre a recuperação de contexto relevante (retrieve_context) antes de recorrer a ferramentas externas, como Google ou Wikipedia.

# 2. **Fornecer Sugestões Personalizadas**:
#    - Adapte os recursos recomendados às necessidades específicas e ao perfil do estudante.
#    - Inclua vídeos, exemplos práticos e artigos que correspondam ao estilo de aprendizado e plano de execução do estudante.

# 3. **Apresentar Respostas de Forma Clara e Estruturada**:
#    - Organize as recomendações para que sejam fáceis de seguir e implementar.
#    - Se necessário, inclua orientações para ações práticas ou próximas etapas.

# ### Parâmetros de Entrada:
# - **Consulta do Usuário**: "{consulta_usuario}"
# - **Perfil do Estudante**: {perfil_do_estudante}
# - **Plano de Execução**: {plano_de_execucao}

# ### Fluxo de Execução:
# 1. **Análise Inicial**: Interprete a consulta do usuário e identifique quais recursos serão mais úteis.
# 2. **Seleção da Ferramenta**: Escolha a ferramenta adequada com base na consulta e nos recursos disponíveis.
# 3. **Pesquisa e Recuperação**: Aplique a ferramenta selecionada para encontrar a melhor resposta possível.
# 4. **Refinamento e Personalização**: Adapte as recomendações ao perfil e ao plano de execução do estudante.
# 5. **Resposta ao Usuário**: Formule uma resposta final clara, contendo os recursos encontrados e orientações específicas.

# ### Ferramentas Disponíveis:
# {tool_names}

# - Utilize as ferramentas fornecidas conforme necessário. 
# - Priorize **retrieve_context** para fornecer contexto relevante sempre que possível.

# ### Rascunho das Interações Realizadas:
# {agent_scratchpad}

# ### Lista Completa de Ferramentas:
# {tools}

# ---

# ### Nota Importante:
# - Sempre priorize a ferramenta `retrieve_context` para fornecer contexto educativo relevante antes de utilizar Google ou Wikipedia.
# - Garanta que todas as respostas sejam apresentadas de forma clara e direcionada ao objetivo do estudante.

# ---

# ### Exemplo de Fluxo:
# Question: {consulta_usuario}
# Thought: [Sua análise do que fazer]
# Action: [Nome da ação a ser realizada]
# Action Input: [Parâmetros da ação]
# Observation: [Resultado da ação]
# Thought: [Análise após observar o resultado]
# ... (Repita o ciclo, se necessário)
# Final Answer: [Resposta final ao usuário]

# - **Formato das Queries**: string  
# - **Siga Estritamente o Formato** para garantir consistência e clareza na resposta.
# """

# class RetrievalAgent:
#     """
#     Agente responsável por recuperar e sugerir recursos educacionais com base nas consultas do estudante.

#     Atributos:
#         model (ChatOpenAI): Modelo de linguagem usado para interações.
#         qdrant_handler: Handler para o banco vetorial Qdrant.
#         embeddings: Objeto para geração de embeddings de consultas.
#         agent (AgentExecutor): Agente que executa as tarefas definidas.
#         prompt (ChatPromptTemplate): Template do prompt para o agente.
#         output_parser (ReActOutputParser): Parser para verificar a saída do agente.
#         student_email (str): Email do estudante.
#         disciplina (str): Disciplina do estudante.
#     """

#     def __init__(self, qdrant_handler, embeddings, student_email: str, disciplina: str, model_name: str = "gpt-4o-mini"):
#         """
#         Inicializa o RetrievalAgent com os componentes necessários.
        
#         Args:
#             qdrant_handler: Handler para consultas ao banco vetorial Qdrant.
#             embeddings: Gerador de embeddings para consultas.
#             student_email (str): Email do estudante.
#             disciplina (str): Disciplina do estudante.
#             model_name (str): Nome do modelo de linguagem a ser utilizado.
#         """
#         self.model = ChatOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)
#         self.qdrant_handler = qdrant_handler
#         self.embeddings = embeddings
#         self.student_email = student_email
#         self.disciplina = disciplina

#         self.prompt = ChatPromptTemplate.from_template(RETRIEVAL_PROMPT)
#         self.output_parser = ReActOutputParser()

#         # Define wrapper functions para as ferramentas
#         def retrieve_context_tool(query: str) -> str:
#             return self.retrieve_context(query)
#         retrieve_context_tool.__name__ = "retrieve_context"

#         def search_youtube_tool(query: str) -> str:
#             return self.search_youtube(query)
#         search_youtube_tool.__name__ = "search_youtube"

#         def search_wikipedia_tool(query: str) -> str:
#             return self.search_wikipedia(query)
#         search_wikipedia_tool.__name__ = "search_wikipedia"

#         def search_google_tool(query: str) -> str:
#             return self.search_google(query)
#         search_google_tool.__name__ = "search_google"

#         # Cria a lista de ferramentas
#         tools = [
#             Tool(name="retrieve_context", func=retrieve_context_tool, description="Recupera contexto."),
#             Tool(name="search_youtube", func=search_youtube_tool, description="Busca vídeos no YouTube."),
#             Tool(name="search_wikipedia", func=search_wikipedia_tool, description="Pesquisa na Wikipedia."),
#             Tool(name="search_google", func=search_google_tool, description="Pesquisa no Google."),
#         ]

#         self.agent = create_react_agent(
#             llm=self.model,
#             tools=tools,
#             prompt=self.prompt,
#             output_parser=self.output_parser
#         )

#         self.agent_executor = AgentExecutor(
#             agent=self.agent,
#             tools=tools,
#             handle_parsing_errors=True,
#             verbose=True,
#             max_iterations=30,
#             return_intermediate_steps=True  # Retorna as etapas intermediárias para análise
#         )


#     def retrieve_context(self, query: str) -> str:
#         """
#         Recupera contexto relevante do banco vetorial Qdrant com base em uma consulta.
#         """
#         print(f"retrieve_context called with query: {query}")  # Debug

#         if not isinstance(query, str):
#             raise ValueError("O parâmetro 'query' deve ser uma string.")

#         embedding = self.embeddings.embed_query(query)
#         print(f"Generated embedding: {embedding[:5]}...")  # Debug - mostra parte do embedding

#         results = self.qdrant_handler.similarity_search(
#             embedding=embedding,
#             student_email=self.student_email,
#             disciplina=self.disciplina,
#             k=5,
#         )
#         print(f"Retrieved {len(results)} relevant documents")  # Debug

#         # Combine o conteúdo dos documentos recuperados
#         context = "\n".join([result['content'] for result in results])
#         return context or "Nenhum contexto relevante encontrado."

#     def search_youtube(self, query: str) -> str:
#         """
#         Busca o vídeo mais relevante no YouTube com base na consulta.
#         """
#         print(f"Searching YouTube for: {query}")  # Debug

#         from youtubesearchpython import VideosSearch
#         search = VideosSearch(query, limit=1)
#         result = search.result()
#         print(f"Search Result: {result}")  # Debug

#         if result['result']:
#             video = result['result'][0]
#             return f"Título: {video['title']}\nLink: {video['link']}"
#         return "Nenhum vídeo encontrado."

#     def search_wikipedia(self, query: str) -> str:
#         """
#         Realiza uma pesquisa na Wikipedia e retorna o resumo da página encontrada.
#         """
#         import wikipediaapi
#         wiki = wikipediaapi.Wikipedia('pt')
#         page = wiki.page(query)

#         if page.exists():
#             return f"Título: {page.title}\nResumo: {page.summary[:500]}"
#         return "Página não encontrada."

#     def search_google(self, query: str) -> str:
#         """
#         Realiza uma pesquisa no Google e retorna o primeiro resultado relevante.
#         """
#         from serpapi import GoogleSearch
#         search = GoogleSearch({"q": query, "api_key": "YOUR_SERPAPI_KEY"})
#         result = search.get_dict()
#         if "organic_results" in result and result["organic_results"]:
#             top_result = result["organic_results"][0]
#             return f"Título: {top_result['title']}\nLink: {top_result['link']}"
#         return "Nenhum resultado encontrado."

#     async def invoke(self, query: dict, student_profile, execution_plan):
#         try:
#             query_str = " ".join([f"{k}: {v}" for k, v in query.items()])
#             inputs = {
#                 "consulta_usuario": query_str,
#                 "perfil_do_estudante": student_profile,
#                 "plano_de_execucao": execution_plan,
#                 "input": query_str
#             }

#             print(f"Invoking agent with inputs: {inputs}")  # Debug

#             response = await self.agent_executor.ainvoke(inputs)
#             print(f"Raw response from agent: {response}")  # Debug

#             if not response:
#                 print("Agent did not return a valid response.")  # Debug
#                 return "Não foi possível gerar uma resposta."

#             return response.get("resultado", "Nenhum resultado encontrado.")
#         except Exception as e:
#             print(f"Error executing query: {e}")  # Debug
#             return f"Erro ao executar a consulta: {e}"

# PROMPT_AGENTE_ANALISE_PROGRESSO = """
# Você é o Agente de Análise de Progresso. Sua responsabilidade é avaliar o desempenho do estudante e fornecer feedback corretivo, se necessário.

# ### Responsabilidades:
# - **Avaliar o Progresso**: Verifique se o estudante está avançando conforme o plano de execução.
# - **Fornecer Feedback**: Identifique áreas de dificuldade e sugira melhorias.
# - **Ajustar o Plano**: Sinalize se o plano precisa ser revisado.

# ### Entrada:
# - **Histórico de Interações**: {historico_de_interacoes}
# - **Progresso Atual da Sessão**: {session_progress}
# - **Plano de Execução**: {plano_de_execucao}

# ### Tarefas:
# 1. **Analisar o Histórico**: Examine as interações para identificar padrões de dificuldade.
# 2. **Comparar com o Plano**: Verifique se o progresso está alinhado com os objetivos definidos.
# 3. **Fornecer Feedback**: Prepare um relatório com sugestões e observações.

# **Exemplo de Feedback**:
# "O estudante tem demonstrado dificuldade com conceitos fundamentais de álgebra linear. Recomendo focar em exercícios básicos antes de avançar para tópicos mais complexos."
# """

# class ProgressAnalysisAgent:
#     def __init__(self, student_profile, execution_plan, mongo_uri, database_name, session_id, user_email, disciplina, model_name="gpt-4o"):
#         self.student_profile = student_profile
#         self.execution_plan = execution_plan
#         self.session_id = session_id
#         self.user_email = user_email
#         self.disciplina = disciplina
#         self.model = ChatOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)
#         self.history = CustomMongoDBChatMessageHistory(
#             user_email=self.user_email,
#             disciplina=self.disciplina,
#             connection_string=mongo_uri,
#             session_id=self.session_id,
#             database_name=database_name,
#             collection_name="chat_history",
#             session_id_key="session_id",
#             history_key="history",
#         )
#         self.prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", PROMPT_AGENTE_ANALISE_PROGRESSO),
#                 MessagesPlaceholder(variable_name="history"),
#             ]
#         )
#         self.agent = create_tool_calling_agent(self.model, [self.analyze_progress], self.prompt)
#         self.agent_executor = AgentExecutor(agent=self.agent, tools=[self.analyze_progress])

#     @tool
#     def analyze_progress(self):
#         """Analisa o progresso do estudante."""
#         return "Analisando o progresso do estudante."

#     def invoke(self, session_progress: str):
#         # Obter o histórico de mensagens
#         history_messages = self.history.messages
#         # Formatar o prompt
#         formatted_prompt = self.prompt.format(
#             historico_de_interacoes="\n".join([msg.content for msg in history_messages]),
#             session_progress=session_progress,
#             plano_de_execucao=self.execution_plan
#         )
#         # Invocar o agente
#         response = self.agent_executor.invoke({"history": history_messages})
#         return response
    


# PROMPT_AGENTE_CONVERSASIONAL = """
# Você é o Agente de Pensamento Crítico, o tutor principal responsável por ensinar o estudante, se comunicar de forma eficaz e promover o desenvolvimento do pensamento crítico.

# ### Responsabilidades:
# - **Ensino do Conteúdo**: Apresente conceitos de forma clara e adaptada ao nível do estudante.
# - **Comunicação Eficaz**: Use exemplos personalizados e linguagem apropriada ao perfil do estudante.
# - **Desenvolvimento do Pensamento Crítico**: Incentive o estudante a refletir e encontrar respostas por conta própria.

# ### Entrada:
# - **Perfil do Estudante**: {perfil_do_estudante}
# - **Plano de Execução**: {plano_de_execucao}
# - **Histórico de Interações**: {historico_de_interacoes}

# ### Tarefas:
# 1. **Responda de forma personalizada**: Use o perfil e plano do estudante para adaptar sua resposta.
# 2. **Inicie perguntas reflexivas**: Ajude o estudante a desenvolver habilidades críticas e resolver problemas fazendo perguntas e instigando o pensamento c ritico.
# 3. **Verifique o alinhamento com o plano**: Certifique-se de que sua resposta está de acordo com o plano de execução.
# 4. **Forneça exemplos práticos**: Use exemplos e analogias para facilitar a compreensão do estudante (se aplicável).
# 5. **Incentive a participação ativa**: Encoraje o estudante a se envolver na conversa e fazer perguntas.
# 6. **Analise as instrucoes recebidas do Agente Analitico para tomar acoes**: 

# **Exemplo de Interação**:
# *Entrada*: "Não entendo como resolver essa equação diferencial."
# *Resposta*: "Vamos resolver isso juntos. O que você já sabe sobre integrais? Talvez possamos começar por aí."

# **Formato de Saída**:
# - Uma resposta clara e relevante para o estudante promovendo o pensamento critico.
# """

# class ChatAgent:
#     def __init__(self, student_profile, execution_plan, mongo_uri, database_name, session_id, user_email, disciplina, model_name="gpt-4o-mini"):
#         self.student_profile = student_profile
#         self.execution_plan = execution_plan
#         self.session_id = session_id
#         self.user_email = user_email
#         self.disciplina = disciplina
#         self.model = ChatOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)
#         self.history = CustomMongoDBChatMessageHistory(
#             user_email=self.user_email,
#             disciplina=self.disciplina,
#             connection_string=mongo_uri,
#             session_id=self.session_id,
#             database_name=database_name,
#             collection_name="chat_history",
#             session_id_key="session_id",
#             history_key="history",
#         )
#         self.prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("ai", "{instruction}"),
#                 ("human", "{input}"),
#                 ("system", PROMPT_AGENTE_CONVERSASIONAL),
#             ]
#         )
#         self.chain = self.build_chain()

#     def build_chain(self):

#         chain = (
#             self.prompt
#             | self.model

#         )
#         return chain

#     def invoke(self, user_input: str):
#         # Obter o histórico de mensagens
#         history_messages = self.history.messages
#         # Formatar o prompt
#         formatted_prompt = self.prompt.format(
#             perfil_do_estudante=self.student_profile,
#             plano_de_execucao=self.execution_plan,
#             historico_de_interacoes="\n".join([msg.content for msg in history_messages]),
#             input=user_input
#         )
#         # Invocar o agente
#         response = self.invoke({"input": user_input, "history": history_messages})
#         # Salvar a interação no histórico
#         self.history.add_message(BaseMessage(content=user_input, role="human"))
#         self.history.add_message(AIMessage(content=response, role="assistant"))
#         return response