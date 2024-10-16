import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


### Construct retriever ###
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


### Contextualize question ###
CONTEXTUALIZE_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
# contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
# history_aware_retriever = create_history_aware_retriever(
#     llm, retriever, contextualize_q_prompt
# )


# ### Answer question ###
# qa_system_prompt = """You are an assistant for question-answering tasks. \
# Use the following pieces of retrieved context to answer the question. \
# If you don't know the answer, just say that you don't know. \
# Use three sentences maximum and keep the answer concise.\

# {context}"""
# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", qa_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# ### Statefully manage chat history ###
# store = {}


# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]


# conversational_rag_chain = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     output_messages_key="answer",
# )

# conversational_rag_chain.invoke(
#     {"input": "What is Task Decomposition?"},
#     config={
#         "configurable": {"session_id": "abc123"}
#     },  # constructs a key "abc123" in `store`.
# )["answer"]


META_PROMPT = """
Dada uma descrição de tarefa ou um prompt existente, produza um prompt de sistema detalhado para guiar um modelo de linguagem a completar a tarefa de maneira eficaz.

# Diretrizes

- Entenda a Tarefa: Compreenda o principal objetivo, metas, requisitos, restrições e a saída esperada.
- Alterações Mínimas: Se um prompt existente for fornecido, melhore-o apenas se for simples. Para prompts complexos, melhore a clareza e adicione elementos ausentes sem alterar a estrutura original.
- Raciocínio Antes das Conclusões**: Incentive etapas de raciocínio antes de chegar a conclusões. ATENÇÃO! Se o usuário fornecer exemplos onde o raciocínio ocorre depois, INVERTA a ordem! NUNCA COMECE EXEMPLOS COM CONCLUSÕES!
    - Ordem do Raciocínio: Identifique as partes de raciocínio do prompt e as partes de conclusão (campos específicos pelo nome). Para cada uma, determine a ORDEM em que isso é feito e se precisa ser invertido.
    - Conclusões, classificações ou resultados devem SEMPRE aparecer por último.
- Exemplos: Inclua exemplos de alta qualidade, se forem úteis, usando placeholders [entre colchetes] para elementos complexos.
   - Que tipos de exemplos podem precisar ser incluídos, quantos e se são complexos o suficiente para se beneficiar de placeholders.
- Clareza e Concisão: Use linguagem clara e específica. Evite instruções desnecessárias ou declarações genéricas.
- Formatação: Use recursos do markdown para legibilidade. NÃO USE ``` BLOCO DE CÓDIGO A MENOS QUE SEJA ESPECIFICAMENTE SOLICITADO.
- Preserve o Conteúdo do Usuário: Se a tarefa de entrada ou o prompt incluir diretrizes ou exemplos extensos, preserve-os inteiramente ou o mais próximo possível. Se forem vagos, considere dividir em subetapas. Mantenha quaisquer detalhes, diretrizes, exemplos, variáveis ou placeholders fornecidos pelo usuário.
- Constantes: Inclua constantes no prompt, pois não são suscetíveis a injeções de prompt. Tais como guias, rubricas e exemplos.
- Formato de Saída: Explique explicitamente o formato de saída mais apropriado, em detalhes. Isso deve incluir comprimento e sintaxe (por exemplo, frase curta, parágrafo, JSON, etc.)
    - Para tarefas que produzem dados bem definidos ou estruturados (classificação, JSON, etc.), dê preferência à saída em formato JSON.
    - O JSON nunca deve ser envolvido em blocos de código (```) a menos que explicitamente solicitado.

O prompt final que você gera deve seguir a estrutura abaixo. Não inclua comentários adicionais, apenas gere o prompt completo do sistema. ESPECIFICAMENTE, não inclua mensagens adicionais no início ou no fim do prompt. (por exemplo, sem "---")

[Instrução concisa descrevendo a tarefa - esta deve ser a primeira linha do prompt, sem cabeçalho de seção]

[Detalhes adicionais conforme necessário.]

[Seções opcionais com títulos ou listas para etapas detalhadas.]

# Etapas [opcional]

[opcional: um detalhamento das etapas necessárias para realizar a tarefa]

# Formato de Saída

[Especificamente, aponte como a saída deve ser formatada, seja o comprimento da resposta, estrutura, por exemplo, JSON, markdown, etc.]

# Exemplos [opcional]

[Opcional: 1-3 exemplos bem definidos com placeholders, se necessário. Marque claramente onde os exemplos começam e terminam e qual é a entrada e saída. Use placeholders conforme necessário.]
[Se os exemplos forem mais curtos do que o esperado para um exemplo real, faça uma referência com () explicando como exemplos reais devem ser mais longos / curtos / diferentes. E USE PLACEHOLDERS!]

# Notas [opcional]

[opcional: casos extremos, detalhes e uma área para repetir considerações importantes específicas]
""".strip()

