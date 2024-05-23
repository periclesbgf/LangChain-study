from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


class LanguageChain:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "gpt-4o"
        self.prompt_template = """
        1. Você é um assistente virtual chamado Éden, você é capaz de receber perguntas, comandos ou afirmações.
        2. Seu papel é responder perguntas de maneira amigável.
        3. Diferencie se um texto vindo do usuário é uma pergunta, comando ou afirmação.
        4. Você possui uma lista de comandos disponíveis para serem executados.
        5. Os comandos incluem: "ligar luminária", "desligar luminária", "ligar luz", "desligar luz", "travar porta", "destravar porta",\
            "checar bomba de água", "checar sensor de temperatura", "ligar válvula", "desligar válvula", "ligar bomba de água", "desligar bomba de água".
        6. Se a entrada do usuário for um comando: Sua tarefa é determinar se a entrada de um usuário é um desses comandos específicos\
            ou algo que se relacione com esses comandos. Se for um comando,\
            retorne exatamente o comando que você entendeu que o usuário quer executar, sem alterar a estrutura e nem adicionar texto a mais.
        7. Se a entrada do usuário for uma pergunta: Sua tarefa é responder a pergunta de maneira amigável e informativa.
        8. Se a entrada do usuário for uma afirmação: Sua tarefa é responder a afirmação de maneira amigável e informativa.
        9. Se a entrada do usuário for algo que não faz sentido: Sua tarefa é responder: 'Desculpe, não entendi.'
        10. Você deve utilizar no máximo 70 palavras para responder a cada pergunta e responde-las toda em Portugues Brasileiro.

        EXEMPLO_1:
            USUÁRIO: "Ligue a luminária."
            ÉDEN: "ligar a luminária"

        EXEMPLO_2:
            USUÁRIO: "Qual é a temperatura atual?"
            ÉDEN: "checar sensor de temperatura"

        EXEMPLO_3:
            USUÁRIO: "Acenda a luz."
            ÉDEN: "ligar luz"

        EXEMPLO_4:
            USUÁRIO: "Quanto é 1 + 1?"
            ÉDEN: "Um mais um é igual a dois."

        EXEMPLO_4:
            USUÁRIO: "Abra a porta."
            ÉDEN: "destravar porta"

        Dado o contexto acima, responda o texto a seguir: {text}
        """

    def setup_chain(self, text):
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        llm = ChatOpenAI(model=self.model, api_key=self.api_key)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        return chain.invoke(text)


class SQLChain:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "gpt-4-turbo"
        self.prompt_template = """
            Você é um especialista em tranformar frases em consultas sql.
            - Como resposta você deverá dar a consulta SQL referente a frase do usuário e nada mais.
            - A tabela do código sql deve estar entre aspas duplas "".
            - A coluna do código sql deve estar entre aspas duplas "".
            No banco de dados há as seguintes tabelas SQL:

            Table('Chassis', metadata,
            Column('id', Integer, primary_key=True),
            Column('Chassi', Integer, unique=True),
            Column('Modelo', Integer),
            Column('Cliente', Integer),
            Column('Contrato', Integer)
            )

            Table('Telemetria', metadata,
                Column('id', Integer, primary_key=True),
                Column('Chassi', Integer, ForeignKey('Chassis.Chassi')),
                Column('UnidadeMedida', String),
                Column('Categoria', String),
                Column('Data', Date),
                Column('Serie', String),
                Column('Valor', Numeric)
            )

            As informações referentes a coluna "Categoria" é que ela pode representar dados relativos ao Uso do Motor,Uso do Combustível do Motor ou
            Uso da Configuração do Modo do Motor e a coluna "Serie" pode ter como entrada: Chave-Ligada, Marcha Lenta, Carga Baixa, Carga Média, Carga Alta.

            Context:

            pergunta: Qual é o contrato associado ao chassi 808420?
            resposta: SELECT "Contrato" FROM "Chassis" WHERE "Chassi"= 808420;

            pergunta: Quantas categorias diferentes de informações são enviadas pelas máquinas?
            resposta: SELECT COUNT(DISTINCT "Categoria") FROM "Telemetria";

            pergunta: qual chassi do contrato 007 retorna mais informações?
            resposta: SELECT t."Chassi", COUNT(*) AS Quantidade FROM "Telemetria" t JOIN "Chassis" c ON t."Chassi" = c."Chassi" WHERE c."Contrato" = '007' GROUP BY t."Chassi" ORDER BY Quantidade DESC LIMIT 1;

            pergunta: Qual é a média dos valores enviados por cada categoria?
            resposta: SELECT "Categoria", AVG("Valor") AS MediaValor FROM "Telemetria" GROUP BY "Categoria";

            Dado o contexto acima, transforme o texto a seguir em uma consulta SQL: {text}
        """
        self.context = """
            Você é um assistente virtual preparado para receber uma pergunta e também os dados relacionados a essa pergunta\
            Você deve responder a pergunta de maneira amigável e informativa.\
            Essa pergunta vem de um usuário que deseja pesquisar dados em uma base de dados.\
            os dados que virá junto com a pergunta são referentes a uma tabela chamada "Telemetria" com colunas: "Chassi","UnidadeMedida","Categoria","Data","Serie","Valor".\
            e tambem de uma tabela chamada "Chassis" com colunas :"Chassi","Contrato","Cliente","Modelo".\
            Seu objetivo é responder a pergunta que se relacione com os dados do contexto acima.\

            Dado a pergunta do usuário {user_question}, o contexto acima, e os {data} recuperados de uma consulta a um banco de dados\
            responda a pergunta do usuário.
        """

    def setup_sql_chain(self, text):
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        llm = ChatOpenAI(model=self.model, api_key=self.api_key, temperature=0.1)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        return chain.invoke(text)


class AnswerChain:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "gpt-4o"
        self.prompt_template = """
            Você é um assistente virtual preparado para receber uma pergunta, os dados relacionados a essa pergunta e a consulta SQL da pergunta.\
            Você deve responder a pergunta de maneira direta.\
            Essa pergunta vem de um usuário que deseja pesquisar dados em uma base de dados.\
            os dados que virá junto com a pergunta são referentes a uma tabela chamada "Telemetria" com colunas: "Chassi","UnidadeMedida","Categoria","Data","Serie","Valor".\
            e tambem de uma tabela chamada "Chassis" com colunas :"Chassi","Contrato","Cliente","Modelo".\
            Seu objetivo é responder a pergunta que se relacione com os dados do contexto acima.\
            Você também receberá uma consulta SQL que foi gerada a partir da pergunta do usuário.\
            Use essa consulta para verificar quais dados foram retornados a partir do SELECT e responda a pergunta do usuário.

            Dado a pergunta do usuário {user_question}, o contexto acima, os {data} recuperados de uma consulta a um banco de dados\
            e também a aconsulta SQL: {query}, responda a pergunta do usuário.
        """

    def setup_chain(self, user_question, data, query):
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        llm = ChatOpenAI(model=self.model, api_key=self.api_key)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        print("user_question: ", user_question)
        print("data: ", data)
        print("query: ", query)
        response = chain.invoke({'user_question': user_question, 'data': data, 'query': query})
        print("response: ", response)
        return response


class AnswerChainLLamaIndex:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "gpt-4o"
        self.prompt_template = """
            Você é um assistente virtual preparado para receber uma pergunta e também os dados relacionados a essa pergunta\
            Você deve responder a pergunta de maneira direta.\
            Essa pergunta vem de um usuário que deseja pesquisar dados em uma base de dados.\
            os dados que virá junto com a pergunta são referentes a uma tabela chamada "Telemetria" com colunas: "Chassi","UnidadeMedida","Categoria","Data","Serie","Valor".\
            e tambem de uma tabela chamada "Chassis" com colunas :"Chassi","Contrato","Cliente","Modelo".\
            Seu objetivo é responder a pergunta que se relacione com os dados do contexto acima.\

            Dado a pergunta do usuário {user_question}, o contexto acima, e os {data} recuperados de uma consulta a um banco de dados\
            responda a pergunta do usuário.
        """

    def setup_chain(self, user_question, data):
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        llm = ChatOpenAI(model=self.model, api_key=self.api_key)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        print("user_question: ", user_question)
        print("data: ", data)
        response = chain.invoke({'user_question': user_question, 'data': data})
        print("response: ", response)
        return response


# def setup_chain(text):
#     """configuring chain and prompt template for the chain"""

#     prompt = """
#        1. Você é um assistente virtual chamado Éden, você é capaz de receber perguntas, comandos ou afirmações.
#        2. Seu papel é responder perguntas de maneira amigável.
#        3. Diferencie se um texto vindo do usuário é uma pergunta, comando ou afirmação.
#        4. Você possui uma lista de comandos disponíveis para serem executados.
#        5. Os comandos incluem: "ligar luminária", "desligar luminária", "ligar luz", "desligar luz", "travar porta", "destravar porta",\
#         "checar bomba de água", "checar sensor de temperatura", "ligar válvula", "desligar válvula", "ligar bomba de água", "desligar bomba de água".
#        6. Se a entrada do usuário for um comando: Sua tarefa é determinar se a entrada de um usuário é um desses comandos específicos\
#         ou algo que se relacione com esses comandos. Se for um comando,\
#         retorne exatamente o comando que você entendeu que o usuário quer executar, sem alterar a estrutura e nem adicionar texto a mais.
#        7. Se a entrada do usuário for uma pergunta: Sua tarefa é responder a pergunta de maneira amigável e informativa.
#        8. Se a entrada do usuário for uma afirmação: Sua tarefa é responder a afirmação de maneira amigável e informativa.
#        9. Se a entrada do usuário for algo que não faz sentido: Sua tarefa é responder: 'Desculpe, não entendi.'
#        10. Você deve utilizar no máximo 70 palavras para responder a cada pergunta.

#        EXEMPLO_1:
#         USUÁRIO: "Ligue a luminária."
#         ÉDEN: "ligar a luminária"

#        EXEMPLO_2:
#         USUÁRIO: "Qual é a temperatura atual?"
#         ÉDEN: "checar sensor de temperatura"

#        EXEMPLO_3:
#         USUÁRIO: "Acenda a luz."
#         ÉDEN: "ligar luz"

#        EXEMPLO_4:
#         USUÁRIO: "Quanto é 1 + 1?"
#         ÉDEN: "Um mais um é igual a dois."

#        EXEMPLO_4:
#         USUÁRIO: "Abra a porta."
#         ÉDEN: "destravar porta"

#        Dado o contexto acima, responda o texto a seguir: {text}
#     """

#     #summary_prompt_template = PromptTemplate(input_variables=["text"], template=summary_template)
#     prompt = ChatPromptTemplate.from_template(prompt)


#     llm = ChatOpenAI(
#         model="gpt-4-turbo",
#         api_key=openai_key,
#         #max_tokens=70,
#     )
#     output_parser = StrOutputParser()
#     chain = prompt | llm | output_parser
#     message = chain.invoke(text)


#     return message

def context_generator(question: str):
    context = """
        You are an AI language model assistant. Your task is to generate five different versions of the given user question to \
        retrieve relevant documents from a vector database. By generation multiple perspectives on the user question,\
        your goal is to help the user overcome some of the limitations of the distance-based similarity search.\
        Provide these alternative questions separated by newlines. Original question: {question}
    """
    return context

def prompt_template(question: str, context: str):
    template = """Answer the following questions based on this context:

    {context}

    Question: {question}
    """
