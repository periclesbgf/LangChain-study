from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
import logging
from fastapi.logger import logger as fastapi_logger


class CommandChain:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "gpt-4o"
        self.prompt_template = """
        1. Você é um assistente virtual chamado Éden, você é capaz de receber perguntas, comandos ou afirmações.
        2. Seu papel é responder perguntas de maneira amigável.
        3. Diferencie se um texto vindo do usuário é uma pergunta, comando ou afirmação.
        4. Você possui uma lista de comandos disponíveis para serem executados.
        5. Os comandos incluem: "ligar luminária", "desligar luminária", "ligar luz", "desligar luz", "travar porta", "destravar porta",\
            "checar bomba de água", "ligar válvula", "desligar válvula", "ligar bomba de água", "desligar bomba de água".
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
        return chain.invoke(text), None


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



class SQLSchoolChain:
    """
    This class is responsible for setting up the chain for the SQL School task.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "gpt-4o"
        self.prompt_template_selec_tables = """
        [Contexto]
        Você é um especialista em bases de dados, capaz de analisar uma pergunta de um usuário e, a partir de um esquema de base de dados, descrever quais tabelas e colunas são relevantes para responder à pergunta.

        [Objetivo]
        Sua tarefa é analisar o texto fornecido e descrever quais tabelas e colunas são relevantes para responder à pergunta.

        [Instruções]
        - O formato de saída deve ser JSON na seguinte estrutura:

            "Nome da primeira tabela": ["lista de colunas relevantes"],
            "Nome da segunda tabela": ["lista de colunas relevantes"],
            ...
            "Nome da n-ésima tabela": ["lista de colunas relevantes"]


        - Em cada tabela com informações relevantes para responder à pergunta, ordene as colunas importantes em ordem decrescente, considerando a importância para responder à pergunta (ordenar da coluna mais relevante para a menos relevante).
        - Adicione apenas colunas que ajudem a responder à pergunta realizada.
        - Adicione apenas tabelas e colunas presentes no esquema apresentado.
        - Caso mais de uma tabela apresente informações importantes, lembre-se de adicionar na lista de colunas de ambas as tabelas as colunas que podem unir essas tabelas. Caso contrário, adicione apenas as colunas da tabela de interesse.

        [Exemplo 1]
        Pergunta: 'Quais disciplinas estou cursando atualmente?'

        Resposta (entre colchetes):
            "Disciplinas": ["Periodo", "CursoID"],
            "Cursos": ["NomeCurso", "CursoID"]

        [Exemplo 2]
        Pergunta: 'Em quais matérias fui para a final?'

        Resposta (entre colchetes):
            "Disciplinas": ["NomeDisciplina", "DisciplinaID"],
            "HistoricoEscolar": ["Final", "DisciplinaID"]

        [Tarefa]
        Analise o [Esquema] a seguir e descreva quais tabelas e colunas são relevantes para responder à pergunta.

        [Esquema]
        Tabelas:
            - Tabela: "Cursos"
            Colunas:
                - Nome: "CursoID"
                Tipo: Integer
                Chave Primária: True
                Descrição: ID único do curso
                Exemplo de dados: 1, 2, 3, ...
                - Nome: "NomeCurso"
                Tipo: String(100)
                Descrição: Nome do curso
                Exemplo de dados: "Ciência da Computação", "Design", "Sistemas de Informação", "Gestão de TI", "Análise e Desenvolvimento de Sistemas"

            - Tabela: "Disciplinas"
            Colunas:
                - Nome: "DisciplinaID"
                Tipo: Integer
                Chave Primária: True
                Descrição: ID único da disciplina
                Exemplo de dados: 1, 2, 3, ...
                - Nome: "NomeDisciplina"
                Tipo: String(100)
                Descrição: Nome da disciplina
                Exemplo de dados: "Modelagem e Projeto de BD", "Design de Produtos", "Gestão de Projetos"
                - Nome: "Periodo"
                Tipo: Integer
                Descrição: Período da disciplina
                Exemplo de dados: 1, 2, 3, ... 8
                - Nome: "CursoID"
                Tipo: Integer
                Chave Estrangeira: "Cursos"."CursoID"
                Descrição: ID do curso ao qual a disciplina pertence
                Exemplo de dados: 1, 2, 3, ...

            - Tabela: "AssuntosSemanais"
            Colunas:
                - Nome: "AssuntoID"
                Tipo: Integer
                Chave Primária: True
                Descrição: ID único do assunto semanal
                Exemplo de dados: 1, 2, 3, ...
                - Nome: "DisciplinaID"
                Tipo: Integer
                Chave Estrangeira: "Disciplinas"."DisciplinaID"
                Descrição: ID da disciplina associada ao assunto semanal
                Exemplo de dados: 1, 2, 3, ...
                - Nome: "Semana"
                Tipo: Integer
                Descrição: Número da semana
                Exemplo de dados: 1, 2, 3, ... 20
                - Nome: "Assunto"
                Tipo: String(255)
                Descrição: Descrição do assunto semanal
                Exemplo de dados: "Introdução ao Banco de Dados", "Planejamento de Projetos", "Introdução à Segurança da Informação", "Metodologias de Desenvolvimento"

            - Tabela: "HistoricoEscolar"
            Colunas:
                - Nome: "HistoricoID"
                Tipo: Integer
                Chave Primária: True
                Descrição: ID único do histórico escolar
                Exemplo de dados: 1, 2, 3, ...
                - Nome: "DisciplinaID"
                Tipo: Integer
                Chave Estrangeira: "Disciplinas"."DisciplinaID"
                Descrição: ID da disciplina no histórico escolar
                Exemplo de dados: 1, 2, 3, ...
                - Nome: "Semestre"
                Tipo: String(10)
                Descrição: Semestre em que a disciplina foi cursada
                Exemplo de dados: "2021.1", "2021.2", "2042.1"
                - Nome: "AV1"
                Tipo: Float
                Descrição: Nota da primeira avaliação (Importante: A AV1 abrange os assuntos das semanas 1 a 10)
                Exemplo de dados: 5.0, 7.5, 10.0
                - Nome: "AV2"
                Tipo: Float
                Descrição: Nota da segunda avaliação (Importante: A AV2 abrange os assuntos das semanas 11 a 20)
                Exemplo de dados: 5.0, 8.5, 7.0
                - Nome: "Final"
                Tipo: Float
                Descrição: Nota da avaliação final (Importante: A nota final é calculada a partir das notas da AV1 e AV2)
                Exemplo de dados: 2.0, 7.5, 10.0

        Aqui vai uma pergunta. Lembre-se com atenção das instruções e resolva o problema conforme solicitado:
        [Pergunta]
        {pergunta}
        """

        self.prompt_build_sql_query = """
        [Contexto]
        Você é um especialista em bases de dados, capaz de analisar as tabelas e colunas relevantes para responder a uma pergunta de um usuário.
        Dado um [Esquema] de um banco de dados, as tabelas e suas respectivas colunas relevantes como [TabelasImportantes] e a [Pergunta],
        sua tarefa é decompor a pergunta em subperguntas e gerar o código SQL correspondente utilizando PostgreSQL.
        Lembre-se que o ano letivo é dividido em dois semestres, sendo o primeiro semestre representado por "ano.1" e o segundo semestre por "ano.2".\
        Ou seja, o ano letivo 2023 é representado por "2023.1" e "2023.2".
        Lembre-se que o ano atual é 2024.1

        [Restrições]
        - As consultas geradas devem ser em PostgreSQL.
        - Utilize apenas nomes de tabelas e colunas presentes no esquema.
        - Selecione apenas as colunas necessárias para responder à pergunta, sem incluir colunas ou valores desnecessários.
        - Inclua apenas as tabelas necessárias em FROM ou JOIN.
        - Certifique-se de que as tabelas estão presentes na [TabelasImportantes].
        - Não utilize LIMIT com IN, ALL, ANY ou SOME em subconsultas.
        - Se utilizar funções MAX ou MIN, primeiro faça o JOIN da tabela, depois use SELECT MAX("coluna") ou SELECT MIN("coluna").
        - Não utilize ILIKE, LIKE, SIMILAR TO, ou outras comparações de string que não sejam =, <>, <, >, <=, >=.
        - Todas as tabelas e colunas devem ser referenciadas com o nome correto e entre aspas duplas.

        [Esquema]
        Tabelas:
            - Tabela: "Cursos"
            Colunas:
                - Nome: "CursoID"
                Tipo: Integer
                Chave Primária: True
                Descrição: ID único do curso
                Exemplo de dados: 1, 2, 3, ...
                - Nome: "NomeCurso"
                Tipo: String(100)
                Descrição: Nome do curso
                Exemplo de dados: "Ciência da Computação", "Design", "Sistemas de Informação", "Gestão de TI", "Análise e Desenvolvimento de Sistemas"

            - Tabela: "Disciplinas"
            Colunas:
                - Nome: "DisciplinaID"
                Tipo: Integer
                Chave Primária: True
                Descrição: ID único da disciplina
                Exemplo de dados: 1, 2, 3, ...
                - Nome: "NomeDisciplina"
                Tipo: String(100)
                Descrição: Nome da disciplina
                Exemplo de dados: "Modelagem e Projeto de BD", "Design de Produtos", "Gestão de Projetos"
                - Nome: "Periodo"
                Tipo: Integer
                Descrição: Período da disciplina
                Exemplo de dados: 1, 2, 3, ... 8
                - Nome: "CursoID"
                Tipo: Integer
                Chave Estrangeira: "Cursos"."CursoID"
                Descrição: ID do curso ao qual a disciplina pertence
                Exemplo de dados: 1, 2, 3, ...

            - Tabela: "AssuntosSemanais"
            Colunas:
                - Nome: "AssuntoID"
                Tipo: Integer
                Chave Primária: True
                Descrição: ID único do assunto semanal
                Exemplo de dados: 1, 2, 3, ...
                - Nome: "DisciplinaID"
                Tipo: Integer
                Chave Estrangeira: "Disciplinas"."DisciplinaID"
                Descrição: ID da disciplina associada ao assunto semanal
                Exemplo de dados: 1, 2, 3, ...
                - Nome: "Semana"
                Tipo: Integer
                Descrição: Número da semana
                Exemplo de dados: 1, 2, 3, ... 20
                - Nome: "Assunto"
                Tipo: String(255)
                Descrição: Descrição do assunto semanal
                Exemplo de dados: "Introdução ao Banco de Dados", "Planejamento de Projetos", "Introdução à Segurança da Informação", "Metodologias de Desenvolvimento"

            - Tabela: "HistoricoEscolar"
            Colunas:
                - Nome: "HistoricoID"
                Tipo: Integer
                Chave Primária: True
                Descrição: ID único do histórico escolar
                Exemplo de dados: 1, 2, 3, ...
                - Nome: "DisciplinaID"
                Tipo: Integer
                Chave Estrangeira: "Disciplinas"."DisciplinaID"
                Descrição: ID da disciplina no histórico escolar
                Exemplo de dados: 1, 2, 3, ...
                - Nome: "Semestre"
                Tipo: String(10)
                Descrição: Semestre em que a disciplina foi cursada
                Exemplo de dados: "2021.1", "2021.2", "2042.1"
                - Nome: "AV1"
                Tipo: Float
                Descrição: Nota da primeira avaliação (Importante: A AV1 abrange os assuntos das semanas 1 a 10)
                Exemplo de dados: 5.0, 7.5, 10.0
                - Nome: "AV2"
                Tipo: Float
                Descrição: Nota da segunda avaliação (Importante: A AV2 abrange os assuntos das semanas 11 a 20)
                Exemplo de dados: 5.0, 8.5, 7.0
                - Nome: "Final"
                Tipo: Float
                Descrição: Nota da avaliação final (Importante: A nota final é calculada a partir das notas da AV1 e AV2)
                Exemplo de dados: 2.0, 7.5, 10.0

        Para interpretar a pergunta do usuário, você pode levar em consideração o significado dos valores, mas para construir a query utilize os valores presentes na coluna.

        [Pergunta]
        {user_question}
        [TabelasImportantes]
        {importantTables}

        Decomponha a pergunta em subperguntas, considerando as [Restrições], e gere a consulta PostgreSQL final, pensando no passo a passo \
        (Gere apenas o valor da SQL FINAL, em uma linha só, sem traços especiais como ```):
        """

        self.final_prompt = """
        [Contexto]
        Você é um assistente virtual chamado Eden preparado para receber uma pergunta, os dados relacionados a essa pergunta e a consulta SQL da pergunta.\
        Você deve responder a pergunta de maneira direta.\
        Essa pergunta vem de um usuário que deseja pesquisar dados em uma base de dados.\
        os dados que virá junto com a pergunta são referentes a um [Esquema].\
        Você também receberá uma consulta SQL que foi gerada a partir da pergunta do usuário.\
        Use essa consulta para verificar quais dados foram retornados a partir do SELECT e responda a pergunta do usuário.
        Se por algum motivo retornoar None na consulta da nota, é porque essa nota não esta disponivel, \
        seja porquê a disciplina não foi cursada ou porque ele nao precisou fazer a prova por outro motivo.
        [Objetivo]
        Seu objetivo é responder a pergunta que se relacione com os dados do contexto acima.\

        Dado a pergunta do usuário {user_question}, o contexto acima, os {data} recuperados de uma consulta a um banco de dados\
        e também a aconsulta SQL: {query}, responda a pergunta do usuário.
        """

    def setup_chain(self, text):
        initial_prompt = ChatPromptTemplate.from_template(self.prompt_template_selec_tables)
        sql_query_prompt = ChatPromptTemplate.from_template(self.prompt_build_sql_query)

        llm = ChatOpenAI(model=self.model, api_key=self.api_key, temperature=0)

        output_parser_json = JsonOutputParser()
        output_parser = StrOutputParser()

        selec_tables_chain = (
            initial_prompt
            | llm
            | output_parser_json
        )
        sql_query_builder_chain = (
            sql_query_prompt
            | llm
            | output_parser
        )

        selec_tables_chain_result = selec_tables_chain.invoke({'pergunta': text})

        sql_query_builder_chain_result = sql_query_builder_chain.invoke({
            'importantTables': selec_tables_chain_result,
            'user_question': text
        })

        return sql_query_builder_chain_result, selec_tables_chain_result


    def output_chain(self, user_question, importantTables, data, query):
        final_prompt = ChatPromptTemplate.from_template(self.final_prompt)
        llm = ChatOpenAI(model=self.model, api_key=self.api_key, temperature=0.5)
        output_parser = StrOutputParser()
        chain = final_prompt | llm | output_parser

        response = chain.invoke({'user_question': user_question, 'importantTables': importantTables, 'data': data, 'query': query})

        return response


class ClassificationChain:
    """
    This class is responsible for setting up the chain for the Classification task.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "gpt-4o"
        self.prompt_clasificator = """
        [Contexto]
        Você é um especialista em analisar texto e determinar a classificação de uma questão em um tópico específico.
        Você tem três rotas possíveis para classificar a questão: `Comando`, `ConsultarBancoDeDados` ou `outros`.

        [Objetivo]
        Sua tarefa é classificar a questão em um dos tópicos acima.

        [Rotas]
        - Comando:
            [Descrição]
            Essa rota determina se a questão é um comando a ser executado em um sistema de automação residencial ou Se é uma solicitação de dados de um dispositivo em casa.

            [Exemplo]
            Questões que solicitam a execução de uma ação. Essa ação pode ser ligar ou desligar um dispositivo, abrir ou fechar uma porta, entre outras ações envolvendo automação residencial.
            Dentro de comandos, você pode encontrar palavras-chave como "ligar", "desligar", "abrir", "fechar".
            Ao escolher comando, a pergunta ou ação do usuário irá para uma determinada rota que irá passar por um processamento específico.
            Este processamento irá identificar o comando específico que o usuário deseja executar.
            Tambem irá identificar se o usuário quer solicitar dados de algum dispositivo em casa.

        - ConsultarBancoDeDados:
            [Descrição]
            Essa rota determina se a questão é uma solicitação de dados de um banco de dados envolvendo seu historico escolar. Ou algo referente a faculdade.\

            [Exemplos]
            Questões que solicitam informações sobre disciplinas cursadas, notas, semestres, cursos, entre outras informações relacionadas ao histórico escolar.
            Dentro de consultar banco de dados, você pode encontrar palavras-chave como "disciplina", "curso", "nota", "semestre".
            Ao escolher consultar banco de dados, a pergunta ou ação do usuário irá para uma determinada rota que irá passar por um processamento específico.
            Este processamento irá identificar as informações solicitadas pelo usuário e retornar a resposta correta.

        - Outros:
            [Descrição]
            Essa rota determina se a questão não se encaixa nas rotas de comando ou consultar banco de dados.

            [Exemplos]
            Questões que não se encaixam nas rotas de comando ou consultar banco de dados.
            Dentro de outros, pode encontrar perguntas diversas que não se encaixam em nenhuma das rotas anteriores.
            Esta rota irá identificar que a questão não se encaixa nas rotas de comando ou consultar banco de dados.

        Baseado no contexto acima, classifique a questão a seguir em uma das [Rotas] acima. Retorne apenas \
        o nome da [Rotas].  Não responda mais do que uma palavra:

        <pergunta>
        {question}
        </pergunta>
        """


    def setup_chain(self, text):
        prompt = ChatPromptTemplate.from_template(self.prompt_clasificator)
        llm = ChatOpenAI(model=self.model, api_key=self.api_key)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        return chain.invoke(text)


class DefaultChain():
    """
    This class is responsible for setting up the chain for the Default task.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "gpt-4o"
        self.prompt_template = """
        [Contexto]
        Você é um assistente virtual chamado Éden.

        [Objetivo]
        Seu papel é responder perguntas de maneira amigável.

        [Prgunta]
        {text}
        """
    def setup_chain(self, text):
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        llm = ChatOpenAI(model=self.model, api_key=self.api_key)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        return chain.invoke(text)


class RetrievalChain:
    """
    This class is responsible for setting up the chain for the Retrieval task.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "gpt-4o"
        self.prompt_template = """
        [Contexto]
        Você é um assistente virtual chamado Éden, capaz de ler informações recuperadas de um banco de dados vetorial\
        e responder perguntas com base nesses dados.
        Os documentos virão em formato JSON, contendo informações relevantes para responder a pergunta do usuário.
        Os documentos virão com informações quebradas em partes, cada parte contendo uma ou várias informações relevantes.
        Os documentos terão notas associadas a cada parte, indicando a relevância da informação para responder a pergunta.

        [Objetivo]
        Sua tarefa é responder a pergunta do usuário com base nos [dados] fornecidos.

        [Instruções]
        - Leia os documentos fornecidos.
        - Com base na pergunta do usuário, tente responder a pergunta utilizando as informações contidas nos documentos.
        - Utilize as informações contidas nos documentos para responder a pergunta do usuário.
        - Responda a pergunta de maneira amigável e informativa.

        [Pergunta]
        {text}

        [dados]
        {data}

        """

    def format_data_for_prompt(self, results):
        scores = [result[1] for result in results]
        contents = [result[0].page_content for result in results]

        formatted_data = {
            "scores": scores,
            "contents": contents
        }

        print("formatted_data: ", formatted_data)
        return formatted_data

    def setup_chain(self, text, data):
        prompt_template = """
        [Contexto]
        Você é um assistente virtual chamado Éden, capaz de ler informações recuperadas de um banco de dados vetorial\
        e responder perguntas com base nesses dados.
        Os documentos virão em formato JSON, contendo informações relevantes para responder a pergunta do usuário.
        Os documentos virão com informações quebradas em partes, cada parte contendo uma ou várias informações relevantes.
        Os documentos terão notas associadas a cada parte, indicando a relevância da informação para responder a pergunta.

        [Objetivo]
        Sua tarefa é responder a pergunta do usuário com base nos [dados] fornecidos.

        [Instruções]
        - Leia os documentos fornecidos.
        - Com base na pergunta do usuário, tente responder a pergunta utilizando as informações contidas nos documentos.
        - Utilize as informações contidas nos documentos para responder a pergunta do usuário.
        - Responda a pergunta de maneira amigável e informativa.

        [Pergunta]
        {text}

        [dados]
        {data}
        """

        documents = "\n\n".join(
            [f"Score: {score}\nContent: {content}" for score, content in zip(data["scores"], data["contents"])]
        )
        prompt_text = prompt_template.format(text=text, data=documents)

        prompt = ChatPromptTemplate.from_template(prompt_text)
        llm = ChatOpenAI(model=self.model, api_key=self.api_key)
        output_parser = StrOutputParser()

        chain = prompt | llm | output_parser

        response = chain.invoke({
            'text': text,
            'data': documents
        })
        print("response: ", response)
        return response
