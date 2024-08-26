from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

user = os.getenv("POSTGRES_USER")
password = os.getenv("PASSWORD")
host = os.getenv("HOST")
port = os.getenv("PORT")
database = os.getenv("DATABASE")

# Criar a conexão com o banco de dados PostgreSQL
engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
metadata = MetaData()

# Refletir a tabela existente "Programa"
programa_table = Table('Programa', metadata, autoload_with=engine)

# Criar uma sessão
Session = sessionmaker(bind=engine)
session = Session()

# Dados extraídos do PDF
dados_programa = [
    (1, '17/08/2021', '90 minutos teóricos', 'Introdução à Programação.', 'Apresentação do curso, conceitos básicos de paradigmas de programação, introdução a C e Haskell, regras da CESAR School.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (2, '20/08/2021', '90 minutos teóricos', 'Estruturas de Decisão e Repetição.', 'Aula teórica explorando blocos de decisão (if-then-else) e repetição (for e while loops).', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (3, '23/08/2021', '60 minutos práticos', 'Acompanhamento de Projeto.', 'Acompanhamento individual de projeto e esclarecimento de dúvidas.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (4, '24/08/2021', '90 minutos teóricos', 'Matrizes e Funções.', 'Descrever como implementar uma matriz em C e desenvolver funções para modularizar o código.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (5, '27/08/2021', '90 minutos teóricos', 'Funções e Structs.', 'Continuação sobre funções com exemplos e introdução a structs.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (6, '31/08/2021', '90 minutos teóricos', 'Kick-off', None, 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (7, '03/09/2021', '90 minutos teóricos', 'Structs e Ponteiros', 'Continuação sobre structs com exemplos e introdução ao conceito de ponteiros.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (8, '06/09/2021', '0 minutos', 'Feriado', None, None, None),
    (9, '07/09/2021', '0 minutos', 'Feriado', None, None, None),
    (10, '10/09/2021', '90 minutos teóricos', 'Ponteiros e Alocação Dinâmica de Memória.', 'Continuação sobre ponteiros com exemplos e introdução à alocação dinâmica de memória.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (11, '13/09/2021', '60 minutos práticos', 'Acompanhamento de Projeto.', 'Acompanhamento individual de projeto e esclarecimento de dúvidas.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (12, '14/09/2021', '90 minutos teóricos', 'Alocação Dinâmica de Memória e Listas Encadeadas.', 'Continuação sobre alocação dinâmica de memória e introdução a listas encadeadas.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (13, '17/09/2021', '90 minutos teóricos', 'Listas Encadeadas, Pilhas e Filas.', 'Continuação sobre listas encadeadas com exemplos e introdução a pilhas e filas.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (14, '20/09/2021', '60 minutos práticos', 'Acompanhamento de Projeto.', 'Acompanhamento individual de projeto e esclarecimento de dúvidas.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (15, '21/09/2021', '90 minutos teóricos', 'Recursão.', 'Explorar o conceito de funções recursivas com exemplos.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (16, '24/09/2021', '90 minutos teóricos', 'Recursão e Revisão.', 'Continuação dos conceitos de recursão e revisão geral da unidade com exercícios.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (17, '27/09/2021', '60 minutos práticos', 'Acompanhamento de Projeto.', 'Acompanhamento individual de projeto e esclarecimento de dúvidas.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (18, '28/09/2021', '90 minutos teóricos', 'Revisão.', 'Aula prática com exercícios de revisão sobre todos os tópicos vistos ao longo do semestre.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (19, '01/10/2021', '90 minutos teóricos', 'Primeira Avaliação.', 'Prova prática com uma lista de projetos na linguagem de programação C.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (20, '04/10/2021', '60 minutos práticos', 'Acompanhamento de Projeto.', 'Acompanhamento individual de projeto e esclarecimento de dúvidas.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (21, '05/10/2021', '90 minutos teóricos', 'Aula de Feedback e Conceitos Finais de C.', 'Feedback sobre a primeira unidade e conceitos finais de C, incluindo preprocessadores, bibliotecas, leitura e escrita de arquivos.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (22, '08/10/2021', '90 minutos teóricos', 'Conceitos Finais de C e Introdução a Haskell.', 'Estrutura de um programa em C e introdução a Haskell com o compilador Glasgow.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (23, '11/10/2021', '0 minutos', 'Feriado', None, None, None),
    (24, '12/10/2021', '0 minutos', 'Feriado', None, None, None),
    (25, '15/10/2021', '90 minutos teóricos', 'Introdução a Haskell e Listas.', 'Continuação da introdução a Haskell e implementação de listas em Haskell.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (26, '18/10/2021', '60 minutos práticos', 'Acompanhamento de Projeto.', 'Acompanhamento individual de projeto e esclarecimento de dúvidas.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (27, '19/10/2021', '90 minutos teóricos', 'Listas e Tuplas.', 'Continuação sobre listas com exemplos e introdução a tuplas.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (28, '22/10/2021', '90 minutos teóricos', 'Tuplas e Tipos Algébricos.', 'Continuação sobre tuplas com exemplos e descrição de tipos algébricos.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (29, '25/10/2021', '60 minutos práticos', 'Acompanhamento de Projeto.', 'Acompanhamento individual de projeto e esclarecimento de dúvidas.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (30, '26/10/2021', '90 minutos teóricos', 'Tipos Algébricos e Polimorfismo.', 'Continuação sobre tipos algébricos e introdução ao polimorfismo.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (31, '29/10/2021', '90 minutos teóricos', 'Polimorfismo e Type Class.', 'Continuação de polimorfismo com exemplos e introdução a type class.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (32, '01/11/2021', '0 minutos', 'Feriado', None, None, None),
    (33, '02/11/2021', '0 minutos', 'Feriado', None, None, None),
    (34, '05/11/2021', '90 minutos teóricos', 'Type Class e Funções de Alta Ordem.', 'Continuação de type class com exemplos e introdução a funções de alta ordem.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (35, '08/11/2021', '60 minutos práticos', 'Acompanhamento de Projeto.', 'Acompanhamento individual de projeto e esclarecimento de dúvidas.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (36, '09/11/2021', '90 minutos teóricos', 'Funções de Alta Ordem e Revisão.', 'Continuação sobre funções de alta ordem e revisão do assunto visto na unidade com exemplos.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (37, '12/11/2021', '60 minutos práticos', 'Revisão.', 'Resolução de exercícios cobrindo todo o assunto visto na unidade.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (38, '15/11/2021', '0 minutos', 'Feriado', None, None, None),
    (39, '16/11/2021', '60 minutos práticos', 'Revisão.', 'Resolução de exercícios cobrindo todo o assunto visto na unidade.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (40, '19/11/2021', '90 minutos teóricos', 'Segunda Avaliação.', 'Prova prática com projetos de programação em Haskell.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (41, '23/11/2021', '90 minutos teóricos', 'Encerramento do Semestre.', 'Aula final sobre a disciplina, troca de ideias e feedbacks.', 'Computador, Slides, Conexão Wifi, Zoom Meeting', 'Sala de Aula Virtual (Zoom)'),
    (42, '10/12/2021', '0 minutos', 'Segunda Chamada', None, None, None),
    (43, '15/12/2021', '0 minutos', 'Avaliação Final', None, None, None)
]

# Função para converter string de data para objeto datetime
def str_to_date(date_str):
    return datetime.strptime(date_str, '%d/%m/%Y')

# Inserir dados na tabela "Programa"
for encontro, data, carga_horaria, conteudo, estrategia, recursos, espaco in dados_programa:
    session.execute(programa_table.insert().values(
        encontro=encontro,
        data=str_to_date(data),
        carga_horaria=carga_horaria,
        conteudo=conteudo,
        estrategia=estrategia if estrategia is not None else None,
        recursos=recursos if recursos is not None else None,
        espaco=espaco if espaco is not None else None
    ))

# Commit e fechar a sessão
session.commit()
session.close()
