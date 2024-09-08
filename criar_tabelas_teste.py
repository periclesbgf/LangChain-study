from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, Date, ForeignKey, DECIMAL
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

# Definir tabelas
# Trabalhar melhor o coeficiente de rendimento com objetivo de trabalhar no metodo de ensino
alunos_table = Table('Alunos', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('nome', String(100), nullable=False),
    Column('matricula', String(50), unique=True, nullable=False),
    Column('email', String(100), nullable=False), #tirar
    Column('telefone', String(15), nullable=True), # tirar
    Column('curso', String(100), nullable=False),
    Column('periodo', Integer, nullable=False),
    Column('semestre', String(10), nullable=False)
)

disciplinas_table = Table('Disciplinas', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('nome', String(100), nullable=False),
    Column('codigo', String(50), unique=True, nullable=False),
    Column('carga_horaria', Integer, nullable=False),
    Column('curso', String(100), nullable=False),
    Column('periodo', Integer, nullable=False),
    Column('semestre', String(10), nullable=False),
    Column('ementa', Text, nullable=True),
    Column('objetivos', Text, nullable=True)
)

aulas_table = Table('Aulas', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('data', Date, nullable=False),
    Column('carga_horaria', Integer, nullable=False),
    Column('conteudo', Text, nullable=False),
    Column('disciplina_id', Integer, ForeignKey('Disciplinas.id'), nullable=False)
)

avaliacoes_table = Table('Avaliacoes', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('tipo', String(50), nullable=False),
    Column('data', Date, nullable=False),
    Column('disciplina_id', Integer, ForeignKey('Disciplinas.id'), nullable=False)
)

notas_table = Table('Notas', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('aluno_id', Integer, ForeignKey('Alunos.id'), nullable=False),
    Column('avaliacao_id', Integer, ForeignKey('Avaliacoes.id'), nullable=False),
    Column('nota', DECIMAL(5, 2), nullable=False)
)

programa_table = Table('Programa', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('encontro', Integer, nullable=False),
    Column('data', Date, nullable=False),
    Column('carga_horaria', String(50), nullable=False),
    Column('conteudo', Text, nullable=False),
    Column('estrategia', Text, nullable=True),
    Column('recursos', Text, nullable=True),
    Column('espaco', Text, nullable=True)
)

# Criar as tabelas no banco de dados
metadata.create_all(engine)
