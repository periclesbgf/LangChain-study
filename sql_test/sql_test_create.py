from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, ForeignKey, Date, DateTime, Interval, CheckConstraint
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

user = os.getenv("POSTGRES_USER")
password = os.getenv("PASSWORD")
host = os.getenv("HOST")
port = os.getenv("PORT")
database = os.getenv("DATABASE")

# Create connection to PostgreSQL database
engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
metadata = MetaData()

# Tabela de Alunos
tabela_alunos = Table('Alunos', metadata,
    Column('MatriculaAluno', Integer, primary_key=True),
    Column('Nome', String(100), nullable=False),
    Column('Email', String(100), unique=True, nullable=False),
    Column('PreferenciaEstudo', String(50)),
    Column('UltimoLogin', DateTime)
)

# Tabela de Cursos
tabela_cursos = Table('Cursos', metadata,
    Column('IdCurso', Integer, primary_key=True),
    Column('NomeCurso', String(100), nullable=False),
    Column('Descricao', String),
    Column('MatriculaAluno', Integer, ForeignKey('Alunos.MatriculaAluno'))
)

tabela_cronograma = Table('Cronograma', metadata,
    Column('IdCronograma', Integer, primary_key=True),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso')),
    Column('NomeCronograma', String(100), nullable=False)
)

# Tabela de Encontros
tabela_encontros = Table('Encontros', metadata,
    Column('IdEncontro', Integer, primary_key=True),
    Column('IdCronograma', Integer, ForeignKey('Cronograma.IdCronograma')),
    Column('NumeroEncontro', Integer, nullable=False),
    Column('DataEncontro', Date, nullable=False),
    Column('Conteudo', String, nullable=False),
    Column('Estrategia', String, nullable=False)
)

# Tabela de Atividades
tabela_atividades = Table('Atividades', metadata,
    Column('IdAtividade', Integer, primary_key=True),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso')),
    Column('Titulo', String(200), nullable=False),
    Column('Descricao', String),
    Column('DataEntrega', Date)
)

# Tabela de Progresso do Aluno
tabela_progresso_aluno = Table('ProgressoAluno', metadata,
    Column('IdProgresso', Integer, primary_key=True),
    Column('MatriculaAluno', Integer, ForeignKey('Alunos.MatriculaAluno')),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso')),
    Column('IdAtividade', Integer, ForeignKey('Atividades.IdAtividade')),
    Column('DataConclusao', Date),
    Column('Nota', Float)
)

# Tabela de Preferências de Aprendizagem
tabela_preferencias_aprendizagem = Table('PreferenciasAprendizagem', metadata,
    Column('IdPreferencia', Integer, primary_key=True),
    Column('MatriculaAluno', Integer, ForeignKey('Alunos.MatriculaAluno')),
    Column('TipoPreferencia', String(50), nullable=False),
    Column('ValorPreferencia', String)
)

# Tabela de Sessões de Estudo
tabela_sessoes_estudo = Table('SessoesEstudo', metadata,
    Column('IdSessao', Integer, primary_key=True),
    Column('MatriculaAluno', Integer, ForeignKey('Alunos.MatriculaAluno')),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso')),
    Column('Inicio', DateTime),
    Column('Fim', DateTime),
    Column('Duracao', Interval),
    Column('Produtividade', Integer),
    Column('FeedbackDoAluno', String)
)

# Tabela de Recursos de Aprendizagem
tabela_recursos_aprendizagem = Table('RecursosAprendizagem', metadata,
    Column('IdRecurso', Integer, primary_key=True),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso')),
    Column('Titulo', String(200), nullable=False),
    Column('Tipo', String(50)),
    Column('URL', String),
    Column('Conteudo', String)
)

# Tabela de Interações do Aluno
tabela_interacoes_aluno = Table('InteracoesAluno', metadata,
    Column('IdInteracao', Integer, primary_key=True),
    Column('MatriculaAluno', Integer, ForeignKey('Alunos.MatriculaAluno')),
    Column('IdRecurso', Integer, ForeignKey('RecursosAprendizagem.IdRecurso')),
    Column('TipoInteracao', String(50)),
    Column('DataInteracao', DateTime),
    Column('Duracao', Interval)
)

# Tabela de Feedback de IA
tabela_feedback_ia = Table('FeedbackIA', metadata,
    Column('IdFeedback', Integer, primary_key=True),
    Column('MatriculaAluno', Integer, ForeignKey('Alunos.MatriculaAluno')),
    Column('TipoFeedback', String(50)),
    Column('ConteudoFeedback', String),
    Column('DataCriacao', DateTime)
)

tabela_historico_perguntas_respostas_llm = Table('HistoricoPerguntasRespostasLLM', metadata,
    Column('IdPerguntaResposta', Integer, primary_key=True),
    Column('MatriculaAluno', Integer, ForeignKey('Alunos.MatriculaAluno')),
    Column('DataHoraPergunta', DateTime),
    Column('ConteudoPergunta', String),
    Column('DataHoraResposta', DateTime),
    Column('ConteudoResposta', String),
    Column('ConfidenciaResposta', Float),
    Column('TipoPergunta', String(50))
)

metadata.create_all(engine)