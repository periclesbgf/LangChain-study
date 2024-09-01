from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, ForeignKey, Date, DateTime, Interval
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

user = os.getenv("POSTGRES_USER")
password = os.getenv("PASSWORD")
host = os.getenv("HOST")
port = os.getenv("PORT")
database = os.getenv("DATABASE")

engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
metadata = MetaData()

tabela_alunos = Table('Alunos', metadata,
    Column('MatriculaAluno', Integer, primary_key=True),
    Column('Nome', String(100), nullable=False),
    Column('Email', String(100), unique=True, nullable=False),
    Column('UltimoLogin', DateTime),
)

tabela_perfil_aprendizado_aluno = Table('PerfilAprendizadoAluno', metadata,
    Column('IdPerfil', Integer, primary_key=True),
    Column('MatriculaAluno', Integer, ForeignKey('Alunos.MatriculaAluno')),
    Column('Curso', String(100), ForeignKey('Cursos.NomeCurso')),
    Column('TipoPerfil', String(50), nullable=False),
    Column('PreferenciaEstudo', String(50)),
    Column('PerfilAvaliado', String),
)

tabela_cursos = Table('Cursos', metadata,
    Column('IdCurso', Integer, primary_key=True),
    Column('NomeCurso', String(100), nullable=False),
    Column('Descricao', String),
)

tabela_cronograma = Table('Cronograma', metadata,
    Column('IdCronograma', Integer, primary_key=True),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso')),
    Column('NomeCronograma', String(100), nullable=False)
)

tabela_encontros = Table('Encontros', metadata,
    Column('IdEncontro', Integer, primary_key=True),
    Column('IdCronograma', Integer, ForeignKey('Cronograma.IdCronograma')),
    Column('NumeroEncontro', Integer, nullable=False),
    Column('DataEncontro', Date, nullable=False),
    Column('Conteudo', String, nullable=False),
    Column('Estrategia', String, nullable=False)
)

tabela_atividades = Table('Atividades', metadata,
    Column('IdAtividade', Integer, primary_key=True),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso')),
    Column('Titulo', String(200), nullable=False),
    Column('Descricao', String),
    Column('DataEntrega', Date)
)

tabela_notas_aluno = Table('ProgressoAluno', metadata,
    Column('IdNota', Integer, primary_key=True),
    Column('MatriculaAluno', Integer, ForeignKey('Alunos.MatriculaAluno')),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso')),
    Column('AV1', Float),
    Column('AV2', Float)
)

tabela_sessoes_estudo = Table('SessoesEstudo', metadata,
    Column('IdSessao', Integer, primary_key=True),
    Column('MatriculaAluno', Integer, ForeignKey('Alunos.MatriculaAluno')),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso')),
    Column('Assunto', String(200), nullable=False),
    Column('Inicio', DateTime),
    Column('Fim', DateTime),
    Column('Produtividade', Integer),
    Column('FeedbackDoAluno', String)
)

tabela_recursos_aprendizagem = Table('RecursosAprendizagem', metadata,
    Column('IdRecurso', Integer, primary_key=True),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso')),
    Column('Titulo', String(200), nullable=False),
    Column('Tipo', String(50)),
    Column('URL', String),
    Column('Conteudo', String)
)

tabela_feedback_ia = Table('FeedbackIAPerfil', metadata,
    Column('IdFeedback', Integer, primary_key=True),
    Column('MatriculaAluno', Integer, ForeignKey('Alunos.MatriculaAluno')),
    Column('TipoFeedback', String(50)),
    Column('ConteudoFeedback', String),
    Column('DataAvaliacao', DateTime)
)

tabela_historico_perguntas_respostas_llm = Table('HistoricoPerguntasRespostasLLM', metadata,
    Column('IdPerguntaResposta', Integer, primary_key=True),
    Column('DataHoraPergunta', DateTime),
    Column('ConteudoPergunta', String),
    Column('DataHoraResposta', DateTime),
    Column('ConteudoResposta', String),
    Column('ConfidenciaResposta', Float),
    Column('TipoPergunta', String(50))
)

tabela_sessoes_estudo_perguntas_respostas = Table('SessoesEstudoPerguntasRespostas', metadata,
    Column('Id', Integer, primary_key=True),
    Column('IdSessao', Integer, ForeignKey('SessoesEstudo.IdSessao')),
    Column('IdPerguntaResposta', Integer, ForeignKey('HistoricoPerguntasRespostasLLM.IdPerguntaResposta'))
)

metadata.create_all(engine)