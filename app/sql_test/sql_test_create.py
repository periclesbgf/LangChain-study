from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, ForeignKey, Date, DateTime, Boolean, Enum, JSON, TIMESTAMP, text
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

tabela_usuarios = Table('Usuarios', metadata,
    Column('IdUsuario', Integer, primary_key=True),
    Column('Nome', String(100), nullable=False),
    Column('Email', String(100), unique=True, nullable=False),
    Column('SenhaHash', String(255), nullable=False),
    Column('TipoUsuario', Enum('student', 'educator', 'admin', name='user_type_enum'), nullable=False),
    Column('CriadoEm', TIMESTAMP, server_default=text('CURRENT_TIMESTAMP')),
    Column('AtualizadoEm', TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'), onupdate=text('CURRENT_TIMESTAMP'))
)

tabela_estudantes = Table('Estudantes', metadata,
    Column('IdEstudante', Integer, primary_key=True),
    Column('IdUsuario', Integer, ForeignKey('Usuarios.IdUsuario')),
    Column('Matricula', String(50)),
)
# Periodo e extrair turma com o email da school utilizando a API do classroom/calendar.
# Focar no cronograma/organizacao pessoal. Gerar um forms com perguntas para ja ter o perfil de aprendizado do aluno
# Nivel de produtividade
# Pensar em como vai ser a sessao de estudo. Dizer a arquitetura. De que outras funcionalidades iremos buscar dados.
# pode salvar sessao de estudo, pode salvar atividades, pode salvar notas, pode salvar feedbacks, pode salvar recursos de aprendizagem
# pode salvar perguntas e respostas, pode salvar eventos do calendario, pode salvar encontros, pode salvar cronograma, pode salvar cursos
# Construir um workspace para o aluno subir material de estudo e o sistema fazer a analise do material e sugerir conteudo
tabela_educadores = Table('Educadores', metadata,
    Column('IdEducador', Integer, primary_key=True),
    Column('IdUsuario', Integer, ForeignKey('Usuarios.IdUsuario')),
    Column('Instituicao', String(100)),
    Column('EspecializacaoDisciplina', String(100))
)

tabela_cursos = Table('Cursos', metadata,
    Column('IdCurso', Integer, primary_key=True),
    Column('IdEducador', Integer, ForeignKey('Educadores.IdEducador')),
    Column('NomeCurso', String(100), nullable=False),
    Column('Descricao', String),
    Column('CriadoEm', TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'))
)

tabela_perfil_aprendizado_aluno = Table('PerfilAprendizadoAluno', metadata,
    Column('IdPerfil', Integer, primary_key=True),
    Column('IdEstudante', Integer, ForeignKey('Estudantes.IdEstudante')),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso')),
    Column('TipoPerfil', String(50), nullable=False),
    Column('PreferenciaEstudo', String(50)),
    Column('PerfilAvaliado', String),
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
    Column('IdEstudante', Integer, ForeignKey('Estudantes.IdEstudante')),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso')),
    Column('AV1', Float),
    Column('AV2', Float)
)

tabela_sessoes_estudo = Table('SessoesEstudo', metadata,
    Column('IdSessao', Integer, primary_key=True),
    Column('IdEstudante', Integer, ForeignKey('Estudantes.IdEstudante')),
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
    Column('Conteudo', String),
    Column('VectorId', String(36), nullable=True)
)

tabela_feedback_ia = Table('FeedbackIAPerfil', metadata,
    Column('IdFeedback', Integer, primary_key=True),
    Column('IdEstudante', Integer, ForeignKey('Estudantes.IdEstudante')),
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

tabela_eventos_calendario = Table('EventosCalendario', metadata,
    Column('IdEvento', Integer, primary_key=True),
    Column('GoogleEventId', String, nullable=False),
    Column('Titulo', String(200), nullable=False),
    Column('Descricao', String),
    Column('Inicio', DateTime),
    Column('Fim', DateTime),
    Column('Local', String(200)),
    Column('CriadoPor', String(100))
)

tabela_sessoes_estudo_perguntas_respostas = Table('SessoesEstudoPerguntasRespostas', metadata,
    Column('Id', Integer, primary_key=True),
    Column('IdSessao', Integer, ForeignKey('SessoesEstudo.IdSessao')),
    Column('IdPerguntaResposta', Integer, ForeignKey('HistoricoPerguntasRespostasLLM.IdPerguntaResposta'))
)

metadata.create_all(engine)
