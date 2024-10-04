from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, ForeignKey, Date, DateTime, Boolean, Enum, JSON, TIMESTAMP, text, Index, Text, LargeBinary
from sqlalchemy.dialects.postgresql import UUID
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
    Column('Email', String(100), unique=True, nullable=False, index=True),
    Column('SenhaHash', String(255), nullable=False),
    Column('TipoUsuario', Enum('student', 'educator', 'admin', name='user_type_enum'), nullable=False),
    Column('CriadoEm', TIMESTAMP, server_default=text('CURRENT_TIMESTAMP')),
    Column('AtualizadoEm', TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'), onupdate=text('CURRENT_TIMESTAMP'))
)

tabela_estudantes = Table('Estudantes', metadata,
    Column('IdEstudante', Integer, primary_key=True),
    Column('IdUsuario', Integer, ForeignKey('Usuarios.IdUsuario'), nullable=False, index=True),
    Column('Matricula', String(50), nullable=True, unique=True),
)

tabela_educadores = Table('Educadores', metadata,
    Column('IdEducador', Integer, primary_key=True),
    Column('IdUsuario', Integer, ForeignKey('Usuarios.IdUsuario'), nullable=False, index=True),
    Column('Instituicao', String(100), nullable=False),
    Column('EspecializacaoDisciplina', String(100), nullable=False)
)

tabela_cursos = Table('Cursos', metadata,
    Column('IdCurso', Integer, primary_key=True),
    Column('IdEducador', Integer, ForeignKey('Educadores.IdEducador'), nullable=True, index=True),
    Column('NomeCurso', String(100), nullable=False),
    Column('Ementa', Text),
    Column('Objetivos', Text),
    Column('CriadoEm', TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'))
)

tabela_perfil_aprendizado_aluno = Table('PerfilAprendizadoAluno', metadata,
    Column('IdPerfil', Integer, primary_key=True),
    Column('IdEstudante', Integer, ForeignKey('Estudantes.IdEstudante'), nullable=False, index=True),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso'), nullable=False, index=True),
    Column('TipoPerfil', String(50), nullable=False),
    Column('PreferenciaEstudo', String(50)),
    Column('PerfilAvaliado', Text),
)

tabela_cronograma = Table('Cronograma', metadata,
    Column('IdCronograma', Integer, primary_key=True),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso'), nullable=False, index=True),
    Column('NomeCronograma', String(100), nullable=False)
)

tabela_encontros = Table('Encontros', metadata,
    Column('IdEncontro', Integer, primary_key=True),
    Column('IdCronograma', Integer, ForeignKey('Cronograma.IdCronograma'), nullable=False, index=True),
    Column('NumeroEncontro', Integer, nullable=False),
    Column('DataEncontro', Date, nullable=False),
    Column('Conteudo', Text, nullable=False),
    Column('Estrategia', Text, nullable=True),
    Column('Avaliacao', String(100), nullable=True)  # Pode ser nulo
)

tabela_atividades = Table('Atividades', metadata,
    Column('IdAtividade', Integer, primary_key=True),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso'), nullable=False, index=True),
    Column('Titulo', String(200), nullable=False),
    Column('Descricao', Text),
    Column('DataEntrega', Date)
)

tabela_notas_aluno = Table('ProgressoAluno', metadata,
    Column('IdNota', Integer, primary_key=True),
    Column('IdEstudante', Integer, ForeignKey('Estudantes.IdEstudante'), nullable=False, index=True),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso'), nullable=False, index=True),
    Column('AV1', Float),
    Column('AV2', Float)
)

tabela_sessoes_estudo = Table('SessoesEstudo', metadata,
    Column('IdSessao', Integer, primary_key=True),
    Column('IdEstudante', Integer, ForeignKey('Estudantes.IdEstudante'), nullable=False, index=True),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso'), nullable=False, index=True),
    Column('Assunto', String(200), nullable=False),
    Column('Inicio', DateTime, server_default=text('CURRENT_TIMESTAMP')),
    Column('Fim', DateTime),
    Column('Produtividade', Integer),
    Column('FeedbackDoAluno', Text)
)

tabela_recursos_aprendizagem = Table('RecursosAprendizagem', metadata,
    Column('IdRecurso', Integer, primary_key=True),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso'), nullable=False, index=True),
    Column('Titulo', String(200), nullable=False),
    Column('Tipo', String(50)),
    Column('URL', String(500)),
    Column('Conteudo', Text),
    Column('VectorId', UUID(as_uuid=True), nullable=True)
)

tabela_feedback_ia = Table('FeedbackIAPerfil', metadata,
    Column('IdFeedback', Integer, primary_key=True),
    Column('IdEstudante', Integer, ForeignKey('Estudantes.IdEstudante'), nullable=False, index=True),
    Column('TipoFeedback', String(50), nullable=False),
    Column('ConteudoFeedback', Text),
    Column('DataAvaliacao', DateTime, server_default=text('CURRENT_TIMESTAMP'))
)

tabela_historico_perguntas_respostas_llm = Table('HistoricoPerguntasRespostasLLM', metadata,
    Column('IdPerguntaResposta', Integer, primary_key=True),
    Column('DataHoraPergunta', DateTime, server_default=text('CURRENT_TIMESTAMP'), nullable=False),
    Column('ConteudoPergunta', Text, nullable=False),
    Column('DataHoraResposta', DateTime),
    Column('ConteudoResposta', Text),
    Column('ConfidenciaResposta', Float),
    Column('TipoPergunta', String(50), nullable=False)
)

tabela_eventos_calendario = Table('EventosCalendario', metadata,
    Column('IdEvento', Integer, primary_key=True),
    Column('GoogleEventId', String(100), nullable=False, unique=True),
    Column('Titulo', String(200), nullable=False),
    Column('Descricao', Text),
    Column('Inicio', DateTime, nullable=False),
    Column('Fim', DateTime, nullable=False),
    Column('Local', String(200)),
    Column('CriadoPor', Integer, ForeignKey('Usuarios.IdUsuario'), nullable=False, index=True)
)

tabela_sessoes_estudo_perguntas_respostas = Table('SessoesEstudoPerguntasRespostas', metadata,
    Column('Id', Integer, primary_key=True),
    Column('IdSessao', Integer, ForeignKey('SessoesEstudo.IdSessao'), nullable=False, index=True),
    Column('IdPerguntaResposta', Integer, ForeignKey('HistoricoPerguntasRespostasLLM.IdPerguntaResposta'), nullable=False, index=True)
)

tabela_perfis_felder_silverman = Table('PerfisFelderSilverman', metadata,
    Column('IdPerfil', Integer, primary_key=True),
    Column('NomePerfil', String(50), nullable=False, unique=True),
    Column('Descricao', Text, nullable=False),
    Column('Recomendacoes', Text)
)

metadata.create_all(engine)
