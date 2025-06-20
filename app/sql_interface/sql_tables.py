# app/sql_interface/sql_tables.py

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, ForeignKey, Date, DateTime, Boolean, Enum, JSON, TIMESTAMP, text, Text, TIME, LargeBinary
from sqlalchemy.dialects.postgresql import UUID
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
    Column('SenhaHash', String(255), nullable=True),
    Column('TipoUsuario', Enum('student', 'educator', 'admin', name='user_type_enum'), nullable=False),
    Column('Instituicao', String(100), nullable=True),
    Column('TipoDeConta', Enum('google', 'email', name='account_type_enum'), nullable=False),
    Column('CriadoEm', TIMESTAMP, server_default=text('CURRENT_TIMESTAMP')),
    Column('AtualizadoEm', TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'), onupdate=text('CURRENT_TIMESTAMP'))
)

tabela_estudantes = Table('Estudantes', metadata,
    Column('IdEstudante', Integer, primary_key=True),
    Column('IdUsuario', Integer, ForeignKey('Usuarios.IdUsuario'), nullable=False, index=True),
    Column('Matricula', String(50), nullable=True, unique=True)
)

tabela_educadores = Table('Educadores', metadata,
    Column('IdEducador', Integer, primary_key=True),
    Column('IdUsuario', Integer, ForeignKey('Usuarios.IdUsuario'), nullable=False, index=True),
    Column('EspecializacaoDisciplina', String(100), nullable=False)
)

tabela_estudante_curso = Table('EstudanteCurso', metadata,
    Column('Id', Integer, primary_key=True),
    Column('IdEstudante', Integer, ForeignKey('Estudantes.IdEstudante'), nullable=False, index=True),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso'), nullable=False, index=True),
    Column('CriadoEm', TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'))
)

tabela_cursos = Table('Cursos', metadata,
    Column('IdCurso', Integer, primary_key=True),
    Column('IdEducador', Integer, ForeignKey('Educadores.IdEducador'), nullable=True, index=True),
    Column('NomeEducador', String(100), nullable=True),
    Column('NomeCurso', String(100), nullable=False),
    Column('Ementa', Text),
    Column('Objetivos', Text),
    Column('HorarioInicio', TIME, nullable=False),
    Column('HorarioFim', TIME, nullable=False),
    Column('CriadoEm', TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'))
)

tabela_perfil_aprendizado_aluno = Table('PerfilAprendizadoAluno', metadata,
    Column('IdPerfil', Integer, primary_key=True),
    Column('IdUsuario', Integer, ForeignKey('Usuarios.IdUsuario'), nullable=False, index=True),
    Column('DadosPerfil', JSON, nullable=False),  # Armazenando o perfil completo em JSON
    Column('IdPerfilFelderSilverman', Integer, ForeignKey('PerfisFelderSilverman.IdPerfil'), nullable=True),
    Column('DataUltimaAtualizacao', DateTime, server_default=text('CURRENT_TIMESTAMP'), onupdate=text('CURRENT_TIMESTAMP'))  # Rastreando a última atualização
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
    Column('Avaliacao', String(100), nullable=True),
    Column('HorarioInicio', TIME, nullable=False),
    Column('HorarioFim', TIME, nullable=False),
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
    Column('Inicio', DateTime, nullable=True),
    Column('Fim', DateTime, nullable=True),
    Column('Produtividade', Integer, nullable=True),
    Column('FeedbackDoAluno', Text, nullable=True),
    Column('HistoricoConversa', JSON, nullable=True),
    Column('PreferenciaHorario', String(50), nullable=True),
)

tabela_recursos_aprendizagem = Table('RecursosAprendizagem', metadata,
    Column('IdRecurso', Integer, primary_key=True),
    Column('IdCurso', Integer, ForeignKey('Cursos.IdCurso'), nullable=True, index=True),  # Recurso pode ser compartilhado
    Column('IdUsuario', Integer, ForeignKey('Usuarios.IdUsuario'), nullable=False, index=True),
    Column('Titulo', String(200), nullable=False),
    Column('Tipo', Enum('video', 'documento', 'link', 'outro', name='recurso_tipo_enum'), nullable=False),
    Column('URL', String(500)),
    Column('CaminhoArquivo', String(500), nullable=True),  # Caminho para o arquivo armazenado
    Column('EnviadoPor', Integer, ForeignKey('Usuarios.IdUsuario'), nullable=True),  # Quem enviou
    Column('FonteAutomatica', Boolean, default=False),  # Carregado automaticamente, ex: via Classroom
    Column('Descricao', Text),
    Column('VectorId', UUID(as_uuid=True), nullable=True),  # Integração com o banco vetorial
    Column('CriadoEm', TIMESTAMP, server_default=text('CURRENT_TIMESTAMP')),
    Column('AtualizadoEm', TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'), onupdate=text('CURRENT_TIMESTAMP'))
)

tabela_feedback_ia = Table('FeedbackIAPerfil', metadata,
    Column('IdFeedback', Integer, primary_key=True),
    Column('IdEstudante', Integer, ForeignKey('Estudantes.IdEstudante'), nullable=False, index=True),
    Column('TipoFeedback', String(50), nullable=False),
    Column('ConteudoFeedback', Text),
    Column('DataAvaliacao', DateTime, server_default=text('CURRENT_TIMESTAMP'))
)

tabela_eventos_calendario = Table('EventosCalendario', metadata,
    Column('IdEvento', Integer, primary_key=True),
    Column('GoogleEventId', String(250), nullable=True, unique=True),
    Column('Titulo', String(200), nullable=False),
    Column('Descricao', Text),
    Column('Inicio', DateTime, nullable=False),
    Column('Fim', DateTime, nullable=False),
    Column('Local', String(200)),
    Column('Categoria', Enum('aula', 'estudo_individual', 'prova', 'estudo_em_grupo', 'tarefa', 'outro', name='evento_categoria_enum'), nullable=False),
    Column('Importancia', Enum('Urgente', 'alta', 'media', 'baixa', name='evento_importancia_enum'), nullable=False),
    Column('Material', String(200)),
    Column('CriadoPor', Integer, ForeignKey('Usuarios.IdUsuario'), nullable=False, index=True)
)

tabela_perfis_felder_silverman = Table('PerfisFelderSilverman', metadata,
    Column('IdPerfil', Integer, primary_key=True),
    Column('NomePerfil', String(50), nullable=False, unique=True),
    Column('Descricao', Text, nullable=False),
    Column('Recomendacoes', Text)
)

tabela_feedback_plataforma = Table('FeedbackPlataforma', metadata,
    Column('IdFeedback', Integer, primary_key=True),
    Column('IdUsuario', Integer, ForeignKey('Usuarios.IdUsuario'), nullable=False, index=True),
    Column('TipoFeedback', Enum('bug', 'sugestao', 'elogio', name='user_type_enum'), nullable=False),
    Column('ConteudoFeedback', Text),
    Column('DataAvaliacao', DateTime, server_default=text('CURRENT_TIMESTAMP'))
)

tabela_suporte_tecnico = Table('SuporteTecnico', metadata,
    Column('IdChamado', Integer, primary_key=True),
    Column('IdUsuario', Integer, ForeignKey('Usuarios.IdUsuario'), nullable=False, index=True),
    Column('TipoChamado', Enum('duvida', 'problema', 'sugestao', name='chamado_tipo_enum'), nullable=False),
    Column('ConteudoChamado', Text),
    Column('DataChamado', DateTime, server_default=text('CURRENT_TIMESTAMP'))
)

tabela_notificacoes = Table('Notificacoes', metadata,
    Column('IdNotificacao', Integer, primary_key=True),
    Column('IdUsuario', Integer, ForeignKey('Usuarios.IdUsuario'), nullable=False, index=True),
    Column('Titulo', String(200), nullable=False),
    Column('Descricao', Text),
    Column('DataEnvio', DateTime, server_default=text('CURRENT_TIMESTAMP')),
    Column('Lida', Boolean, default=False)
)

tabela_support_requests = Table('SupportRequests', metadata,
    Column('IdSupportRequest', UUID(as_uuid=True), primary_key=True),
    Column('UserEmail', String(100), ForeignKey('Usuarios.Email'), nullable=False),
    Column('MessageType', String(50), nullable=False),
    Column('Subject', String(200), nullable=False),
    Column('Page', String(200), nullable=False),
    Column('Message', Text, nullable=False),
    Column('CreatedAt', TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'))
)

tabela_support_request_images = Table('SupportRequestImages', metadata,
    Column('IdImage', Integer, primary_key=True),
    Column('IdSupportRequest', UUID(as_uuid=True), ForeignKey('SupportRequests.IdSupportRequest'), nullable=False, index=True),
    Column('ImageData', LargeBinary, nullable=False)
)

#metadata.create_all(engine)
