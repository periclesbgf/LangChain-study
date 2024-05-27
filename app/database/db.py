from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Date, Numeric, ForeignKey
from sqlalchemy.orm import sessionmaker
from faker import Faker
import random
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

# Tabela de Professores
professores_table = Table('Professores', metadata,
    Column('ProfessorID', Integer, primary_key=True, autoincrement=True, unique=True),
    Column('Nome', String(100), nullable=False)
)

# Tabela de Coordenadores (subconjunto de Professores)
coordenadores_table = Table('Coordenadores', metadata,
    Column('CoordenadorID', Integer, primary_key=True, autoincrement=True, unique=True),
    Column('ProfessorID', Integer, ForeignKey('Professores.ProfessorID'), nullable=False, unique=True)
)

# Tabela de Cursos
cursos_table = Table('Cursos', metadata,
    Column('NomeCurso', String(100), nullable=False, unique=True, primary_key=True),
    Column('CoordenadorID', Integer, ForeignKey('Coordenadores.CoordenadorID'), nullable=False)
)

# Tabela de Disciplinas
disciplinas_table = Table('Disciplinas', metadata,
    Column('DisciplinaID', Integer, primary_key=True, autoincrement=True),
    Column('NomeDisciplina', String(100), nullable=False),
    Column('CursoID', String(100), ForeignKey('Cursos.NomeCurso')),
    Column('ProfessorID', Integer, ForeignKey('Professores.ProfessorID')),
    Column('Periodo', Integer, nullable=False)
)

# Tabela de Alunos
alunos_table = Table('Alunos', metadata,
    Column('AlunoID', Integer, primary_key=True, autoincrement=True),
    Column('Nome', String(100), nullable=False),
    Column('DataNascimento', Date),
    Column('DataMatricula', Date, nullable=False)
)

# Tabela de Matriculas
matriculas_table = Table('Matriculas', metadata,
    Column('MatriculaID', Integer, primary_key=True, autoincrement=True),
    Column('AlunoID', Integer, ForeignKey('Alunos.AlunoID')),
    Column('DisciplinaID', Integer, ForeignKey('Disciplinas.DisciplinaID')),
    Column('DataMatricula', Date, nullable=False)
)

# Tabela de Historico Escolar
historico_escolar_table = Table('HistoricoEscolar', metadata,
    Column('HistoricoID', Integer, primary_key=True, autoincrement=True),
    Column('AlunoID', Integer, ForeignKey('Alunos.AlunoID')),
    Column('DisciplinaID', Integer, ForeignKey('Disciplinas.DisciplinaID')),
    Column('DataConclusao', Date, nullable=False),
    Column('Nota', Numeric(4, 2))
)

metadata.create_all(engine)

# Instanciar Faker
fake = Faker('pt_BR')

# Criar uma sessão
Session = sessionmaker(bind=engine)
session = Session()

# Inserir Professores
professores = []
for _ in range(20):
    nome_completo = fake.name()
    professor_id = session.execute(professores_table.insert().values(Nome=nome_completo)).inserted_primary_key[0]
    professores.append(professor_id)

# Coordenadores (subconjunto de Professores)
coordenadores = random.sample(professores, 5)
for professor_id in coordenadores:
    session.execute(coordenadores_table.insert().values(ProfessorID=professor_id))
session.commit()

# Obter coordenadores IDs para associar com cursos
coordenador_ids = session.execute(coordenadores_table.select()).fetchall()

# Cursos
cursos = ['Ciência da Computação', 'Design', 'Gestão de TI', 'Análise e Desenvolvimento de Sistemas', 'Sistemas de Informação']
for curso, coordenador in zip(cursos, coordenador_ids):
    session.execute(cursos_table.insert().values(NomeCurso=curso, CoordenadorID=coordenador.CoordenadorID))
session.commit()

# Verificar se os cursos foram inseridos corretamente
cursos_inseridos = session.execute(cursos_table.select()).fetchall()
if not cursos_inseridos:
    raise Exception("Falha ao inserir cursos")

# Disciplinas
disciplinas = {
    'Ciência da Computação': [
        ('Introdução à Computação', 1), ('Matemática para Computação', 1), ('Sistemas Digitais', 1),
        ('Fundamentos de Programação', 1), ('Fundamentos de Projeto 1', 1), ('Projeto 1', 1),
        ('Programação Imperativa e Funcional', 2), ('Algoritmos e Estrutura de Dados', 2),
        ('Lógica para Computação', 2), ('Fundamentos de Desenvolvimento de Software', 2),
        ('Fundamentos de Projetos 2', 2), ('Projeto 2', 2), ('Orientação a Objetos', 3),
        ('Infraestrutura de Hardware (AOC)', 3), ('Requisitos e Validação', 3),
        ('Infraestrutura de Software (SO)', 3), ('Fundamentos de Projetos 3', 3),
        ('Projeto 3', 3), ('Modelagem e Projeto de BD', 4), ('Infraestrutura de Comunicação (RSD)', 4),
        ('Estatística e Probabilidade', 4), ('Teoria da Computação', 4),
        ('Fundamentos de Projetos 4', 4), ('Projeto 4', 4), ('Teoria dos Grafos', 5),
        ('Análise e Visualização de Dados', 5), ('Interação Humano Computador', 5),
        ('Fundamentos de Computação Concorrente, Paralela e Distribuída', 5),
        ('Fundamentos de Projetos 5', 5), ('Projeto 5', 5), ('Aprendizado de Máquina', 6),
        ('Sistemas Embarcados', 6), ('Segurança da Informação', 6), ('Análise de Algoritmo', 6),
        ('Fundamentos de Projetos 6', 6), ('Projeto 6', 6), ('Eletiva 1', 7), ('Eletiva 2', 7),
        ('Tópicos Contemporâneos em Computação 1', 7), ('Tópicos Contemporâneos em Computação 2', 7),
        ('Fundamentos de Projetos 7', 7), ('Projeto 7', 7), ('Eletiva 3', 8),
        ('Eletiva 4', 8), ('Tópicos Contemporâneos em Computação 3', 8),
        ('Tópicos Contemporâneos em Computação 4', 8), ('Fundamentos de Projetos 8', 8),
        ('Projeto de Trabalho de Conclusão de Curso', 8)
    ],
    'Design': [
        ('Filosofia Prática', 1), ('Fundamentos da Pesquisa Social', 1), ('Sistemas Digitais', 1),
        ('Design da Experiência', 1), ('Fundamentos de Projeto 1', 1), ('Projeto 1', 1),
        ('Estudos de Futuro', 2), ('Análise e Comunicação da Pesquisa', 2),
        ('Semiótica', 2), ('Representação Visual', 2), ('Fundamentos de Projetos 2', 2),
        ('Projeto 2', 2), ('História e Teoria do Design I', 3), ('Projeto Gráfico', 3),
        ('Técnicas de Ideação', 3), ('Design Especulativo', 3), ('Fundamentos de Projetos 3', 3),
        ('Projeto 3', 3), ('Representação Tridimensional', 4), ('Design de Interfaces I', 4),
        ('Prototipação e Testes com Usuários I', 4), ('História e Teoria do Design II', 4),
        ('Fundamentos de Projetos 4', 4), ('Projeto 4', 4), ('Prototipação e Testes com Usuários II', 5),
        ('Design de Interfaces II', 5), ('Design de Produtos', 5), ('Design de Serviços', 5),
        ('Fundamentos de Projetos 5', 5), ('Projeto 5', 5), ('Fabricação Digital', 6),
        ('Design de Interfaces Ubíquas', 6), ('Teoria e Futuro do Design', 6),
        ('Design e Meio Ambiente', 6), ('Fundamentos de Projetos 6', 6), ('Projeto 6', 6),
        ('Eletiva 1', 7), ('Eletiva 2', 7), ('Tópicos Contemporâneos em Design 1', 7),
        ('Tópicos Contemporâneos em Design 2', 7), ('Fundamentos de Projetos 7', 7), ('Projeto 7', 7),
        ('Eletiva 3', 8), ('Eletiva 4', 8), ('Tópicos Contemporâneos em Design 3', 8),
        ('Tópicos Contemporâneos em Design 4', 8), ('Fundamentos de Projetos 8', 8),
        ('Projeto de Trabalho de Conclusão de Curso', 8)
    ],
    'Gestão de TI': [
        ('Introdução à Computação', 1), ('Matemática para Computação', 1), ('Design da Experiência', 1),
        ('Fundamentos de Programação', 1), ('Projeto 1', 1), ('Orientação a Objetos', 2),
        ('Técnicas de Ideação', 2), ('Gestão de Pessoas', 2), ('Fundamentos de Desenvolvimento de Software', 2),
        ('Projeto 2', 2), ('Estudos de Futuro', 3), ('Modelagem e Projeto de BD', 3),
        ('Requisitos e Validação', 3), ('Gestão de Projetos', 3), ('Projeto 3', 3),
        ('Negócios na Internet', 4), ('Interação Humano Máquina', 4), ('Estatística e Probabilidade', 4),
        ('Empreendedorismo', 4), ('Projeto 4', 4), ('Eletiva I', 5), ('Governança em TI', 5),
        ('Tópicos em Direito (LGPD)', 5), ('Análise e Visualização de Dados', 5), ('Projeto 5', 5)
    ],
    'Análise e Desenvolvimento de Sistemas': [
        ('Introdução à Computação', 1), ('Matemática para Computação', 1), ('Sistemas Digitais', 1),
        ('Fundamentos de Programação', 1), ('Projeto 1', 1), ('FP1: Gestão de Pessoas', 1),
        ('Programação Imperativa', 2), ('Interfaces Humano Computador', 2), ('Lógica para Computação', 2),
        ('Fundamentos de Desenvolvimento de Software', 2), ('Projeto 2', 2), ('FP2: Gestão de Projetos', 2),
        ('Orientação a Objetos', 3), ('Algoritmos e Estrutura de Dados', 3),
        ('Infraestrutura de Software (SO)', 3), ('Infraestrutura de Software (RSD)', 3),
        ('Estatística e Probabilidade', 3), ('Projeto 3', 3), ('Modelagem e Projeto de BD', 4),
        ('Requisitos, Projeto de Software e Validação', 4),
        ('Fundamentos de Computação Concorrente, Paralela e Distribuída', 4),
        ('Desenvolvimento Web', 4), ('Disciplina Optativa: Libras e Outras', 4), ('Projeto 4', 4),
        ('Segurança da Informação', 5), ('Desenvolvimento Mobile', 5), ('Eletiva 2', 5),
        ('Eletiva 3', 5), ('Negócios na Internet', 5), ('Projeto 5', 5)
    ],
    'Sistemas de Informação': [
        ('Introdução à Computação', 1), ('Matemática para Computação', 1), ('Sistemas Digitais', 1),
        ('Fundamentos de Programação', 1), ('Projeto 1', 1), ('FP1: Gestão de Pessoas', 1),
        ('Programação Imperativa', 2), ('Interfaces Humano Computador', 2), ('Lógica para Computação', 2),
        ('Fundamentos de Desenvolvimento de Software', 2), ('Projetos 2', 2), ('FP2: Gestão de Projetos', 2),
        ('Orientação a Objetos', 3), ('Infraestrutura de Hardware (AOC)', 3),
        ('Administração de Empresas', 3), ('Algoritmos e Estrutura de Dados', 3),
        ('Projeto 3', 3), ('FP3: Metodologia Científica', 3), ('Modelagem e Projeto de BD', 4),
        ('Infraestrutura de Comunicação (RSD)', 4), ('Estatística e Probabilidade', 4),
        ('Infraestrutura de Software (SO)', 4), ('Projeto 4', 4), ('FP4: Empreendedorismo', 4),
        ('Requisitos, Projeto de Software e Validação', 5), ('Análise e Visualização de Dados', 5),
        ('Pensamento Empresarial', 5), ('Fundamentos de Computação Concorrente, Paralela e Distribuída', 5),
        ('Projeto 5', 5), ('FP5: Marketing', 5), ('Aprendizado de Máquina', 6),
        ('Arquitetura de Software Nativa em Nuvem', 6), ('Segurança da Informação', 6),
        ('Desenvolvimento Web', 6), ('Projeto 6', 6), ('FP6: Tópicos em Direito', 6),
        ('Eletiva 1', 7), ('Eletiva 2', 7), ('DevOps', 7), ('Desenvolvimento Mobile', 7),
        ('Projeto 7', 7), ('Pré-Projeto de TCC', 7), ('Eletiva 3', 8), ('Eletiva 4', 8),
        ('Tópicos Contemporâneos em Sistema de Informação', 8), ('Governança de TI', 8),
        ('FP8: Acompanhamento de TCC', 8)
    ]
}

for curso, disc_list in disciplinas.items():
    # Buscar o ID do curso pelo nome do curso
    curso_id_result = session.execute(cursos_table.select().where(cursos_table.c.NomeCurso == curso)).fetchone()
    if curso_id_result is None:
        raise Exception(f"Curso '{curso}' não encontrado")
    curso_id = curso_id_result.CursoID
    
    for nome_disciplina, periodo in disc_list:
        professor_id = random.choice(professores)
        
        # Inserir disciplina na tabela
        session.execute(disciplinas_table.insert().values(
            NomeDisciplina=nome_disciplina,
            CursoID=curso_id,
            Periodo=periodo
        ))
session.commit()

# Alunos
alunos = []
for _ in range(10):
    nome_completo = fake.name()
    data_nascimento = fake.date_of_birth()
    data_matricula = fake.date_this_decade()
    aluno_id = session.execute(alunos_table.insert().values(Nome=nome_completo, DataNascimento=data_nascimento, DataMatricula=data_matricula)).inserted_primary_key[0]
    alunos.append(aluno_id)

# Matrículas
disciplinas = session.execute(disciplinas_table.select()).fetchall()
for aluno in alunos:
    num_disciplinas = random.randint(1, 6)
    disciplinas_escolhidas = random.sample(disciplinas, num_disciplinas)
    for disciplina in disciplinas_escolhidas:
        data_matricula = fake.date_this_year()
        session.execute(matriculas_table.insert().values(AlunoID=aluno, DisciplinaID=disciplina.DisciplinaID, DataMatricula=data_matricula))

# Histórico Escolar
for aluno in alunos:
    num_disciplinas = random.randint(1, 6)
    disciplinas_escolhidas = random.sample(disciplinas, num_disciplinas)
    for disciplina in disciplinas_escolhidas:
        data_conclusao = fake.date_this_year()
        nota = round(random.uniform(0, 10), 2)
        session.execute(historico_escolar_table.insert().values(AlunoID=aluno, DisciplinaID=disciplina.DisciplinaID, DataConclusao=data_conclusao, Nota=nota))

session.commit()

print("Dados fictícios gerados com sucesso!")
