from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Date, Numeric, ForeignKey
from sqlalchemy.orm import sessionmaker
from faker import Faker
import random
import os
from datetime import datetime
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

# Definir todas as tabelas (assumindo que as tabelas já estão criadas no banco de dados)
professores_table = Table('Professores', metadata, autoload_with=engine)
coordenadores_table = Table('Coordenadores', metadata, autoload_with=engine)
cursos_table = Table('Cursos', metadata, autoload_with=engine)
disciplinas_table = Table('Disciplinas', metadata, autoload_with=engine)
alunos_table = Table('Alunos', metadata, autoload_with=engine)
matriculas_table = Table('Matriculas', metadata, autoload_with=engine)
historico_escolar_table = Table('HistoricoEscolar', metadata, autoload_with=engine)

# Criar uma sessão
Session = sessionmaker(bind=engine)
session = Session()

# Instanciar Faker
fake = Faker('pt_BR')

# Função para gerar uma data de matrícula entre 2019 e 2023, em janeiro ou agosto
def generate_matricula_date():
    year = random.randint(2019, 2023)
    month = random.choice([1, 8])
    day = random.choice([15])  # Escolhendo dia 15 para simplicidade
    return datetime(year, month, day)

# Inserir 500 alunos
alunos = []
for _ in range(10000):
    nome_completo = fake.name()
    data_nascimento = fake.date_of_birth(minimum_age=17, maximum_age=30)
    data_matricula = generate_matricula_date()
    aluno_id = session.execute(alunos_table.insert().values(Nome=nome_completo, DataNascimento=data_nascimento, DataMatricula=data_matricula)).inserted_primary_key[0]
    alunos.append((aluno_id, data_matricula))

# Obter todas as disciplinas
disciplinas = session.execute(disciplinas_table.select()).fetchall()

# Organizar disciplinas por curso e período
disciplinas_por_curso = {}
for disciplina in disciplinas:
    if disciplina.CursoID not in disciplinas_por_curso:
        disciplinas_por_curso[disciplina.CursoID] = {}
    if disciplina.Periodo not in disciplinas_por_curso[disciplina.CursoID]:
        disciplinas_por_curso[disciplina.CursoID][disciplina.Periodo] = []
    disciplinas_por_curso[disciplina.CursoID][disciplina.Periodo].append(disciplina)

# Gerar histórico e matrículas para cada aluno
for aluno_id, data_matricula in alunos:
    curso_id = random.choice(list(disciplinas_por_curso.keys()))
    ano_matricula = data_matricula.year
    mes_matricula = data_matricula.month

    # Calcular o período atual baseado no ano e mês da matrícula
    periodo_atual = ((2023 - ano_matricula) * 2) - (0 if mes_matricula == 1 else 1)
    periodo_atual = min(periodo_atual, 8)  # Limitar o período atual a no máximo 8

    for periodo in range(1, periodo_atual + 1):
        if periodo in disciplinas_por_curso[curso_id]:
            disciplinas_no_periodo = disciplinas_por_curso[curso_id][periodo]
            num_disciplinas = random.randint(1, len(disciplinas_no_periodo))
            disciplinas_escolhidas = random.sample(disciplinas_no_periodo, num_disciplinas)

            for disciplina in disciplinas_escolhidas:
                data_conclusao = fake.date_between_dates(date_start=data_matricula)
                nota = round(random.uniform(0, 10), 2)
                session.execute(historico_escolar_table.insert().values(
                    AlunoID=aluno_id,
                    DisciplinaID=disciplina.DisciplinaID,
                    DataConclusao=data_conclusao,
                    Nota=nota
                ))

                # Inserir matrícula
                session.execute(matriculas_table.insert().values(
                    AlunoID=aluno_id,
                    DisciplinaID=disciplina.DisciplinaID,
                    DataMatricula=data_matricula
                ))

session.commit()

print("Dados fictícios de 500 alunos, matrículas e histórico escolar gerados com sucesso!")
