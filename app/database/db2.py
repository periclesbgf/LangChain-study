from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Date, Numeric, ForeignKey
from sqlalchemy.orm import sessionmaker
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

# Tabela de Coordenadores
coordenadores_table = Table('Coordenadores', metadata,
    Column('CoordenadorID', Integer, primary_key=True, autoincrement=True, unique=True),
    Column('Nome', String(100), nullable=False)
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
    Column('Nota', Numeric(4, 2))
)

metadata.create_all(engine)

# Criar uma sessão
Session = sessionmaker(bind=engine)
session = Session()

# Lista de professores
professores = [
    'Carlos Eduardo Silva', 'Ana Paula Oliveira', 'José Ricardo Souza', 'Maria Fernanda Santos',
    'Luiz Felipe Almeida', 'Mariana Costa', 'Roberto Carlos Lima', 'Juliana Pereira',
    'Fernando Gomes', 'Larissa Rodrigues', 'Paulo Roberto Pinto', 'Tatiana Ferreira',
    'Rafael Alves', 'Aline Araújo', 'Marcelo Vieira', 'Cláudia Cardoso',
    'Lucas Martins', 'Patrícia Rocha', 'André Luiz Mendes', 'Beatriz Ramos',
    'Renato Nogueira', 'Camila Carvalho', 'Rodrigo Cunha', 'Gabriela Castro',
    'Vinícius Teixeira', 'Letícia Ribeiro', 'Fábio Lima', 'Natália Freitas',
    'Márcio Souza', 'Bruna Lopes', 'Gustavo Ferreira', 'Vanessa Silva',
    'Ricardo Pereira', 'Daniela Almeida', 'Eduardo Nascimento', 'Sabrina Torres',
    'João Pedro Fernandes', 'Bianca Martins', 'Felipe Santos', 'Michele Correia',
    'Adriano Oliveira', 'Lívia Araújo', 'Sérgio Gonçalves', 'Priscila Dias',
    'Alberto Costa', 'Viviane Lima', 'Victor Mendes', 'Flávia Barbosa',
    'Maurício Rodrigues', 'Débora Santana'
]

# Inserir Professores
professores_ids = {}
for nome in professores:
    professor_id = session.execute(professores_table.insert().values(Nome=nome)).inserted_primary_key[0]
    professores_ids[nome] = professor_id

# Nomes dos Coordenadores
coordenadores = [
    'João Carlos', 'Mariana Andrade', 'Pedro Henrique', 'Carla Silva', 'Rafael Gomes'
]

# Inserir Coordenadores
coordenadores_ids = {}
for nome in coordenadores:
    coordenador_id = session.execute(coordenadores_table.insert().values(Nome=nome)).inserted_primary_key[0]
    coordenadores_ids[nome] = coordenador_id

# Cursos
cursos = ['Ciência da Computação', 'Design', 'Gestão de TI', 'Análise e Desenvolvimento de Sistemas', 'Sistemas de Informação']
for curso, coordenador_nome in zip(cursos, coordenadores):
    coordenador_id = coordenadores_ids[coordenador_nome]
    session.execute(cursos_table.insert().values(NomeCurso=curso, CoordenadorID=coordenador_id))
session.commit()

# Verificar se os cursos foram inseridos corretamente
cursos_inseridos = session.execute(cursos_table.select()).fetchall()
if not cursos_inseridos:
    raise Exception("Falha ao inserir cursos")

# Disciplinas com professores atribuídos
professor_disciplinas = {
    'Ciência da Computação': [
        ('Introdução à Computação', 1, 'Carlos Eduardo Silva'), ('Matemática para Computação', 1, 'Ana Paula Oliveira'), ('Sistemas Digitais', 1, 'José Ricardo Souza'),
        ('Fundamentos de Programação', 1, 'Maria Fernanda Santos'), ('Fundamentos de Projeto 1', 1, 'Luiz Felipe Almeida'), ('Projeto 1', 1, 'Mariana Costa'),
        ('Programação Imperativa e Funcional', 2, 'Roberto Carlos Lima'), ('Algoritmos e Estrutura de Dados', 2, 'Juliana Pereira'),
        ('Lógica para Computação', 2, 'Fernando Gomes'), ('Fundamentos de Desenvolvimento de Software', 2, 'Larissa Rodrigues'),
        ('Fundamentos de Projetos 2', 2, 'Paulo Roberto Pinto'), ('Projeto 2', 2, 'Tatiana Ferreira'), ('Orientação a Objetos', 3, 'Rafael Alves'),
        ('Infraestrutura de Hardware (AOC)', 3, 'Aline Araújo'), ('Requisitos e Validação', 3, 'Marcelo Vieira'),
        ('Infraestrutura de Software (SO)', 3, 'Cláudia Cardoso'), ('Fundamentos de Projetos 3', 3, 'Lucas Martins'),
        ('Projeto 3', 3, 'Patrícia Rocha'), ('Modelagem e Projeto de BD', 4, 'André Luiz Mendes'), ('Infraestrutura de Comunicação (RSD)', 4, 'Beatriz Ramos'),
        ('Estatística e Probabilidade', 4, 'Renato Nogueira'), ('Teoria da Computação', 4, 'Camila Carvalho'),
        ('Fundamentos de Projetos 4', 4, 'Rodrigo Cunha'), ('Projeto 4', 4, 'Gabriela Castro'), ('Teoria dos Grafos', 5, 'Vinícius Teixeira'),
        ('Análise e Visualização de Dados', 5, 'Letícia Ribeiro'), ('Interação Humano Computador', 5, 'Fábio Lima'),
        ('Fundamentos de Computação Concorrente, Paralela e Distribuída', 5, 'Natália Freitas'),
        ('Fundamentos de Projetos 5', 5, 'Márcio Souza'), ('Projeto 5', 5, 'Bruna Lopes'), ('Aprendizado de Máquina', 6, 'Gustavo Ferreira'),
        ('Sistemas Embarcados', 6, 'Vanessa Silva'), ('Segurança da Informação', 6, 'Ricardo Pereira'), ('Análise de Algoritmo', 6, 'Daniela Almeida'),
        ('Fundamentos de Projetos 6', 6, 'Eduardo Nascimento'), ('Projeto 6', 6, 'Sabrina Torres'), ('Eletiva 1', 7, 'João Pedro Fernandes'), ('Eletiva 2', 7, 'Bianca Martins'),
        ('Tópicos Contemporâneos em Computação 1', 7, 'Felipe Santos'), ('Tópicos Contemporâneos em Computação 2', 7, 'Michele Correia'),
        ('Fundamentos de Projetos 7', 7, 'Adriano Oliveira'), ('Projeto 7', 7, 'Lívia Araújo'), ('Eletiva 3', 8, 'Sérgio Gonçalves'),
        ('Eletiva 4', 8, 'Priscila Dias'), ('Tópicos Contemporâneos em Computação 3', 8, 'Alberto Costa'),
        ('Tópicos Contemporâneos em Computação 4', 8, 'Viviane Lima'), ('Fundamentos de Projetos 8', 8, 'Victor Mendes'),
        ('Projeto de Trabalho de Conclusão de Curso', 8, 'Flávia Barbosa')
    ],
    'Design': [
        ('Filosofia Prática', 1, 'Maurício Rodrigues'), ('Fundamentos da Pesquisa Social', 1, 'Débora Santana'), ('Sistemas Digitais', 1, 'Carlos Eduardo Silva'),
        ('Design da Experiência', 1, 'Ana Paula Oliveira'), ('Fundamentos de Projeto 1', 1, 'José Ricardo Souza'), ('Projeto 1', 1, 'Maria Fernanda Santos'),
        ('Estudos de Futuro', 2, 'Luiz Felipe Almeida'), ('Análise e Comunicação da Pesquisa', 2, 'Mariana Costa'),
        ('Semiótica', 2, 'Roberto Carlos Lima'), ('Representação Visual', 2, 'Juliana Pereira'), ('Fundamentos de Projetos 2', 2, 'Fernando Gomes'),
        ('Projeto 2', 2, 'Larissa Rodrigues'), ('História e Teoria do Design I', 3, 'Paulo Roberto Pinto'), ('Projeto Gráfico', 3, 'Tatiana Ferreira'),
        ('Técnicas de Ideação', 3, 'Rafael Alves'), ('Design Especulativo', 3, 'Aline Araújo'), ('Fundamentos de Projetos 3', 3, 'Marcelo Vieira'),
        ('Projeto 3', 3, 'Cláudia Cardoso'), ('Representação Tridimensional', 4, 'Lucas Martins'), ('Design de Interfaces I', 4, 'Patrícia Rocha'),
        ('Prototipação e Testes com Usuários I', 4, 'André Luiz Mendes'), ('História e Teoria do Design II', 4, 'Beatriz Ramos'),
        ('Fundamentos de Projetos 4', 4, 'Renato Nogueira'), ('Projeto 4', 4, 'Camila Carvalho'), ('Prototipação e Testes com Usuários II', 5, 'Rodrigo Cunha'),
        ('Design de Interfaces II', 5, 'Gabriela Castro'), ('Design de Produtos', 5, 'Vinícius Teixeira'), ('Design de Serviços', 5, 'Letícia Ribeiro'),
        ('Fundamentos de Projetos 5', 5, 'Fábio Lima'), ('Projeto 5', 5, 'Natália Freitas'), ('Fabricação Digital', 6, 'Márcio Souza'),
        ('Design de Interfaces Ubíquas', 6, 'Bruna Lopes'), ('Teoria e Futuro do Design', 6, 'Gustavo Ferreira'),
        ('Design e Meio Ambiente', 6, 'Vanessa Silva'), ('Fundamentos de Projetos 6', 6, 'Ricardo Pereira'), ('Projeto 6', 6, 'Daniela Almeida'),
        ('Eletiva 1', 7, 'Eduardo Nascimento'), ('Eletiva 2', 7, 'Sabrina Torres'), ('Tópicos Contemporâneos em Design 1', 7, 'João Pedro Fernandes'),
        ('Tópicos Contemporâneos em Design 2', 7, 'Bianca Martins'), ('Fundamentos de Projetos 7', 7, 'Felipe Santos'), ('Projeto 7', 7, 'Michele Correia'),
        ('Eletiva 3', 8, 'Adriano Oliveira'), ('Eletiva 4', 8, 'Lívia Araújo'), ('Tópicos Contemporâneos em Design 3', 8, 'Sérgio Gonçalves'),
        ('Tópicos Contemporâneos em Design 4', 8, 'Priscila Dias'), ('Fundamentos de Projetos 8', 8, 'Alberto Costa'),
        ('Projeto de Trabalho de Conclusão de Curso', 8, 'Viviane Lima')
    ],
    'Gestão de TI': [
        ('Introdução à Computação', 1, 'Carlos Eduardo Silva'), ('Matemática para Computação', 1, 'Ana Paula Oliveira'), ('Design da Experiência', 1, 'José Ricardo Souza'),
        ('Fundamentos de Programação', 1, 'Maria Fernanda Santos'), ('Projeto 1', 1, 'Luiz Felipe Almeida'), ('Orientação a Objetos', 2, 'Mariana Costa'),
        ('Técnicas de Ideação', 2, 'Roberto Carlos Lima'), ('Gestão de Pessoas', 2, 'Juliana Pereira'), ('Fundamentos de Desenvolvimento de Software', 2, 'Fernando Gomes'),
        ('Projeto 2', 2, 'Larissa Rodrigues'), ('Estudos de Futuro', 3, 'Paulo Roberto Pinto'), ('Modelagem e Projeto de BD', 3, 'Tatiana Ferreira'),
        ('Requisitos e Validação', 3, 'Rafael Alves'), ('Gestão de Projetos', 3, 'Aline Araújo'), ('Projeto 3', 3, 'Marcelo Vieira'),
        ('Negócios na Internet', 4, 'Cláudia Cardoso'), ('Interação Humano Máquina', 4, 'Lucas Martins'), ('Estatística e Probabilidade', 4, 'Patrícia Rocha'),
        ('Empreendedorismo', 4, 'André Luiz Mendes'), ('Projeto 4', 4, 'Beatriz Ramos'), ('Eletiva I', 5, 'Renato Nogueira'), ('Governança em TI', 5, 'Camila Carvalho'),
        ('Tópicos em Direito (LGPD)', 5, 'Rodrigo Cunha'), ('Análise e Visualização de Dados', 5, 'Gabriela Castro'), ('Projeto 5', 5, 'Vinícius Teixeira')
    ],
    'Análise e Desenvolvimento de Sistemas': [
        ('Introdução à Computação', 1, 'Letícia Ribeiro'), ('Matemática para Computação', 1, 'Fábio Lima'), ('Sistemas Digitais', 1, 'Natália Freitas'),
        ('Fundamentos de Programação', 1, 'Márcio Souza'), ('Projeto 1', 1, 'Bruna Lopes'), ('FP1: Gestão de Pessoas', 1, 'Gustavo Ferreira'),
        ('Programação Imperativa', 2, 'Vanessa Silva'), ('Interfaces Humano Computador', 2, 'Ricardo Pereira'), ('Lógica para Computação', 2, 'Daniela Almeida'),
        ('Fundamentos de Desenvolvimento de Software', 2, 'Eduardo Nascimento'), ('Projeto 2', 2, 'Sabrina Torres'), ('FP2: Gestão de Projetos', 2, 'João Pedro Fernandes'),
        ('Orientação a Objetos', 3, 'Bianca Martins'), ('Algoritmos e Estrutura de Dados', 3, 'Felipe Santos'),
        ('Infraestrutura de Software (SO)', 3, 'Michele Correia'), ('Infraestrutura de Software (RSD)', 3, 'Adriano Oliveira'),
        ('Estatística e Probabilidade', 3, 'Lívia Araújo'), ('Projeto 3', 3, 'Sérgio Gonçalves'), ('Modelagem e Projeto de BD', 4, 'Priscila Dias'),
        ('Requisitos, Projeto de Software e Validação', 4, 'Alberto Costa'),
        ('Fundamentos de Computação Concorrente, Paralela e Distribuída', 4, 'Viviane Lima'),
        ('Desenvolvimento Web', 4, 'Victor Mendes'), ('Disciplina Optativa: Libras e Outras', 4, 'Flávia Barbosa'), ('Projeto 4', 4, 'Maurício Rodrigues'),
        ('Segurança da Informação', 5, 'Débora Santana'), ('Desenvolvimento Mobile', 5, 'Carlos Eduardo Silva'), ('Eletiva 2', 5, 'Ana Paula Oliveira'),
        ('Eletiva 3', 5, 'José Ricardo Souza'), ('Negócios na Internet', 5, 'Maria Fernanda Santos'), ('Projeto 5', 5, 'Luiz Felipe Almeida')
    ],
    'Sistemas de Informação': [
        ('Introdução à Computação', 1, 'Mariana Costa'), ('Matemática para Computação', 1, 'Roberto Carlos Lima'), ('Sistemas Digitais', 1, 'Juliana Pereira'),
        ('Fundamentos de Programação', 1, 'Fernando Gomes'), ('Projeto 1', 1, 'Larissa Rodrigues'), ('FP1: Gestão de Pessoas', 1, 'Paulo Roberto Pinto'),
        ('Programação Imperativa', 2, 'Tatiana Ferreira'), ('Interfaces Humano Computador', 2, 'Rafael Alves'), ('Lógica para Computação', 2, 'Aline Araújo'),
        ('Fundamentos de Desenvolvimento de Software', 2, 'Marcelo Vieira'), ('Projetos 2', 2, 'Cláudia Cardoso'), ('FP2: Gestão de Projetos', 2, 'Lucas Martins'),
        ('Orientação a Objetos', 3, 'Patrícia Rocha'), ('Infraestrutura de Hardware (AOC)', 3, 'André Luiz Mendes'),
        ('Administração de Empresas', 3, 'Beatriz Ramos'), ('Algoritmos e Estrutura de Dados', 3, 'Renato Nogueira'),
        ('Projeto 3', 3, 'Camila Carvalho'), ('FP3: Metodologia Científica', 3, 'Rodrigo Cunha'), ('Modelagem e Projeto de BD', 4, 'Gabriela Castro'),
        ('Infraestrutura de Comunicação (RSD)', 4, 'Vinícius Teixeira'), ('Estatística e Probabilidade', 4, 'Letícia Ribeiro'),
        ('Infraestrutura de Software (SO)', 4, 'Fábio Lima'), ('Projeto 4', 4, 'Natália Freitas'), ('FP4: Empreendedorismo', 4, 'Márcio Souza'),
        ('Requisitos, Projeto de Software e Validação', 5, 'Bruna Lopes'), ('Análise e Visualização de Dados', 5, 'Gustavo Ferreira'),
        ('Pensamento Empresarial', 5, 'Vanessa Silva'), ('Fundamentos de Computação Concorrente, Paralela e Distribuída', 5, 'Ricardo Pereira'),
        ('Projeto 5', 5, 'Daniela Almeida'), ('FP5: Marketing', 5, 'Eduardo Nascimento'), ('Aprendizado de Máquina', 6, 'Sabrina Torres'),
        ('Arquitetura de Software Nativa em Nuvem', 6, 'João Pedro Fernandes'), ('Segurança da Informação', 6, 'Bianca Martins'),
        ('Desenvolvimento Web', 6, 'Felipe Santos'), ('Projeto 6', 6, 'Michele Correia'), ('FP6: Tópicos em Direito', 6, 'Adriano Oliveira'),
        ('Eletiva 1', 7, 'Lívia Araújo'), ('Eletiva 2', 7, 'Sérgio Gonçalves'), ('DevOps', 7, 'Priscila Dias'), ('Desenvolvimento Mobile', 7, 'Alberto Costa'),
        ('Projeto 7', 7, 'Viviane Lima'), ('Pré-Projeto de TCC', 7, 'Victor Mendes'), ('Eletiva 3', 8, 'Flávia Barbosa'), ('Eletiva 4', 8, 'Maurício Rodrigues'),
        ('Tópicos Contemporâneos em Sistema de Informação', 8, 'Débora Santana'), ('Governança de TI', 8, 'Carlos Eduardo Silva'),
        ('FP8: Acompanhamento de TCC', 8, 'Ana Paula Oliveira')
    ]
}

for curso, disc_list in professor_disciplinas.items():
    curso_id = session.execute(cursos_table.select().where(cursos_table.c.NomeCurso == curso)).fetchone()
    if curso_id is None:
        raise Exception(f"Curso '{curso}' não encontrado")
    for nome_disciplina, periodo, professor_nome in disc_list:
        professor_id = professores_ids[professor_nome]
        session.execute(disciplinas_table.insert().values(NomeDisciplina=nome_disciplina, CursoID=curso_id.NomeCurso, ProfessorID=professor_id, Periodo=periodo))
session.commit()

# Verificar inserção de dados
print("Professores:")
for row in session.execute(professores_table.select()).fetchall():
    print(row)

print("\nCoordenadores:")
for row in session.execute(coordenadores_table.select()).fetchall():
    print(row)

print("\nCursos:")
for row in session.execute(cursos_table.select()).fetchall():
    print(row)

print("\nDisciplinas:")
for row in session.execute(disciplinas_table.select()).fetchall():
    print(row)

print("Dados fictícios gerados com sucesso!")
