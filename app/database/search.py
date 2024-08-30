import psycopg2

def execute_query(query):
    try:
        with psycopg2.connect(
            host='localhost',
            port=5432,
            database='postgres',
            user='postgres',
            password='123456789'
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                return results
    except Exception as e:
        print(f"Ocorreu um erro ao executar a query: {e}")
        return None

def list_tables():
    query = "SELECT tablename FROM pg_tables WHERE schemaname='public';"
    results = execute_query(query)
    if results:
        print("Tabelas no banco de dados:")
        for table in results:
            print(table[0])
    else:
        print("Não foi possível recuperar a lista de tabelas.")

def prepare_query(query):
    return query.replace('```sql', '').replace('```', '').strip()

list_tables()

query = """
SELECT * FROM "Aluno";
"""

results = execute_query(query)

if results:
    print("Dados na tabela:")
    for row in results:
        print(row)
else:
    print("Não foi possível recuperar os dados da tabela.")


# Quais disciplinas  tem no 5 periodo de design?
# query = """
# SELECT d."NomeDisciplina"
# FROM "Disciplinas" d
# JOIN "Cursos" c ON d."CursoID" = c."CursoID"
# WHERE c."NomeCurso" = 'Design' AND d."Periodo" = 5;
# """

# Em quais disciplinas fui para a final
# query = """
# SELECT d."NomeDisciplina"
# FROM "HistoricoEscolar" h
# JOIN "Disciplinas" d ON h."DisciplinaID" = d."DisciplinaID"
# WHERE h."Final" IS NOT NULL;
# """