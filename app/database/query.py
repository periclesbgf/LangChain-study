import psycopg2

def execute_query(query):
    try:
        connection = psycopg2.connect(
            host='localhost',
            port=5432,
            database='postgres',
            user='postgres',
            password='123456789'
        )
        cursor = connection.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        connection.close()
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

list_tables()

def prepare_query(query):
    return query.replace('```sql', '').replace('```', '').strip()

query = """
SELECT c."Cliente", COUNT(*) AS Quantidade FROM "Telemetria" t JOIN "Chassis" c ON t."Chassi" = c."Chassi" WHERE t."Serie" = 'E' GROUP BY c."Cliente" ORDER BY Quantidade DESC LIMIT 1;
"""
column_names = execute_query(query)
print("Colunas na tabela Telemetria:")
print(column_names)