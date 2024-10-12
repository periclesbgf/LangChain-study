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
        return [table[0] for table in results]
    else:
        print("Não foi possível recuperar a lista de tabelas.")
        return []

def get_all_data_from_tables():
    tables = list_tables()
    if not tables:
        return
    
    for table in tables:
        print(f"Dados da tabela: {table}")
        query = f'SELECT * FROM "{table}";'
        results = execute_query(query)
        
        if results:
            for row in results:
                print(row)
        else:
            print(f"Não foi possível recuperar os dados da tabela {table}.")

# Executa o código para listar e pegar todos os dados das tabelas
get_all_data_from_tables()
