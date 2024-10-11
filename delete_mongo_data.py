from pymongo import MongoClient

def clear_all_collections():
    try:
        # Conecte-se ao MongoDB
        client = MongoClient('mongodb://admin:123456789@localhost:27017/?authSource=admin')
        
        # Seleciona o banco de dados
        db = client['mongo_student']  # Use o nome do banco de dados aqui

        # Pega todas as coleções no banco de dados
        collections = db.list_collection_names()

        if not collections:
            print("Nenhuma coleção encontrada.")
            return

        # Itera sobre cada coleção e apaga todos os documentos
        for collection_name in collections:
            print(f"Limpando a coleção: {collection_name}")
            collection = db[collection_name]
            result = collection.delete_many({})  # Apaga todos os documentos da coleção
            print(f"Documentos apagados da coleção '{collection_name}': {result.deleted_count}")

    except Exception as e:
        print(f"Ocorreu um erro ao acessar o MongoDB: {e}")

# Executa o código para limpar todas as coleções
clear_all_collections()
