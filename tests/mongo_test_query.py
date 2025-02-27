from pymongo import MongoClient
from bson.json_util import dumps

# Função para conectar ao MongoDB e pegar todos os dados
def get_all_collections_data():
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

        # Itera sobre cada coleção e busca todos os documentos
        for collection_name in collections:
            print(f"Dados da coleção: {collection_name}")
            collection = db[collection_name]

            # Obtém todos os documentos da coleção
            documents = collection.find()
            
            # Converte os documentos para uma lista e imprime
            for doc in documents:
                print(dumps(doc, indent=4))  # Formato JSON para visualização amigável

    except Exception as e:
        print(f"Ocorreu um erro ao acessar o MongoDB: {e}")
        return None

# Executa o código para listar e pegar todos os dados das coleções
get_all_collections_data()
