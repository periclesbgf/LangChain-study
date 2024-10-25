from pymongo import MongoClient
from bson.json_util import dumps

# Função para conectar ao MongoDB e pegar o nome de todas as coleções
def list_collections():
    try:
        # Conecta ao MongoDB
        client = MongoClient('mongodb://admin:123456789@localhost:27017/?authSource=admin')

        # Seleciona o banco de dados
        db = client['mongo_student']  # Use o nome do banco de dados aqui

        # Pega todas as coleções do banco
        collections = db.list_collection_names()

        if not collections:
            print("Nenhuma coleção encontrada.")
            return None

        print("Coleções disponíveis:")
        for i, collection_name in enumerate(collections):
            print(f"{i + 1}. {collection_name}")

        return collections

    except Exception as e:
        print(f"Ocorreu um erro ao acessar o MongoDB: {e}")
        return None

# Função para exibir os documentos de uma coleção específica
def show_collection_data(db, collection_name):
    try:
        collection = db[collection_name]

        # Obtém todos os documentos da coleção
        documents = collection.find()

        print(f"\nConteúdo da coleção: {collection_name}")
        for doc in documents:
            print(dumps(doc, indent=4))  # Formato JSON para melhor visualização

    except Exception as e:
        print(f"Ocorreu um erro ao acessar a coleção '{collection_name}': {e}")

# Função principal para executar o programa
def main():
    client = MongoClient('mongodb://admin:123456789@localhost:27017/?authSource=admin')
    db = client['mongo_student']

    collections = list_collections()
    if not collections:
        return

    try:
        # Recebe a escolha do usuário para a coleção
        choice = int(input("\nEscolha o número da coleção que deseja ver: ")) - 1

        if 0 <= choice < len(collections):
            show_collection_data(db, collections[choice])
        else:
            print("Escolha inválida. Tente novamente.")
    except ValueError:
        print("Por favor, insira um número válido.")

# Executa o programa
if __name__ == "__main__":
    main()
