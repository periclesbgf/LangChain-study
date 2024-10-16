import base64
import io
from pymongo import MongoClient
from PIL import Image
import matplotlib.pyplot as plt
from app.utils import MONGO_URI, MONGO_DB_NAME

# Conectar ao MongoDB
client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
image_collection = db["image_collection"]  # Substitua pelo nome correto da sua coleção

def recuperar_e_exibir_imagem(uuid):
    """
    Recupera a imagem codificada em base64 ou bytes do MongoDB e salva em um arquivo.
    
    :param uuid: O UUID da imagem no MongoDB.
    """
    # Consultar a imagem no MongoDB
    resultado = image_collection.find_one({"_id": uuid})
    
    if resultado:
        print("Imagem encontrada, processando...")

        # Recuperar os dados binários da imagem diretamente
        imagem_data = resultado.get("image_data")
        
        if isinstance(imagem_data, bytes):
            print("Dados da imagem recuperados como bytes.")
            
            # Converter os dados da imagem em um objeto PIL
            img = Image.open(io.BytesIO(imagem_data))
            
            # Salvar a imagem em um arquivo
            img.save("imagem_recuperada.png")
            print("Imagem salva como 'imagem_recuperada.png'.")
        else:
            print("O campo 'image_data' não contém dados binários.")
    else:
        print("Imagem não encontrada no MongoDB.")

# Exemplo de uso
uuid = "d64766ed-5c77-4614-8106-8f9312311960"  # Substitua pelo UUID da imagem que deseja recuperar
recuperar_e_exibir_imagem(uuid)
