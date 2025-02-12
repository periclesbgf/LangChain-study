# app/agent/image_handler.py

import httpx
import base64
import os
import re
from PIL import Image
import io
from tqdm import tqdm
from IPython.display import HTML, display
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from openai import OpenAI
from pydantic import BaseModel, Field
from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain import globals
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser
from openai import OpenAI
import base64
import requests
from PIL import Image
import io


class ImageHandler:
    def __init__(self, api_key):
        # Inicializa a chave da API do OpenAI
        self.api_key = api_key

    def encode_image(self, image_path):
        """Converte um arquivo de imagem em uma string codificada em base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def encode_image_bytes(self, image_bytes):
        """Converte bytes de imagem em uma string codificada em base64."""
        return base64.b64encode(image_bytes).decode("utf-8")

    def resize_image_for_gpt(self, img: Image):
        """
        Redimensiona a imagem para garantir que o menor lado tenha 768px e
        o tamanho máximo seja 2048px.
        """
        max_size = 2048
        min_side = 768

        # Obter dimensões originais
        width, height = img.size
        print(f"Dimensões originais: {width}x{height}")

        # Redimensionar se necessário
        if max(width, height) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)
            print(f"Redimensionado para máximo de {max_size}px.")
        
        # Garantir que o lado menor tenha no mínimo 768px
        if min(width, height) < min_side:
            scaling_factor = min_side / min(width, height)
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            print(f"Redimensionado para garantir mínimo de {min_side}px no lado menor.")
        
        print(f"Dimensões finais: {img.size}")
        return img

    def resize_base64_image(self, base64_string, size=(768, 768)):
        """Redimensiona uma imagem codificada em base64."""
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        
        # Redimensionar a imagem para o tamanho especificado
        resized_img = self.resize_image_for_gpt(img)
        
        # Salvar a imagem redimensionada para bytes
        buffered = io.BytesIO()
        resized_img.save(buffered, format=img.format)
        
        # Retornar a imagem codificada em base64
        resized_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        print(f"Tamanho da imagem codificada em base64: {len(resized_base64)} caracteres")
        return resized_base64

    def image_summarize(self, img_base64):
        """
        Gera uma descrição da imagem usando o modelo GPT-4.

        :param img_base64: String da imagem codificada em base64.
        :return: Descrição da imagem.
        """
        print("Iniciando o processo de sumarização da imagem...")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Definir o payload da requisição
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Descreva a imagem com detalhes para facilitar a recuperação futura em um sitema RAG. Essa descricao ira ser guardada em um banco de dados Vetorial"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]
                }
            ],
            "max_tokens": 300
        }

        print("Payload preparado. Enviando solicitação para a API OpenAI...")

        # Realizar a chamada da API usando a biblioteca requests
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        # Exibir o retorno da API
        print(f"Status da resposta: {response.status_code}")
        print(f"Resposta completa: {response.json()}")

        # Processar a resposta e extrair a descrição
        data = response.json()
        description = data['choices'][0]['message']['content'].strip()
        print(f"Descrição da imagem gerada: {description}")
        return description

    def looks_like_base64(self, sb):
        """Checks if the string appears to be base64-encoded."""
        return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

    def is_image_data(self, b64data):
        """Verifies if the base64 data corresponds to an image."""
        image_signatures = {
            b"\xff\xd8\xff": "jpg",
            b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
            b"\x47\x49\x46\x38": "gif",
            b"\x52\x49\x46\x46": "webp",
        }
        try:
            header = base64.b64decode(b64data)[:8]  # Decode and get the first bytes
            for sig in image_signatures.keys():
                if header.startswith(sig):
                    return True
            return False
        except Exception:
            return False

    def resize_base64_image(self, base64_string, size=(64, 64)):
        """Resizes a base64-encoded image."""
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        # Resize the image
        resized_img = img.resize(size, Image.LANCZOS)
        # Save the resized image to a bytes buffer
        buffered = io.BytesIO()
        resized_img.save(buffered, format=img.format)
        # Encode the resized image to base64
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def split_image_text_types(self, docs):
        """Separates base64-encoded images and texts into distinct lists."""
        b64_images = []
        texts = []
        for doc in docs:
            # Check if the document is of type Document and extract the content
            if isinstance(doc, Document):
                doc = doc.page_content
            if self.looks_like_base64(doc) and self.is_image_data(doc):
                doc = self.resize_base64_image(doc, size=(64, 64))
                b64_images.append(doc)
            else:
                texts.append(doc)
        return {"images": b64_images, "texts": texts}
