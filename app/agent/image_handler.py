# app/agent/image_handler.py

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
import openai

class ImageHandler:
    def __init__(self, api_key):
        # Initialize the OpenAI API key
        self.api_key = api_key

    def encode_image(self, image_path):
        """Converts an image file to a base64-encoded string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def encode_image_bytes(self, image_bytes):
        """Converts image bytes to a base64-encoded string."""
        return base64.b64encode(image_bytes).decode("utf-8")

    async def image_summarize(self, img_base64):
        """
        Generates a description of the image using the OpenAI API.

        :param img_base64: Base64-encoded image string.
        :param prompt: Prompt to guide the image description.
        :return: Description of the image.
        """
        prompt_text = """Você é um assistente encarregado de resumir tabelas para recuperação (retrieval). \
        Dê um resumo conciso da tabela que seja bem otimizado para recuperação. Certifique-se de capturar todos os detalhes. \
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image",
                        "image": {
                            "base64": img_base64,
                        },
                    },
                ],
            }
        ]

        try:
            # Make the API call to OpenAI
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300,
            )

            # Extract the description from the response
            description = response['choices'][0]['message']['content'].strip()
            return description

        except Exception as e:
            print(f"Error during image summarization: {e}")
            return "Não foi possível gerar uma descrição da imagem."

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
