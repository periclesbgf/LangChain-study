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

class ImageHandler:
    def __init__(self):
        pass

    def encode_image(self, image_path):
        """Getting the base64 string from an image"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def image_summarize(self, img_base64, prompt, vlm_client):
        """Make image summary using VLM model"""
        chat_response = vlm_client.chat.completions.create(
            model="llava-hf/llava-1.5-7b-hf",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}",
                        },
                    },
                ],
            }],
            stream=False
        )
        return chat_response.choices[0].message.content.strip()

    def generate_img_summaries(self, path, prompt):
        """
        Generate summaries and base64 encoded strings for images
        path: Path to list of .jpg files extracted
        """
        # Store base64 encoded images
        img_base64_list = []
        # Store image summaries
        image_summaries = []

        # Apply to images
        for img_file in tqdm(sorted(os.listdir(path))):
            if img_file.endswith(".jpg"):
                img_path = os.path.join(path, img_file)
                base64_image = self.encode_image(img_path)
                img_base64_list.append(base64_image)
                generated_summary = self.image_summarize(base64_image, prompt)
                print(generated_summary)
                image_summaries.append(generated_summary)

        return img_base64_list, image_summaries

    def plt_img_base64(self, img_base64):
        """Display base64 encoded string as image"""
        # Create an HTML img tag with the base64 string as the source
        image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
        # Display the image by rendering the HTML
        display(HTML(image_html))

    def looks_like_base64(self, sb):
        """Check if the string looks like base64"""
        return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

    def is_image_data(self, b64data):
        """
        Check if the base64 data is an image by looking at the start of the data
        """
        image_signatures = {
            b"\xff\xd8\xff": "jpg",
            b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
            b"\x47\x49\x46\x38": "gif",
            b"\x52\x49\x46\x46": "webp",
        }
        try:
            header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
            for sig, format in image_signatures.items():
                if header.startswith(sig):
                    return True
            return False
        except Exception:
            return False

    def resize_base64_image(self, base64_string, size=(64, 64)):
        """
        Resize an image encoded as a Base64 string
        """
        # Decode the Base64 string
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        # Resize the image
        resized_img = img.resize(size, Image.LANCZOS)
        # Save the resized image to a bytes buffer
        buffered = io.BytesIO()
        resized_img.save(buffered, format=img.format)
        # Encode the resized image to Base64
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def split_image_text_types(self, docs):
        """
        Split base64-encoded images and texts
        """
        b64_images = []
        texts = []
        for doc in docs:
            # Check if the document is of type Document and extract page_content if so
            if isinstance(doc, Document):
                doc = doc.page_content
            if self.looks_like_base64(doc) and self.is_image_data(doc):
                doc = self.resize_base64_image(doc, size=(64, 64))
                b64_images.append(doc)
            else:
                texts.append(doc)
        return {"images": b64_images, "texts": texts}

    def img_prompt_func(self, data_dict):
        """
        Join the context into a single string
        """
        formatted_texts = "\n".join(data_dict["context"]["texts"])
        messages = []
        # Adding the text for analysis
        text_message = {
            "type": "text",
            "text": (
                "You are an AI assistant with expertise in finance and business metrics.\n"
                "You will be given information that may include text, tables, and charts related to business performance and industry trends.\n"
                "Your task is to analyze this information and provide a clear, concise answer to the user's question.\n"
                "Focus on the most relevant data points and insights that directly address the user's query.\n"
                f"User's question: {data_dict['question']}\n\n"
                "Information provided:\n"
                f"{formatted_texts}"
            ),
        }
        messages.append(text_message)

        # Adding image(s) to the messages if present
        if data_dict["context"]["images"]:
            for image in data_dict["context"]["images"]:
                image_message = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
                messages.append(image_message)
        return [HumanMessage(content=messages)]

    def multi_modal_rag_context_chain(self, retriever):
        """Multi-modal RAG context chain"""
        chain = (
            {
                "context": retriever | RunnableLambda(self.split_image_text_types),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self.img_prompt_func)
        )
        return chain
