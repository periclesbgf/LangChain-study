# # import
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.core import StorageContext
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from IPython.display import Markdown, display
# import chromadb
# from utils import OPENAI_API_KEY
# # set up OpenAI
# import os
# import openai

# openai.api_key = OPENAI_API_KEY

# import requests


# def get_wikipedia_images(title):
#     response = requests.get(
#         "https://en.wikipedia.org/w/api.php",
#         params={
#             "action": "query",
#             "format": "json",
#             "titles": title,
#             "prop": "imageinfo",
#             "iiprop": "url|dimensions|mime",
#             "generator": "images",
#             "gimlimit": "50",
#         },
#     ).json()
#     image_urls = []
#     for page in response["query"]["pages"].values():
#         if page["imageinfo"][0]["url"].endswith(".jpg") or page["imageinfo"][
#             0
#         ]["url"].endswith(".png"):
#             image_urls.append(page["imageinfo"][0]["url"])
#     return image_urls

# from pathlib import Path
# import urllib.request

# image_uuid = 0
# MAX_IMAGES_PER_WIKI = 20

# wiki_titles = {
#     "Tesla Model X",
#     "Pablo Picasso",
#     "Rivian",
#     "The Lord of the Rings",
#     "The Matrix",
#     "The Simpsons",
# }

# data_path = Path("mixed_wiki")
# if not data_path.exists():
#     Path.mkdir(data_path)

# for title in wiki_titles:
#     response = requests.get(
#         "https://en.wikipedia.org/w/api.php",
#         params={
#             "action": "query",
#             "format": "json",
#             "titles": title,
#             "prop": "extracts",
#             "explaintext": True,
#         },
#     ).json()
#     page = next(iter(response["query"]["pages"].values()))
#     wiki_text = page["extract"]

#     with open(data_path / f"{title}.txt", "w") as fp:
#         fp.write(wiki_text)

#     images_per_wiki = 0
#     try:
#         # page_py = wikipedia.page(title)
#         list_img_urls = get_wikipedia_images(title)
#         # print(list_img_urls)

#         for url in list_img_urls:
#             if url.endswith(".jpg") or url.endswith(".png"):
#                 image_uuid += 1
#                 # image_file_name = title + "_" + url.split("/")[-1]

#                 urllib.request.urlretrieve(
#                     url, data_path / f"{image_uuid}.jpg"
#                 )
#                 images_per_wiki += 1
#                 # Limit the number of images downloaded per wiki page to 15
#                 if images_per_wiki > MAX_IMAGES_PER_WIKI:
#                     break
#     except:
#         print(str(Exception("No images found for Wikipedia page: ")) + title)
#         continue