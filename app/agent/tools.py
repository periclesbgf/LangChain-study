from langchain.tools import tool
from youtubesearchpython import VideosSearch
import wikipediaapi
from serpapi import GoogleSearch

# Ferramenta de busca no YouTube
@tool
def search_youtube(query: str) -> str:
    """
    Realiza uma pesquisa no YouTube e retorna o link do vídeo mais relevante.
    """
    videos_search = VideosSearch(query, limit=1)
    results = videos_search.result()
    
    if results['result']:
        video_info = results['result'][0]
        return f"Título: {video_info['title']}\nLink: {video_info['link']}\nDescrição: {video_info.get('descriptionSnippet', 'Sem descrição')}"
    else:
        return "Nenhum vídeo encontrado."

# Ferramenta de busca no Wikipedia
@tool
def search_wikipedia(query: str) -> str:
    """
    Realiza uma pesquisa no Wikipedia e retorna o resumo da página.
    """
    wiki_wiki = wikipediaapi.Wikipedia('pt')  # Português
    page = wiki_wiki.page(query)
    
    if page.exists():
        return f"Título: {page.title}\nResumo: {page.summary[:500]}...\nLink: {page.fullurl}"
    else:
        return "Página não encontrada."

# Ferramenta de busca no Google
@tool
def search_google(query: str) -> str:
    """
    Realiza uma pesquisa no Google e retorna o primeiro link relevante.
    """
    params = {
        "q": query,
        "hl": "pt",  # Português
        "gl": "br",  # Região Brasil
        "api_key": "YOUR_SERPAPI_KEY"  # Substitua com sua chave da API SerpAPI
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    if "organic_results" in results:
        top_result = results["organic_results"][0]
        return f"Título: {top_result['title']}\nLink: {top_result['link']}\nDescrição: {top_result.get('snippet', 'Sem descrição')}"
    else:
        return "Nenhum resultado encontrado no Google."

# # Testando a ferramenta de busca no YouTube
# print(search_youtube("Transformações lineares"))

# # Testando a ferramenta de busca no Wikipedia
# print(search_wikipedia("Transformações lineares"))

# # Testando a ferramenta de busca no Google
# print(search_google("O que são transformações lineares?"))
