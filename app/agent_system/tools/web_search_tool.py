from youtubesearchpython import VideosSearch
import wikipediaapi
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool


tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    # include_domains=[...],
    # exclude_domains=[...],
    # name="...",            # overwrite default tool name
    # description="...",     # overwrite default tool description
    # args_schema=...,       # overwrite default args_schema: BaseModel
)


class WebSearchTools:
    def __init__(self, web_search_engine: TavilySearchResults):
        self.web_search_engine = web_search_engine

    def search_web(self, query: str) -> str:
        """
        Realiza uma pesquisa na web e retorna o resultado mais relevante.
        """
        try:
            #print(f"[WEBSEARCH] Searching web for: {query}")
            search_results = self.web_search_engine.search(query)

            if search_results:
                #print(f"[WEBSEARCH] Found {len(search_results)} results")
                return search_results[0]
            else:
                #print("[WEBSEARCH] No results found")
                return "Nenhum resultado encontrado."
        except Exception as e:
            #print(f"[WEBSEARCH] Error searching web: {str(e)}")
            return "Ocorreu um erro ao buscar na web."

    @tool
    def search_youtube(self, query: str) -> str:
        """
        Realiza uma pesquisa no YouTube e retorna o link do vídeo mais relevante.
        """
        try:
            #print(f"[WEBSEARCH] Searching YouTube for: {query}")
            videos_search = VideosSearch(query, limit=3)  # Aumentamos para 3 resultados
            results = videos_search.result()

            if results['result']:
                videos_info = []
                for video in results['result'][:3]:  # Pegamos os 3 primeiros resultados
                    video_info = {
                        'title': video['title'],
                        'link': video['link'],
                        'channel': video.get('channel', {}).get('name', 'N/A'),
                        'duration': video.get('duration', 'N/A'),
                        'description': video.get('descriptionSnippet', [{'text': 'Sem descrição'}])[0]['text']
                    }
                    videos_info.append(video_info)

                # Formata a resposta com múltiplos vídeos
                response = "Vídeos encontrados:\n\n"
                for i, video in enumerate(videos_info, 1):
                    response += (
                        f"{i}. Título: {video['title']}\n"
                        f"   Link: {video['link']}\n"
                        f"   Canal: {video['channel']}\n"
                        f"   Duração: {video['duration']}\n"
                        f"   Descrição: {video['description']}\n\n"
                    )

                #print(f"[WEBSEARCH] Found {len(videos_info)} videos")
                return response
            else:
                #print("[WEBSEARCH] No videos found")
                return "Nenhum vídeo encontrado."
        except Exception as e:
            #print(f"[WEBSEARCH] Error searching YouTube: {str(e)}")
            return "Ocorreu um erro ao buscar no YouTube."

    def search_wikipedia(self, query: str) -> str:
        """
        Realiza uma pesquisa no Wikipedia e retorna o resumo da página.
        """
        try:
            #print(f"[WEBSEARCH] Searching Wikipedia for: {query}")
            wiki_wiki = wikipediaapi.Wikipedia(
                'TutorBot/1.0 (pericles.junior@cesar.school)',
                'pt'
            )
            page = wiki_wiki.page(query)

            if page.exists():
                summary = (
                    f"Título: {page.title}\n"
                    f"Resumo: {page.summary[:500]}...\n"
                    f"Link: {page.fullurl}"
                )
                #print(f"[WEBSEARCH] Found Wikipedia article: {page.title}")
                return summary
            else:
                #print("[WEBSEARCH] No Wikipedia article found")
                return "Página não encontrada na Wikipedia."
        except Exception as e:
            #print(f"[WEBSEARCH] Error searching Wikipedia: {str(e)}")
            return "Ocorreu um erro ao buscar na Wikipedia."

def route_after_planning(state: AgentState) -> str:
    """
    Determina o próximo nó após o planejamento com base no plano gerado.
    """
    next_step = state.get("next_step", "retrieval")
    #print(f"[ROUTING] Routing after planning: {next_step}")

    if next_step == "websearch":
        return "web_search"
    elif next_step == "retrieval":
        return "retrieve_context"
    else:
        return "direct_answer"

def create_websearch_node(web_tools: WebSearchTools):
    QUERY_OPTIMIZATION_PROMPT = """Você é um especialista em otimizar buscas no YouTube.

    Pergunta original do aluno: {question}
    Histórico da conversa: {chat_history}

    Seu objetivo é criar uma query otimizada para o YouTube que:
    1. Identifique os conceitos principais da pergunta
    2. Adicione termos relevantes para melhorar os resultados (como "tutorial", "explicação", "aula")
    3. Inclua termos técnicos apropriados
    4. Use uma linguagem natural e efetiva para buscas
    5. Mantenha o foco educacional

    Retorne apenas a query otimizada, sem explicações adicionais.
    """

    WEBSEARCH_PROMPT = """
    Você é um assistente especializado em integrar informações da web com o contexto educacional.

    A pergunta original do aluno foi: {question}
    A query otimizada usada foi: {optimized_query}

    Encontrei os seguintes recursos:

    {resources}

    Crie uma resposta educativa que:
    1. Apresente os recursos encontrados de forma organizada
    2. Destaque por que eles são relevantes para a pergunta
    3. Sugira como o aluno pode aproveitar melhor o conteúdo
    4. Inclua os links diretos para facilitar o acesso

    Mantenha os links originais na resposta.
    """

    query_prompt = ChatPromptTemplate.from_template(QUERY_OPTIMIZATION_PROMPT)
    response_prompt = ChatPromptTemplate.from_template(WEBSEARCH_PROMPT)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    async def web_search(state: AgentState) -> AgentState:
        #print("\n[NODE:WEBSEARCH] Starting web search")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        chat_history = format_chat_history(state["chat_history"])

        try:
            # Otimizar a query
            #print(f"[WEBSEARCH] Optimizing query: {latest_question}")
            optimized_query = model.invoke(query_prompt.format(
                question=latest_question,
                chat_history=chat_history
            )).content.strip()
            #print(f"[WEBSEARCH] Optimized query: {optimized_query}")

            # Realizar buscas com a query otimizada
            wiki_result = web_tools.search_wikipedia(optimized_query)
            youtube_result = web_tools.search_youtube(optimized_query)

            # Formatar recursos para o prompt
            resources = (
                "YouTube:\n"
                f"{youtube_result}\n\n"
                "Wikipedia:\n"
                f"{wiki_result}"
            )

            # Gerar resposta
            response = model.invoke(response_prompt.format(
                question=latest_question,
                optimized_query=optimized_query,
                resources=resources
            ))

            # Estruturar contexto para o nó de teaching
            extracted_context = {
                "text": resources,
                "image": {"type": "image", "content": None, "description": ""},
                "table": {"type": "table", "content": None},
                "relevance_analysis": {
                    "text": {"score": 1.0, "reason": "Informação obtida da web"},
                    "image": {"score": 0.0, "reason": "Nenhuma imagem disponível"},
                    "table": {"score": 0.0, "reason": "Nenhuma tabela disponível"},
                    "recommended_context": "text"
                }
            }

            # Atualizar estado
            new_state = state.copy()
            new_state["web_search_results"] = {
                "wikipedia": wiki_result,
                "youtube": youtube_result,
                "optimized_query": optimized_query  # Guardamos a query otimizada para referência
            }
            new_state["extracted_context"] = extracted_context
            new_state["messages"] = list(state["messages"]) + [response]

            #print("[WEBSEARCH] Search completed successfully")
            return new_state

        except Exception as e:
            #print(f"[WEBSEARCH] Error during web search: {str(e)}")
            error_message = "Desculpe, encontrei um erro ao buscar os recursos. Por favor, tente novamente."
            
            new_state = state.copy()
            new_state["messages"] = list(state["messages"]) + [
                AIMessage(content=error_message)
            ]
            return new_state

    return web_search