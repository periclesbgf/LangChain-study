from langchain.tools import tool
from youtubesearchpython import VideosSearch
import wikipediaapi
from serpapi import GoogleSearch
from datetime import datetime, timezone
from database.mongo_database_manager import MongoDatabaseManager
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any


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

class DatabaseUpdateTool:
    def __init__(self, db_manager: 'MongoDatabaseManager'):
        self.db_manager = db_manager

    ### Métodos para Plano de Estudo ###
    async def update_study_plan(self, id_sessao: str, plan_data: dict) -> bool:
        try:
            print(f"[DEBUG]: Preparing study plan update data: {plan_data}")
            
            update_data = {
                "plano_execucao": plan_data.get("plano_execucao", []),
                "duracao_total": plan_data.get("duracao_total", ""),
                "progresso_total": plan_data.get("progresso_total", 0),
                "updated_at": datetime.now(timezone.utc)
            }
            
            success = await self.db_manager.update_study_plan(id_sessao, update_data)
            print(f"[DEBUG]: Study plan update status: {success}")
            return success
            
        except Exception as e:
            print(f"[ERROR]: Error updating study plan: {e}")
            return False

    ### Métodos para Memória do Usuário ###
    async def create_user_memory(self, user_email: str, initial_data: Dict) -> bool:
        """Cria uma nova entrada de memória para o usuário"""
        try:
            print(f"[DEBUG]: Creating memory for user {user_email}")
            memory_data = {
                "Email": user_email,
                "memoria_curto_prazo": initial_data.get("memoria_curto_prazo", []),
                "memoria_longo_prazo": initial_data.get("memoria_longo_prazo", []),
                "ultima_interacao": datetime.now(timezone.utc),
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            collection = self.db_manager.db['student_profiles']
            result = await collection.update_one(
                {"Email": user_email},
                {"$set": {"memoria": memory_data}},
                upsert=True
            )
            
            success = result.modified_count > 0 or result.upserted_id is not None
            print(f"[DEBUG]: Memory creation status: {success}")
            return success
            
        except Exception as e:
            print(f"[ERROR]: Error creating user memory: {e}")
            return False

    async def update_user_memory(self, user_email: str, memory_data: Dict) -> bool:
        """Atualiza a memória do usuário"""
        try:
            print(f"[DEBUG]: Updating memory for user {user_email}")
            update_data = {
                "memoria.memoria_curto_prazo": memory_data.get("memoria_curto_prazo", []),
                "memoria.memoria_longo_prazo": memory_data.get("memoria_longo_prazo", []),
                "memoria.ultima_interacao": datetime.now(timezone.utc),
                "memoria.updated_at": datetime.now(timezone.utc)
            }
            
            collection = self.db_manager.db['student_profiles']
            result = await collection.update_one(
                {"Email": user_email},
                {"$set": update_data}
            )
            
            success = result.modified_count > 0
            print(f"[DEBUG]: Memory update status: {success}")
            return success
            
        except Exception as e:
            print(f"[ERROR]: Error updating user memory: {e}")
            return False

    async def get_user_memory(self, user_email: str) -> Optional[Dict]:
        """Recupera a memória do usuário"""
        try:
            print(f"[DEBUG]: Retrieving memory for user {user_email}")
            collection = self.db_manager.db['student_profiles']
            result = await collection.find_one(
                {"Email": user_email},
                {"memoria": 1}
            )
            
            memory = result.get("memoria") if result else None
            print(f"[DEBUG]: Retrieved memory data: {memory}")
            return memory
            
        except Exception as e:
            print(f"[ERROR]: Error retrieving user memory: {e}")
            return None

    async def delete_user_memory(self, user_email: str) -> bool:
        """Deleta a memória do usuário"""
        try:
            print(f"[DEBUG]: Deleting memory for user {user_email}")
            collection = self.db_manager.db['student_profiles']
            result = await collection.update_one(
                {"Email": user_email},
                {"$unset": {"memoria": ""}}
            )
            
            success = result.modified_count > 0
            print(f"[DEBUG]: Memory deletion status: {success}")
            return success
            
        except Exception as e:
            print(f"[ERROR]: Error deleting user memory: {e}")
            return False

    ### Métodos para Análises ###
    async def create_session_analysis(self, analysis_data: Dict) -> Optional[str]:
        """Cria uma nova análise de sessão"""
        try:
            print(f"[DEBUG]: Creating analysis for session {analysis_data['session_id']}")
            analysis_doc = {
                "session_id": analysis_data["session_id"],
                "user_email": analysis_data["user_email"],
                "timestamp": datetime.now(timezone.utc),
                "analise_comportamental": analysis_data.get("comportamental", {}),
                "analise_aprendizado": analysis_data.get("aprendizado", {}),
                "analise_engajamento": analysis_data.get("engajamento", {}),
                "recomendacoes": analysis_data.get("recomendacoes", []),
                "metricas": analysis_data.get("metricas", {}),
                "created_at": datetime.now(timezone.utc)
            }
            
            collection = self.db_manager.db['session_analysis']
            result = await collection.insert_one(analysis_doc)
            
            inserted_id = str(result.inserted_id)
            print(f"[DEBUG]: Analysis created with ID: {inserted_id}")
            return inserted_id
            
        except Exception as e:
            print(f"[ERROR]: Error creating session analysis: {e}")
            return None

    async def update_session_analysis(self, session_id: str, analysis_data: Dict) -> bool:
        """Atualiza a análise de uma sessão"""
        try:
            print(f"[DEBUG]: Updating analysis for session {session_id}")
            update_data = {
                "analise_comportamental": analysis_data.get("comportamental", {}),
                "analise_aprendizado": analysis_data.get("aprendizado", {}),
                "analise_engajamento": analysis_data.get("engajamento", {}),
                "recomendacoes": analysis_data.get("recomendacoes", []),
                "metricas": analysis_data.get("metricas", {}),
                "updated_at": datetime.now(timezone.utc)
            }
            
            collection = self.db_manager.db['session_analysis']
            result = await collection.update_one(
                {"session_id": session_id},
                {"$set": update_data}
            )
            
            success = result.modified_count > 0
            print(f"[DEBUG]: Analysis update status: {success}")
            return success
            
        except Exception as e:
            print(f"[ERROR]: Error updating session analysis: {e}")
            return False

    async def get_session_analysis(self, session_id: str) -> Optional[Dict]:
        """Recupera a análise de uma sessão"""
        try:
            print(f"[DEBUG]: Retrieving analysis for session {session_id}")
            collection = self.db_manager.db['session_analysis']
            result = await collection.find_one({"session_id": session_id})
            
            print(f"[DEBUG]: Retrieved analysis data: {result}")
            return result
            
        except Exception as e:
            print(f"[ERROR]: Error retrieving session analysis: {e}")
            return None

    async def get_user_analysis_history(self, user_email: str, limit: int = 10) -> List[Dict]:
        """Recupera o histórico de análises do usuário"""
        try:
            print(f"[DEBUG]: Retrieving analysis history for user {user_email}")
            collection = self.db_manager.db['session_analysis']
            cursor = collection.find(
                {"user_email": user_email}
            ).sort("timestamp", -1).limit(limit)
            
            history = await cursor.to_list(length=limit)
            print(f"[DEBUG]: Retrieved {len(history)} analysis records")
            return history
            
        except Exception as e:
            print(f"[ERROR]: Error retrieving user analysis history: {e}")
            return []

    async def delete_session_analysis(self, session_id: str) -> bool:
        """Deleta a análise de uma sessão"""
        try:
            print(f"[DEBUG]: Deleting analysis for session {session_id}")
            collection = self.db_manager.db['session_analysis']
            result = await collection.delete_one({"session_id": session_id})
            
            success = result.deleted_count > 0
            print(f"[DEBUG]: Analysis deletion status: {success}")
            return success
            
        except Exception as e:
            print(f"[ERROR]: Error deleting session analysis: {e}")
            return False

    async def aggregate_user_analytics(self, user_email: str) -> Dict[str, Any]:
        """Agrega análises do usuário para insights de longo prazo"""
        try:
            print(f"[DEBUG]: Aggregating analytics for user {user_email}")
            collection = self.db_manager.db['session_analysis']
            pipeline = [
                {"$match": {"user_email": user_email}},
                {"$sort": {"timestamp": -1}},
                {"$limit": 50},
                {"$group": {
                    "_id": "$user_email",
                    "media_engajamento": {"$avg": "$analise_engajamento.nivel"},
                    "total_sessoes": {"$sum": 1},
                    "areas_interesse": {"$addToSet": "$analise_comportamental.interesses"},
                    "pontos_fortes": {"$addToSet": "$analise_aprendizado.pontos_fortes"},
                    "areas_melhorar": {"$addToSet": "$analise_aprendizado.areas_melhorar"},
                    "ultima_analise": {"$first": "$$ROOT"}
                }}
            ]
            
            result = await collection.aggregate(pipeline).to_list(length=1)
            analytics = result[0] if result else {}
            print(f"[DEBUG]: Aggregated analytics: {analytics}")
            return analytics
            
        except Exception as e:
            print(f"[ERROR]: Error aggregating user analytics: {e}")
            return {}