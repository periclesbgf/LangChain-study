
import asyncio
import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

from database.vector_db import QdrantHandler
from agent_system.states.common_states import AgentState


class RetrievalTools:
    def __init__(self, qdrant_handler: QdrantHandler, student_email: str, disciplina: str, image_collection, session_id: str, state: AgentState):
        self.qdrant_handler = qdrant_handler
        self.student_email = student_email
        self.disciplina = disciplina
        self.session_id = session_id
        self.image_collection = image_collection
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.agent_state = state

        self.RELEVANCE_ANALYSIS_PROMPT = """
        Analise a relevância dos contextos recuperados para a pergunta do usuário.

        Pergunta: {question}

        Contextos recuperados:
        Texto: {text_context}
        Imagem: {image_context}
        Tabela: {table_context}

        Para cada contexto, avalie a relevância em uma escala de 0 a 1 e explique brevemente por quê.
        Retorne um JSON no formato:
        {{
            "text": {{"score": 0.0, "reason": "string"}},
            "image": {{"score": 0.0, "reason": "string"}},
            "table": {{"score": 0.0, "reason": "string"}},
            "recommended_context": "text|image|table|combined"
        }}

        Mantenha o formato JSON exato e use apenas aspas duplas.
        """

    @tool
    async def parallel_context_retrieval(self, question: str) -> Dict[str, Any]:
        """"
        Recupera conteudo educacional do aluno, como, Slides, livros, tabelas.
        Args: Query do Agente educacional buscando conteudo educacional.
        Return: Um dicionário com os contextos de texto, imagem e tabela, e a análise de relevância contendo a pontuação e a razão.
        """
        return {"context": "there are no files in database"}
        text_context, image_context, table_context = await asyncio.gather(
            self._retrieve_text_context(question),
            self._retrieve_image_context(question),
            self._retrieve_table_context(question)
        )

        relevance_analysis = await self.analyze_context_relevance(
            original_question=question,
            text_context=text_context,
            image_context=image_context,
            table_context=table_context
        )

        return {
            "text": text_context,
            "image": image_context,
            "table": table_context,
            "relevance_analysis": relevance_analysis
        }

    async def _retrieve_text_context(self, query: str) -> str:
        try:
            results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=self.student_email,
                session_id=self.session_id,
                disciplina_id=self.disciplina,
                specific_metadata={"type": "text"}
            )
            return "\n".join([doc.page_content for doc in results]) if results else ""
        except Exception as e:
            #print(f"[RETRIEVAL] Error in text retrieval: {e}")
            return ""

    async def _retrieve_image_context(self, query: str) -> Dict[str, Any]:
        try:
            results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=self.student_email,
                session_id=self.session_id,
                disciplina_id=self.disciplina,
                specific_metadata={"type": "image"}
            )
            #print("")
            #print("--------------------------------------------------")
            #print(f"[RETRIEVAL] Image search results: {results}")
            #print("--------------------------------------------------")
            #print("")
            if not results:
                return {"type": "image", "content": None, "description": ""}

            image_uuid = results[0].metadata.get("image_uuid")
            if not image_uuid:
                return {"type": "image", "content": None, "description": ""}

            return await self.retrieve_image_and_description(image_uuid)
        except Exception as e:
            #print(f"[RETRIEVAL] Error in image retrieval: {e}")
            return {"type": "image", "content": None, "description": ""}

    async def _retrieve_table_context(self, query: str) -> Dict[str, Any]:
        try:
            results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=self.student_email,
                session_id=self.session_id,
                disciplina_id=self.disciplina,
                specific_metadata={"type": "table"}
            )

            if not results:
                return {"type": "table", "content": None}

            return {
                "type": "table",
                "content": results[0].page_content,
                "metadata": results[0].metadata
            }
        except Exception as e:
            #print(f"[RETRIEVAL] Error in table retrieval: {e}")
            return {"type": "table", "content": None}

    async def retrieve_image_and_description(self, image_uuid: str) -> Dict[str, Any]:
        """
        Recupera a imagem e sua descrição de forma assíncrona.
        """
        try:
            #print(f"[RETRIEVAL] Recuperando imagem com UUID: {image_uuid}")
            image_data = await self.image_collection.find_one({"_id": image_uuid})
            if not image_data:
                #print(f"[RETRIEVAL] Imagem não encontrada: {image_uuid}")
                return {"type": "error", "message": "Imagem não encontrada"}

            image_bytes = image_data.get("image_data")
            if not image_bytes:
                #print("[RETRIEVAL] Dados da imagem ausentes")
                return {"type": "error", "message": "Dados da imagem ausentes"}

            if isinstance(image_bytes, bytes):
                processed_bytes = image_bytes
            elif isinstance(image_bytes, str):
                processed_bytes = image_bytes.encode('utf-8')
            else:
                #print(f"[RETRIEVAL] Formato de imagem não suportado: {type(image_bytes)}")
                return {"type": "error", "message": "Formato de imagem não suportado"}

            results = self.qdrant_handler.similarity_search_with_filter(
                query="",
                student_email=self.student_email,
                session_id=self.session_id,
                disciplina_id=self.disciplina,
                k=1,
                use_global=False,
                use_discipline=False,
                use_session=True,
                specific_metadata={"image_uuid": image_uuid, "type": "image"}
            )
            #print(f"[RETRIEVAL] Resultados da busca de descrição: {results}")
            if not results:
                return {"type": "error", "message": "Descrição da imagem não encontrada"}
            #print("[RETRIEVAL] Imagem e descrição recuperadas com sucesso")
            #print(f"[RETRIEVAL] Descrição da imagem: {results[0].page_content}")
            return {
                "type": "image",
                "image_bytes": processed_bytes,
                "description": results[0].page_content
            }
        except Exception as e:
            #print(f"[RETRIEVAL] Erro ao recuperar imagem: {e}")
            import traceback
            traceback.print_exc()
            return {"type": "error", "message": str(e)}

    async def analyze_context_relevance(
        self,
        original_question: str,
        text_context: str,
        image_context: Dict[str, Any],
        table_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            if not any([text_context, image_context, table_context]):
                #print("[RETRIEVAL] No context available for relevance analysis")
                return self._get_default_analysis()

            prompt = ChatPromptTemplate.from_template(self.RELEVANCE_ANALYSIS_PROMPT)

            image_description = ""
            if image_context and isinstance(image_context, dict):
                image_description = image_context.get("description", "")

            # Tratamento seguro para contexto de tabela
            table_content = ""
            if table_context and isinstance(table_context, dict):
                table_content = table_context.get("content", "")

            # Garantir que todos os contextos são strings antes de aplicar slice
            text_preview = str(text_context)[:500] + "..." if text_context and len(str(text_context)) > 500 else str(text_context or "")
            image_preview = str(image_description)[:500] + "..." if len(str(image_description)) > 500 else str(image_description)
            table_preview = str(table_content)[:500] + "..." if len(str(table_content)) > 500 else str(table_content)

            response = await self.model.ainvoke(prompt.format(
                question=original_question,
                text_context=text_preview,
                image_context=image_preview,
                table_context=table_preview
            ))

            try:
                cleaned_content = response.content.strip()
                if cleaned_content.startswith("```json"):
                    cleaned_content = cleaned_content[7:]
                if cleaned_content.endswith("```"):
                    cleaned_content = cleaned_content[:-3]
                cleaned_content = cleaned_content.strip()

                analysis = json.loads(cleaned_content)
                #print(f"[RETRIEVAL] Relevance analysis: {analysis}")

                required_fields = ["text", "image", "table", "recommended_context"]
                if not all(field in analysis for field in required_fields):
                    raise ValueError("Missing required fields in analysis")

                return analysis

            except json.JSONDecodeError as e:
                #print(f"[RETRIEVAL] Error parsing relevance analysis: {e}")
                #print(f"[RETRIEVAL] Invalid JSON content: {cleaned_content}")
                return self._get_default_analysis()
            except ValueError as e:
                #print(f"[RETRIEVAL] Validation error: {e}")
                return self._get_default_analysis()

        except Exception as e:
            #print(f"[RETRIEVAL] Error in analyze_context_relevance: {e}")
            return self._get_default_analysis()

    def _get_default_analysis(self) -> Dict[str, Any]:
        return {
            "text": {"score": 0, "reason": "Default fallback due to analysis error"},
            "image": {"score": 0, "reason": "Default fallback due to analysis error"},
            "table": {"score": 0, "reason": "Default fallback due to analysis error"},
            "recommended_context": "combined"
        }


# class HistoryTools:
#     def __init__(self, qdrant_handler: QdrantHandler, student_email: str, disciplina: str, session_id: str):
#         self.qdrant_handler = qdrant_handler
#         self.student_email = student_email
#         self.disciplina = disciplina
#         self.session_id = session_id

#     @tool
#     async def retrieve_coversation_history(self, number_of_messages: int = 5) -> Dict[str, Any]:
#         """
#         Recupera o histórico de conversas do aluno.
#         """
#         try:
#             results = self.qdrant_handler.similarity_search_with_filter(
#                 query="",
#                 student_email=self.student_email,
#                 session_id=self.session_id,
#                 disciplina_id=self.disciplina,
#                 k=number_of_messages,
#                 use_global=False,
#                 use_discipline=False,
#                 use_session=True,
#                 specific_metadata={"type": "message"}
#             )

#             if not results:
#                 return {"messages": []}

#             messages = [{"content": doc.page_content, "timestamp": doc.created_at} for doc in results]
#             return {"messages": messages}
#         except Exception as e:
#             #print(f"[HISTORY] Error in retrieve_coversation_history: {e}")
#             return {"messages": []}

# class TavilySearchTools:
#     def __init__(self, api_key: str):
#         self.tavily_search = TavilySearchResults(
#             max_results=5,
#             include_answer=True,
#             include_raw_content=True,
#             include_images=False,
#             api_key=api_key
#         )

#     @tool
#     async def search_tavily(self, question: str) -> Dict[str, Any]:
#         """
#         Realiza uma busca no Tavily para recuperar resultados relevantes.
#         """
#         try:
#             results = await self.tavily_search.run(query=question)