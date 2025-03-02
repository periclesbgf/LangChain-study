from typing import Any, TypedDict, List, Dict, Optional
from langgraph.graph import END, Graph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import json
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from database.vector_db import QdrantHandler
from dataclasses import dataclass
import base64
import asyncio
from typing import Dict, Any
from youtubesearchpython import VideosSearch
import wikipediaapi
from database.mongo_database_manager import MongoDatabaseManager
import time
from datetime import datetime, timezone

# Modelos de dados
class UserProfile(BaseModel):
    Nome: str
    Email: str
    EstiloAprendizagem: Dict[str, str]
    Feedback: Optional[Dict[str, Any]] = None
    PreferenciaAprendizado: Optional[Dict[str, Any]] = None

@dataclass
class ExecutionStep:
    titulo: str
    duracao: str
    descricao: str
    conteudo: List[str]
    recursos: List[Dict]
    atividade: Dict
    progresso: int

class AgentState(TypedDict):
    messages: List[BaseMessage]
    current_plan: str
    user_profile: Dict[str, Any]
    extracted_context: Dict[str, Any]
    next_step: str | None
    iteration_count: int
    chat_history: List[BaseMessage]
    needs_retrieval: bool
    evaluation_reason: str
    web_search_results: Dict[str, str]
    answer_type: str | None
    current_progress: Dict[str, Any]
    session_id: str
    thoughts: str

# Gerenciador de progresso (MongoDB)
class StudyProgressManager(MongoDatabaseManager):
    def __init__(self, db_name: str = "study_plans"):
        super().__init__()
        self.collection_name = db_name

    async def sync_progress_state(self, session_id: str) -> bool:
        try:
            collection = self.db[self.collection_name]
            plan = await collection.find_one({"id_sessao": session_id})
            if not plan:
                return False
            plano_execucao = plan.get("plano_execucao", [])
            modified = False
            for step in plano_execucao:
                original_progress = step.get("progresso", 0)
                corrected_progress = min(max(float(original_progress), 0), 100)
                if original_progress != corrected_progress:
                    step["progresso"] = corrected_progress
                    modified = True
            if modified:
                total_steps = len(plano_execucao)
                progresso_total = sum(step["progresso"] for step in plano_execucao) / total_steps
                await collection.update_one(
                    {"id_sessao": session_id},
                    {
                        "$set": {
                            "plano_execucao": plano_execucao,
                            "progresso_total": round(progresso_total, 2),
                            "updated_at": datetime.now(timezone.utc)
                        }
                    }
                )
            return True
        except Exception as e:
            return False

    async def get_study_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            collection = self.db[self.collection_name]
            plan = await collection.find_one(
                {"id_sessao": session_id},
                {"_id": 0, "plano_execucao": 1, "progresso_total": 1}
            )
            if not plan:
                return None
            if "plano_execucao" in plan:
                for step in plan["plano_execucao"]:
                    if "progresso" not in step:
                        step["progresso"] = 0
                    else:
                        step["progresso"] = min(max(float(step["progresso"]), 0), 100)
                total_steps = len(plan["plano_execucao"])
                if total_steps > 0:
                    progresso_total = sum(step["progresso"] for step in plan["plano_execucao"]) / total_steps
                    plan["progresso_total"] = round(progresso_total, 2)
            return {
                "plano_execucao": plan.get("plano_execucao", []),
                "progresso_total": plan.get("progresso_total", 0)
            }
        except Exception as e:
            return None

    async def update_step_progress(
        self,
        session_id: str,
        step_index: int,
        new_progress: int
    ) -> bool:
        try:
            if not 0 <= new_progress <= 100:
                raise ValueError("Progresso deve estar entre 0 e 100")
            collection = self.db[self.collection_name]
            plan = await collection.find_one({"id_sessao": session_id})
            if not plan:
                return False
            plano_execucao = plan.get("plano_execucao", [])
            if step_index >= len(plano_execucao):
                raise ValueError(f"Índice de etapa inválido: {step_index}")
            plano_execucao[step_index]["progresso"] = new_progress
            total_steps = len(plano_execucao)
            progresso_total = sum(step["progresso"] for step in plano_execucao) / total_steps
            result = await collection.update_one(
                {"id_sessao": session_id},
                {
                    "$set": {
                        "plano_execucao": plano_execucao,
                        "progresso_total": round(progresso_total, 2),
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            return result.modified_count > 0
        except Exception as e:
            return False

    async def get_step_details(
        self,
        session_id: str,
        step_index: int
    ) -> Optional[Dict[str, Any]]:
        try:
            collection = self.db[self.collection_name]
            plan = await collection.find_one({"id_sessao": session_id})
            if not plan or "plano_execucao" not in plan:
                return None
            plano_execucao = plan["plano_execucao"]
            if step_index >= len(plano_execucao):
                return None
            return plano_execucao[step_index]
        except Exception as e:
            return None

    async def mark_step_completed(
        self,
        session_id: str,
        step_index: int
    ) -> bool:
        return await self.update_step_progress(session_id, step_index, 100)

    async def get_next_incomplete_step(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        try:
            collection = self.db[self.collection_name]
            plan = await collection.find_one({"id_sessao": session_id})
            if not plan or "plano_execucao" not in plan:
                return None
            for index, step in enumerate(plan["plano_execucao"]):
                if step.get("progresso", 0) < 100:
                    return {
                        "index": index,
                        "step": step
                    }
            return None
        except Exception as e:
            return None

    async def get_study_summary(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        try:
            collection = self.db[self.collection_name]
            plan = await collection.find_one({"id_sessao": session_id})
            if not plan:
                return {
                    "error": "Plano não encontrado",
                    "session_id": session_id
                }
            plano_execucao = plan.get("plano_execucao", [])
            total_steps = len(plano_execucao)
            completed_steps = sum(1 for step in plano_execucao if step.get("progresso", 0) == 100)
            return {
                "session_id": session_id,
                "total_steps": total_steps,
                "completed_steps": completed_steps,
                "progress_percentage": plan.get("progresso_total", 0),
                "started_at": plan.get("created_at"),
                "last_updated": plan.get("updated_at"),
                "estimated_duration": plan.get("duracao_total", "60 minutos")
            }
        except Exception as e:
            return {
                "error": str(e),
                "session_id": session_id
            }

# Funções auxiliares para o histórico de chat
def filter_chat_history(messages: List[BaseMessage]) -> List[BaseMessage]:
    filtered_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            filtered_messages.append(msg)
        elif isinstance(msg, AIMessage):
            try:
                content = json.loads(msg.content)
                if isinstance(content, dict):
                    if content.get("type") == "multimodal":
                        filtered_content = {
                            "type": "multimodal",
                            "content": content["content"]
                        }
                        filtered_messages.append(AIMessage(content=filtered_content["content"]))
                    else:
                        filtered_messages.append(msg)
                else:
                    filtered_messages.append(msg)
            except json.JSONDecodeError:
                filtered_messages.append(msg)
    return filtered_messages

def format_chat_history(messages: List[BaseMessage], max_messages: int = 3) -> str:
    filtered_messages = filter_chat_history(messages[-max_messages:])
    formatted_history = []
    for msg in filtered_messages:
        role = 'Aluno' if isinstance(msg, HumanMessage) else 'Tutor'
        content = msg.content
        if isinstance(content, str):
            formatted_history.append(f"{role}: {content}")
    return "\n".join(formatted_history)

# Função auxiliar para extrair somente a resposta final (entre <resposta> e </resposta>)
def extract_final_answer(text: str) -> str:
    """Extrai o conteúdo entre <resposta> e </resposta>."""
    import re
    match = re.search(r"<resposta>(.*?)</resposta>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# Ferramentas de recuperação
class RetrievalTools:
    def __init__(self, qdrant_handler: QdrantHandler, student_email: str, disciplina: str, image_collection, session_id: str, state: AgentState):
        self.qdrant_handler = qdrant_handler
        self.student_email = student_email
        self.disciplina = disciplina
        self.session_id = session_id
        self.image_collection = image_collection
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.state = state
        self._query_cache = {}
        
        self.QUESTION_TRANSFORM_PROMPT = """
        Você é um especialista em transformar perguntas para melhorar a recuperação de contexto.
        Histórico da conversa: {chat_history}
        Pergunta original: {question}
        O usuário pode fazer perguntas que remetam a perguntas anteriores, então é importante analisar o histórico da conversa.
        Retorne apenas a pergunta reescrita, sem explicações adicionais.
        """

        self.RELEVANCE_ANALYSIS_PROMPT = """
        Analise a relevância dos contextos recuperados para a pergunta do usuário.
        Pergunta: {question}
        Contextos recuperados:
        Texto: {text_context}
        Imagem: {image_context}
        Tabela: {table_context}
        Para cada contexto, avalie a relevância em uma escala de 0 a 1 e explique brevemente por quê.
        Retorne um JSON no formato:
            "text": "score": 0.0, "reason": "string",
            "image": "score": 0.0, "reason": "string",
            "table": "score": 0.0, "reason": "string",
            "recommended_context": "text|image|table|combined"
        Mantenha o formato JSON exato e use apenas aspas duplas.
        """

    async def transform_question(self, question: str) -> str:
        cache_key = f"transform:{question.strip().lower()[:50]}"
        if cache_key in self._query_cache:
            print(f"[RETRIEVAL] Using cached question transformation")
            return self._query_cache[cache_key]
        formatted_history = format_chat_history(self.state["chat_history"], max_messages=4)
        prompt = ChatPromptTemplate.from_template(self.QUESTION_TRANSFORM_PROMPT)
        response = await self.model.ainvoke(prompt.format(
            chat_history=formatted_history,
            question=question,
        ))
        transformed_question = response.content.strip()
        self._query_cache[cache_key] = transformed_question
        return transformed_question

    async def parallel_context_retrieval(self, question: str) -> Dict[str, Any]:
        text_question, image_question, table_question = await asyncio.gather(
            self.transform_question(question),
            self.transform_question(question),
            self.transform_question(question)
        )
        text_context, image_context, table_context = await asyncio.gather(
            self.retrieve_text_context(text_question),
            self.retrieve_image_context(image_question),
            self.retrieve_table_context(table_question)
        )
        relevance_analysis = await self.analyze_context_relevance(
            original_question=question,
            text_context=text_context,
            image_context=image_context,
            table_context=table_context
        )
        print(relevance_analysis)
        return {
            "text": text_context,
            "image": image_context,
            "table": table_context,
            "relevance_analysis": relevance_analysis
        }

    async def retrieve_text_context(self, query: str) -> str:
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
            return ""

    async def retrieve_image_context(self, query: str) -> Dict[str, Any]:
        try:
            results = self.qdrant_handler.similarity_search_with_filter(
                query=query,
                student_email=self.student_email,
                session_id=self.session_id,
                disciplina_id=self.disciplina,
                specific_metadata={"type": "image"}
            )
            if not results:
                return {"type": "image", "content": None, "description": ""}
            image_uuid = results[0].metadata.get("image_uuid")
            if not image_uuid:
                return {"type": "image", "content": None, "description": ""}
            return await self.retrieve_image_and_description(image_uuid)
        except Exception as e:
            return {"type": "image", "content": None, "description": ""}

    async def retrieve_table_context(self, query: str) -> Dict[str, Any]:
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
            return {"type": "table", "content": None}

    async def retrieve_image_and_description(self, image_uuid: str) -> Dict[str, Any]:
        cache_key = f"image:{image_uuid}"
        if hasattr(self, '_query_cache') and cache_key in self._query_cache:
            print(f"[RETRIEVAL] Using cached image data for {image_uuid[:8]}...")
            return self._query_cache[cache_key]
        try:
            image_data = await self.image_collection.find_one({"_id": image_uuid})
            if not image_data:
                return {"type": "error", "message": "Imagem não encontrada"}
            image_bytes = image_data.get("image_data")
            if not image_bytes:
                return {"type": "error", "message": "Dados da imagem ausentes"}
            if isinstance(image_bytes, bytes):
                processed_bytes = image_bytes
            elif isinstance(image_bytes, str):
                processed_bytes = image_bytes.encode('utf-8')
            else:
                return {"type": "error", "message": "Formato de imagem não suportado"}
            if "description" in image_data and image_data["description"]:
                result = {
                    "type": "image",
                    "image_bytes": processed_bytes,
                    "description": image_data["description"]
                }
                if hasattr(self, '_query_cache'):
                    self._query_cache[cache_key] = result
                return result
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
            if not results:
                return {"type": "error", "message": "Descrição da imagem não encontrada"}
            result = {
                "type": "image",
                "image_bytes": processed_bytes,
                "description": results[0].page_content
            }
            if hasattr(self, '_query_cache'):
                self._query_cache[cache_key] = result
            return result
        except Exception as e:
            print(f"[RETRIEVAL] Error retrieving image {image_uuid}: {str(e)}")
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
                return self._get_default_analysis()
            cache_key = f"relevance:{original_question.strip().lower()[:50]}"
            if cache_key in self._query_cache:
                print(f"[RETRIEVAL] Using cached relevance analysis")
                return self._query_cache[cache_key]
            has_image = image_context and isinstance(image_context, dict) and image_context.get("description")
            has_table = table_context and isinstance(table_context, dict) and table_context.get("content")
            has_text = bool(text_context)
            if has_text and not has_image and not has_table:
                simple_analysis = {
                    "text": {"score": 0.9, "reason": "Único contexto disponível"},
                    "image": {"score": 0.0, "reason": "Nenhuma imagem disponível"},
                    "table": {"score": 0.0, "reason": "Nenhuma tabela disponível"},
                    "recommended_context": "text"
                }
                self._query_cache[cache_key] = simple_analysis
                return simple_analysis
            if has_image and not has_text and not has_table:
                simple_analysis = {
                    "text": {"score": 0.0, "reason": "Nenhum texto disponível"},
                    "image": {"score": 0.9, "reason": "Única imagem disponível"},
                    "table": {"score": 0.0, "reason": "Nenhuma tabela disponível"},
                    "recommended_context": "image"
                }
                self._query_cache[cache_key] = simple_analysis
                return simple_analysis
            if has_table and not has_text and not has_image:
                simple_analysis = {
                    "text": {"score": 0.0, "reason": "Nenhum texto disponível"},
                    "image": {"score": 0.0, "reason": "Nenhuma imagem disponível"},
                    "table": {"score": 0.9, "reason": "Única tabela disponível"},
                    "recommended_context": "table"
                }
                self._query_cache[cache_key] = simple_analysis
                return simple_analysis
            prompt = ChatPromptTemplate.from_template(self.RELEVANCE_ANALYSIS_PROMPT)
            image_description = image_context.get("description", "") if has_image else ""
            table_content = table_context.get("content", "") if has_table else ""
            text_preview = text_context[:300] + "..." if text_context and len(text_context) > 300 else text_context or ""
            image_preview = image_description[:300] + "..." if image_description and len(image_description) > 300 else image_description
            table_preview = str(table_content)[:300] + "..." if table_content and len(str(table_content)) > 300 else str(table_content or "")
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
                required_fields = ["text", "image", "table", "recommended_context"]
                if not all(field in analysis for field in required_fields):
                    raise ValueError("Missing required fields in analysis")
                self._query_cache[cache_key] = analysis
                return analysis
            except (json.JSONDecodeError, ValueError) as e:
                default_analysis = self._get_default_analysis()
                self._query_cache[cache_key] = default_analysis
                return default_analysis
        except Exception as e:
            return self._get_default_analysis()

    def _get_default_analysis(self) -> Dict[str, Any]:
        return {
            "text": {"score": 0.0, "reason": "Default text relevance score"},
            "image": {"score": 0.0, "reason": "Default image relevance score"},
            "table": {"score": 0.0, "reason": "Default table relevance score"},
            "recommended_context": "combined"
        }

# Ferramentas de busca na web
class WebSearchTools:
    def __init__(self):
        pass

    def search_youtube(self, query: str) -> str:
        try:
            videos_search = VideosSearch(query, limit=3)
            results = videos_search.result()
            if results['result']:
                videos_info = []
                for video in results['result'][:3]:
                    video_info = {
                        'title': video['title'],
                        'link': video['link'],
                        'channel': video.get('channel', {}).get('name', 'N/A'),
                        'duration': video.get('duration', 'N/A'),
                        'description': video.get('descriptionSnippet', [{'text': 'Sem descrição'}])[0]['text']
                    }
                    videos_info.append(video_info)
                response = "Vídeos encontrados:\n\n"
                for i, video in enumerate(videos_info, 1):
                    response += (
                        f"{i}. Título: {video['title']}\n"
                        f"   Link: {video['link']}\n"
                        f"   Canal: {video['channel']}\n"
                        f"   Duração: {video['duration']}\n"
                        f"   Descrição: {video['description']}\n\n"
                    )
                return response
            else:
                return "Nenhum vídeo encontrado."
        except Exception as e:
            return "Ocorreu um erro ao buscar no YouTube."

    def search_wikipedia(self, query: str) -> str:
        try:
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
                return summary
            else:
                return "Página não encontrada na Wikipedia."
        except Exception as e:
            return "Ocorreu um erro ao buscar na Wikipedia."

# Criação do prompt ReAct
def create_react_prompt():
    REACT_PROMPT = """# Assistente Educacional Proativo com Abordagem ReAct

Você é um tutor educacional proativo que usa a abordagem ReAct (Raciocinar + Agir) para ensinar ativamente os estudantes. Sua missão é liderar o processo de aprendizagem, não apenas respondendo passivamente, mas ensinando ativamente o conteúdo da etapa atual do plano de estudos e estimulando o pensamento crítico do aluno.

## PERFIL DO ALUNO:
Nome: {nome}
Estilo de Aprendizagem:
- Percepção: {percepcao} (Sensorial/Intuitivo)
- Entrada: {entrada} (Visual/Verbal)
- Processamento: {processamento} (Ativo/Reflexivo)
- Entendimento: {entendimento} (Sequencial/Global)

## ETAPA ATUAL DO PLANO:
Título: {titulo}
Descrição: {descricao}
Progresso: {progresso}%

## HISTÓRICO DA CONVERSAÇÃO:
{chat_history}

## PERGUNTA OU MENSAGEM DO ALUNO:
{question}

## DIRETRIZES PARA ENSINO PROATIVO:

1. <pensamento>Seu raciocínio pedagógico detalhado. Considere se imagens ou recursos visuais seriam úteis para este tópico e este aluno específico. Para alunos com estilo visual de aprendizagem, material visual é particularmente valioso.</pensamento>
2. <ação>Estratégia de ensino escolhida e detalhes. Se você decidir que imagens seriam úteis para este tópico, escolha 'retrieval' para buscar material visual relevante.</ação>
3. <observação>Reflexão sobre o processo de ensino</observação>
4. <resposta>Sua resposta final para o aluno (clara, estruturada e adaptada ao contexto)
   * Introduza o tópico atual e conecte-o ao plano de estudos
   * Se uma imagem for recuperada: VOCÊ DEVE FAZER REFERÊNCIAS EXPLICÍTIMAS a ela no início da sua explicação (ex: "Como podemos ver na imagem...", "Conforme ilustrado na figura..."). Incorpore detalhes da descrição da imagem para mostrar como ela se relaciona com o conteúdo.
   * Explique os conceitos fundamentais com clareza e profundidade
   * Forneça exemplos relevantes e aplicações práticas
   * Inclua perguntas de reflexão para estimular o pensamento crítico
   * Conclua com uma síntese e aponte para o próximo tópico do plano
</resposta>

IMPORTANTE:
- VOCÊ DEVE SEMPRE COLOCAR A RESPOSTA FINAL DENTRO DAS TAGS <resposta></resposta>
- Apenas o conteúdo dentro das tags <resposta></resposta> será mostrado ao aluno
- Todo seu raciocínio, ações e observações devem estar nas tags apropriadas e não serão visíveis
- NÃO ENVIE para o front os conteúdos de <pensamento>, <ação> e <observação>
"""
    return ChatPromptTemplate.from_template(REACT_PROMPT)

def identify_current_step(plano_execucao: List[Dict]) -> ExecutionStep:
    if not plano_execucao:
        return ExecutionStep(
            titulo="Introdução ao Tema",
            duracao="30 minutos",
            descricao="Introdução aos conceitos básicos do tema e exploração inicial dos objetivos de aprendizagem",
            conteudo=["Conceitos fundamentais", "Objetivos de aprendizagem", "Contextualização do tema"],
            recursos=[{"tipo": "texto", "titulo": "Material introdutório"}],
            atividade={"tipo": "reflexão", "descrição": "Reflexão inicial sobre o tema"},
            progresso=0
        )
    for step in plano_execucao:
        if "progresso" not in step:
            step["progresso"] = 0
        else:
            step["progresso"] = min(max(float(step["progresso"]), 0), 100)
        for campo in ["conteudo", "recursos", "atividade"]:
            if campo not in step or not step[campo]:
                if campo == "conteudo":
                    step[campo] = ["Conceito principal", "Aplicações práticas"]
                elif campo == "recursos":
                    step[campo] = [{"tipo": "texto", "titulo": "Material de apoio"}]
                elif campo == "atividade":
                    step[campo] = {"tipo": "exercício", "descrição": "Praticar o conceito aprendido"}
    for step in plano_execucao:
        current_progress = step["progresso"]
        if 0 < current_progress < 100:
            return ExecutionStep(
                titulo=step["titulo"],
                duracao=step["duracao"],
                descricao=step["descricao"],
                conteudo=step["conteudo"],
                recursos=step["recursos"],
                atividade=step["atividade"],
                progresso=current_progress
            )
    for step in plano_execucao:
        if step["progresso"] == 0:
            return ExecutionStep(
                titulo=step["titulo"],
                duracao=step["duracao"],
                descricao=step["descricao"],
                conteudo=step["conteudo"],
                recursos=step["recursos"],
                atividade=step["atividade"],
                progresso=0
            )
    last_step = plano_execucao[-1]
    return ExecutionStep(
        titulo=f"Revisão: {last_step['titulo']}",
        duracao=last_step["duracao"],
        descricao=f"Revisão dos conceitos principais e avaliação do aprendizado em: {last_step['descricao']}",
        conteudo=last_step["conteudo"],
        recursos=last_step["recursos"],
        atividade=last_step["atividade"],
        progresso=last_step["progresso"]
    )

# Criação do nó ReAct
def create_react_node(retrieval_tools: RetrievalTools, web_tools: WebSearchTools, progress_manager: StudyProgressManager):
    react_prompt = create_react_prompt()
    model = ChatOpenAI(model="gpt-4o", temperature=0.2)
    model_stream = ChatOpenAI(model="gpt-4o", temperature=0.2, streaming=True)
    
    async def react_process(state: AgentState) -> AgentState:
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        chat_history = format_chat_history(state["chat_history"], max_messages=5)
        try:
            has_image_context = False
            image_description = ""
            if "extracted_context" in state and isinstance(state["extracted_context"], dict):
                context = state["extracted_context"]
                if "image" in context and isinstance(context["image"], dict):
                    image_data = context["image"]
                    if (image_data.get("type") == "image" and 
                        image_data.get("description") and 
                        image_data.get("image_bytes")):
                        has_image_context = True
                        image_description = image_data.get("description", "")
                        print(f"[REACT_AGENT] Found existing image context: {image_description[:50]}...")
                    else:
                        print(f"[REACT_AGENT] Image data found but incomplete.")
                if has_image_context and image_description:
                    latest_question = f"{latest_question}\n\n[CONTEXTO: Há uma imagem disponível com a seguinte descrição: '{image_description}'. Faça referência a ela na sua resposta.]"
                    print(f"[REACT_AGENT] Added image context to question")
                elif "image" in context:
                    latest_question = f"{latest_question}\n\n[CONTEXTO: NÃO há imagens disponíveis para este tópico. Não mencione ou faça referência a imagens na sua resposta.]"
                    print(f"[REACT_AGENT] Added warning about no valid images")
            
            current_plan = state.get("current_plan", "{}")
            plano_execucao = []
            try:
                if isinstance(current_plan, dict) and "plano_execucao" in current_plan:
                    print(f"[REACT_AGENT] Plano encontrado em formato de dicionário")
                    plano_execucao = current_plan.get("plano_execucao", [])
                else:
                    print(f"[REACT_AGENT] Tentando converter plano de JSON")
                    plano_execucao = json.loads(current_plan).get("plano_execucao", [])
                current_progress = state.get("current_progress", {})
                if isinstance(current_progress, dict) and "plano_execucao" in current_progress:
                    print(f"[REACT_AGENT] Usando plano do current_progress")
                    plano_execucao = current_progress.get("plano_execucao", plano_execucao)
                print(f"[REACT_AGENT] Plano de execução encontrado com {len(plano_execucao)} etapas")
                current_step = identify_current_step(plano_execucao)
            except Exception as plan_error:
                print(f"[REACT_AGENT] Erro ao processar plano: {str(plan_error)}")
                current_step = ExecutionStep(
                    titulo="Introdução ao Tema",
                    duracao="30 minutos",
                    descricao="Exploração dos conceitos fundamentais do tema",
                    conteudo=["Conceitos fundamentais", "Objetivos de aprendizagem"],
                    recursos=[{"tipo": "texto", "titulo": "Material introdutório"}],
                    atividade={"tipo": "reflexão", "descrição": "Reflexão inicial sobre o tema"},
                    progresso=0
                )
            
            user_profile = state.get("user_profile", {})
            estilos = user_profile.get("EstiloAprendizagem", {})
            required_styles = [("Percepcao", "não especificado"), 
                              ("Entrada", "não especificado"), 
                              ("Processamento", "não especificado"), 
                              ("Entendimento", "não especificado")]
            for style, default in required_styles:
                if style not in estilos:
                    estilos[style] = default
            is_first_message = len(state["chat_history"]) == 0
            is_vague_message = len(latest_question.split()) < 5 or any(term in latest_question.lower() 
                                                                      for term in ["oi", "olá", "começar", "iniciar", 
                                                                                   "continuar", "seguir", "ok", "bom", 
                                                                                   "entendi", "compreendi"])
            enriched_question = latest_question
            if is_first_message or is_vague_message:
                conteudos = ", ".join(current_step.conteudo[:3]) if current_step.conteudo else "conceitos fundamentais"
                enriched_question = (
                    f"{latest_question}\n\n"
                    f"[CONTEXTO PARA O TUTOR: Esta é uma mensagem {'inicial' if is_first_message else 'vaga'} do aluno. "
                    f"Inicie proativamente o ensino do tópico atual: '{current_step.titulo}', "
                    f"explicando os conceitos principais: {conteudos}, seguindo o plano de estudos. "
                    f"Não mencione este contexto na sua resposta, apenas ensine o conteúdo de forma natural.]"
                )
                print(f"[REACT_AGENT] Mensagem enriquecida para estimular ensino proativo")
            
            params = {
                "nome": user_profile.get("Nome", "Estudante"),
                "percepcao": estilos.get("Percepcao", "não especificado"),
                "entrada": estilos.get("Entrada", "não especificado"),
                "processamento": estilos.get("Processamento", "não especificado"),
                "entendimento": estilos.get("Entendimento", "não especificado"),
                "titulo": current_step.titulo,
                "descricao": current_step.descricao,
                "progresso": current_step.progresso,
                "question": enriched_question,
                "chat_history": chat_history
            }
            print(f"[REACT_AGENT] Processing question: '{latest_question[:50]}...'")
            reaction = await model.ainvoke(react_prompt.format(**params))
            react_content = reaction.content
            thoughts = extract_section(react_content, "pensamento")
            action = extract_section(react_content, "ação")
            observation = extract_section(react_content, "observação")
            print("\n[REACT_AGENT] ==== INÍCIO DO PENSAMENTO ====")
            print(thoughts)
            print("[REACT_AGENT] ==== FIM DO PENSAMENTO ====\n")
            print(f"[REACT_AGENT] Ação escolhida: {action[:100]}...")
            print("\n[REACT_AGENT] ==== INÍCIO DA OBSERVAÇÃO ====")
            print(observation)
            print("[REACT_AGENT] ==== FIM DA OBSERVAÇÃO ====\n")
            # Aqui usamos extract_final_answer para obter somente o conteúdo dentro de <resposta>
            final_response = extract_final_answer(react_content)
            if current_step.titulo.lower() not in final_response.lower() and len(final_response) < 1000:
                print(f"[REACT_AGENT] Resposta não contém menção ao tópico atual - adicionando contexto")
                final_response = (
                    f"Vamos continuar explorando o tópico '{current_step.titulo}'.\n\n{final_response}"
                )
            action_result = await process_action(
                action, 
                state, 
                latest_question,
                retrieval_tools, 
                web_tools, 
                progress_manager
            )
            response_content = format_response_with_image(final_response, action_result)
            new_state = state.copy()
            user_message = HumanMessage(content=latest_question)
            if isinstance(response_content, dict) and "type" in response_content and response_content["type"] == "multimodal":
                response = AIMessage(content=json.dumps(response_content))
                history_message = AIMessage(content=json.dumps(response_content))
            else:
                response = AIMessage(content=response_content)
                history_message = response
            new_state["messages"] = list(state["messages"]) + [response]
            new_state["chat_history"] = list(state["chat_history"]) + [
                user_message,
                history_message
            ]
            print(f"[REACT_AGENT] Adicionado ao histórico.")
            new_state["thoughts"] = thoughts
            if "context" in action_result:
                new_state["extracted_context"] = action_result["context"]
            if "progress" in action_result:
                new_state["current_progress"] = action_result["progress"]
            is_substantive = len(final_response) > 800 and "direct_teaching" in action.lower()
            if is_substantive and current_step.progresso < 100:
                try:
                    print(f"[REACT_AGENT] Interação substantiva detectada - atualizando progresso automaticamente")
                    progress_increment = min(10, 100 - current_step.progresso)
                    await progress_manager.update_step_progress(
                        state["session_id"],
                        next((i for i, step in enumerate(plano_execucao) 
                             if step.get("titulo") == current_step.titulo), 0),
                        current_step.progresso + progress_increment
                    )
                except Exception as progress_error:
                    print(f"[REACT_AGENT] Erro ao atualizar progresso: {str(progress_error)}")
            return new_state
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_message = "Desculpe, encontrei um erro ao processar sua mensagem. Por favor, tente novamente."
            response = AIMessage(content=error_message)
            new_state = state.copy()
            new_state["messages"] = list(state["messages"]) + [response]
            new_state["chat_history"] = list(state["chat_history"]) + [
                HumanMessage(content=latest_question),
                response
            ]
            new_state["error"] = str(e)
            return new_state
    
    async def react_process_streaming_fixed(state: AgentState):
        start_time = time.time()
        step_time = time.time()
        print(f"\n[REACT_AGENT] Starting streaming process at {datetime.now().strftime('%H:%M:%S')}")
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        chat_history = format_chat_history(state["chat_history"], max_messages=5)
        has_image_context = False
        image_description = ""
        try:
            yield {"type": "processing", "content": "Analisando sua pergunta..."}
            if "extracted_context" in state and isinstance(state["extracted_context"], dict):
                context = state["extracted_context"]
                if "image" in context and isinstance(context["image"], dict):
                    image_data = context["image"]
                    if (image_data.get("type") == "image" and 
                        image_data.get("description") and 
                        image_data.get("image_bytes")):
                        has_image_context = True
                        image_description = image_data.get("description", "")
                        latest_question = f"{latest_question}\n\n[CONTEXTO: Há uma imagem disponível com a seguinte descrição: '{image_description}'. Faça referência a ela na sua resposta.]"
                    elif "image" in context:
                        latest_question = f"{latest_question}\n\n[CONTEXTO: NÃO há imagens disponíveis para este tópico. Não mencione ou faça referência a imagens na sua resposta.]"
            current_plan = state.get("current_plan", "{}")
            plano_execucao = []
            current_step = None
            try:
                if isinstance(current_plan, dict) and "plano_execucao" in current_plan:
                    plano_execucao = current_plan.get("plano_execucao", [])
                else:
                    plano_execucao = json.loads(current_plan).get("plano_execucao", [])
                current_progress = state.get("current_progress", {})
                if isinstance(current_progress, dict) and "plano_execucao" in current_progress:
                    plano_execucao = current_progress.get("plano_execucao", plano_execucao)
                current_step = identify_current_step(plano_execucao)
            except Exception as e:
                print(f"[REACT_AGENT] Error in plan processing: {str(e)}")
                current_step = ExecutionStep(
                    titulo="Etapa não encontrada",
                    duracao="N/A",
                    descricao="Informações do plano de execução não disponíveis",
                    conteudo=[],
                    recursos=[],
                    atividade={},
                    progresso=0
                )
            user_profile = state.get("user_profile", {})
            estilos = user_profile.get("EstiloAprendizagem", {})
            for style, default in [("Percepcao", "não especificado"), 
                                   ("Entrada", "não especificado"), 
                                   ("Processamento", "não especificado"), 
                                   ("Entendimento", "não especificado")]:
                if style not in estilos:
                    estilos[style] = default
            is_first_message = len(state["chat_history"]) == 0
            is_vague_message = len(latest_question.split()) < 5 or any(term in latest_question.lower() 
                                                                      for term in ["oi", "olá", "começar", "iniciar", 
                                                                                   "continuar", "seguir", "ok", "bom", 
                                                                                   "entendi", "compreendi"])
            enriched_question = latest_question
            if is_first_message or is_vague_message:
                conteudos = ", ".join(current_step.conteudo[:3]) if current_step.conteudo else "conceitos fundamentais"
                enriched_question = (
                    f"{latest_question}\n\n"
                    f"[CONTEXTO PARA O TUTOR: Esta é uma mensagem {'inicial' if is_first_message else 'vaga'} do aluno. "
                    f"Inicie proativamente o ensino do tópico atual: '{current_step.titulo}', "
                    f"explicando os conceitos principais: {conteudos}, seguindo o plano de estudos. "
                    f"Não mencione este contexto na sua resposta, apenas ensine o conteúdo de forma natural.]"
                )
                print(f"[REACT_AGENT] Mensagem enriquecida para estimular ensino proativo")
            params = {
                "nome": user_profile.get("Nome", "Estudante"),
                "percepcao": estilos.get("Percepcao", "não especificado"),
                "entrada": estilos.get("Entrada", "não especificado"),
                "processamento": estilos.get("Processamento", "não especificado"),
                "entendimento": estilos.get("Entendimento", "não especificado"),
                "titulo": current_step.titulo,
                "descricao": current_step.descricao,
                "progresso": current_step.progresso,
                "question": latest_question,
                "chat_history": chat_history
            }
            yield {"type": "processing", "content": "Elaborando uma resposta..."}
            stream = model_stream.astream(react_prompt.format(**params))
            full_reaction = ""
            action_str = ""
            action_detected = False
            action_task = None
            import re
            cleanr = re.compile('<.*?>')
            sent_text = set()
            async for chunk in stream:
                print(f"[REACT_AGENT] Received chunk: {chunk.content}")
                if chunk.content:
                    full_reaction += chunk.content
                    resp_match = re.search(r'<resposta>(.*?)</resposta>', full_reaction, re.DOTALL)
                    if resp_match:
                        final_answer = resp_match.group(1).strip()
                        answer_sentences = re.split(r'(?<=[.!?])\s+', final_answer)
                        for sent in answer_sentences:
                            clean_sent = sent.strip()
                            if clean_sent and clean_sent not in sent_text and len(clean_sent) > 5:
                                sent_text.add(clean_sent)
                                yield {"type": "chunk", "content": clean_sent + " "}
                                print(f"[REACT_AGENT] Streaming FINAL ANSWER chunk: {clean_sent[:30]}...")
                    else:
                        clean_text = re.sub(r'<pensamento>.*?</pensamento>', '', full_reaction, flags=re.DOTALL)
                        clean_text = re.sub(r'<ação>.*?</ação>', '', clean_text, flags=re.DOTALL)
                        clean_text = re.sub(r'<action>.*?</action>', '', clean_text, flags=re.DOTALL)
                        clean_text = re.sub(r'<observação>.*?</observação>', '', clean_text, flags=re.DOTALL)
                        clean_text = re.sub(r'<observation>.*?</observation>', '', clean_text, flags=re.DOTALL)
                        clean_text = re.sub(cleanr, '', clean_text).strip()
                        paragraphs = re.split(r'\n\s*\n', clean_text)
                        filtered_text = ""
                        for para in paragraphs:
                            if len(para.strip()) > 30 and not re.match(r'^\d+\.', para.strip()):
                                filtered_text = para.strip()
                                break
                        if filtered_text:
                            sentences = re.split(r'(?<=[.!?])\s+', filtered_text)
                            for sent in sentences[:-1]:
                                clean_sent = sent.strip()
                                if clean_sent and clean_sent not in sent_text and len(clean_sent) > 5:
                                    sent_text.add(clean_sent)
                                    yield {"type": "chunk", "content": clean_sent + " "}
                                    print(f"[REACT_AGENT] Streaming filtered content: {clean_sent[:30]}...")
                    if not action_detected and ("</ação>" in chunk.content or "</action>" in chunk.content):
                        action_str = extract_section(full_reaction, "ação")
                        if not action_str:
                            action_str = extract_section(full_reaction, "action")
                        if action_str:
                            action_detected = True
                            action_task = asyncio.create_task(process_action(
                                action_str, state, latest_question, retrieval_tools, web_tools, progress_manager
                            ))
                            yield {"type": "processing", "content": "Executando ações necessárias..."}
            if not action_detected:
                action_str = extract_section(full_reaction, "ação") or extract_section(full_reaction, "action") or ""
                action_task = asyncio.create_task(process_action(
                    action_str, state, latest_question, retrieval_tools, web_tools, progress_manager
                ))
            format_task = asyncio.create_task(asyncio.to_thread(extract_final_answer, full_reaction))
            final_response = await format_task
            action_result = {"type": "none", "content": "No action result"}
            try:
                if action_task:
                    action_result = await asyncio.wait_for(action_task, timeout=1.5)
            except asyncio.TimeoutError:
                print(f"[REACT_AGENT] Action processing timed out, continuing with response")
                action_result = {"type": "timeout", "content": "Action timed out"}
            except Exception as e:
                print(f"[REACT_AGENT] Error processing action: {str(e)}")
                action_result = {"type": "error", "content": f"Error: {str(e)}"}
            response_content = format_response_with_image(final_response, action_result)
            is_substantive = len(final_response) > 800 and action_str and "direct_teaching" in action_str.lower()
            if is_substantive and current_step and current_step.progresso < 100:
                try:
                    progress_increment = min(10, 100 - current_step.progresso)
                    await progress_manager.update_step_progress(
                        state["session_id"],
                        next((i for i, step in enumerate(plano_execucao) 
                            if step.get("titulo") == current_step.titulo), 0),
                        current_step.progresso + progress_increment
                    )
                except Exception as e:
                    print(f"[REACT_AGENT] Progress update error: {str(e)}")
            if isinstance(response_content, dict) and response_content.get("type") == "multimodal":
                text_content = response_content["content"]
                import re
                paragraphs = re.split(r'\n\s*\n', text_content)
                text_chunks = []
                for paragraph in paragraphs:
                    if len(paragraph) < 300:
                        text_chunks.append(paragraph)
                    else:
                        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                        for sentence in sentences:
                            if len(sentence) < 200:
                                text_chunks.append(sentence)
                            else:
                                words = sentence.split()
                                current_chunk = ""
                                for word in words:
                                    if len(current_chunk) + len(word) + 1 > 100:
                                        text_chunks.append(current_chunk)
                                        current_chunk = word
                                    else:
                                        current_chunk = current_chunk + " " + word if current_chunk else word
                                if current_chunk:
                                    text_chunks.append(current_chunk)
                for i in range(len(text_chunks)-1):
                    if text_chunks[i].endswith(('.', '!', '?')) and not text_chunks[i+1].startswith(('.', '!', '?', ',', ':', ';')):
                        text_chunks[i] = text_chunks[i] + "\n\n"
                for i, chunk in enumerate(text_chunks):
                    yield {"type": "chunk", "content": chunk}
                    if i > 0 and i % 10 == 0:
                        print(f"[REACT_AGENT] Sent {i}/{len(text_chunks)} text chunks")
                yield {
                    "type": "image", 
                    "content": response_content["content"],
                    "image": response_content["image"]
                }
                response = AIMessage(content=json.dumps(response_content))
                history_message = AIMessage(content=json.dumps(response_content))
            else:
                text_content = response_content
                import re
                paragraphs = re.split(r'\n\s*\n', text_content)
                text_chunks = []
                for paragraph in paragraphs:
                    if len(paragraph) < 300:
                        text_chunks.append(paragraph)
                    else:
                        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                        for sentence in sentences:
                            if len(sentence) < 200:
                                text_chunks.append(sentence)
                            else:
                                words = sentence.split()
                                current_chunk = ""
                                for word in words:
                                    if len(current_chunk) + len(word) + 1 > 100:
                                        text_chunks.append(current_chunk)
                                        current_chunk = word
                                    else:
                                        current_chunk = current_chunk + " " + word if current_chunk else word
                                if current_chunk:
                                    text_chunks.append(current_chunk)
                for i in range(len(text_chunks)-1):
                    if text_chunks[i].endswith(('.', '!', '?')) and not text_chunks[i+1].startswith(('.', '!', '?', ',', ':', ';')):
                        text_chunks[i] = text_chunks[i] + "\n\n"
                for i, chunk in enumerate(text_chunks):
                    yield {"type": "chunk", "content": chunk}
                    if i > 0 and i % 20 == 0:
                        print(f"[REACT_AGENT] Sent {i}/{len(text_chunks)} text chunks")
                response = AIMessage(content=text_content)
                history_message = AIMessage(content=text_content)
            yield {"type": "complete", "content": f"Resposta completa em {time.time() - start_time:.2f}s."}
            state["messages"] = list(state["messages"]) + [response]
            state["chat_history"] = list(state["chat_history"]) + [
                HumanMessage(content=latest_question),
                history_message
            ]
            state["thoughts"] = extract_section(full_reaction, "pensamento")
            if "context" in action_result:
                state["extracted_context"] = action_result["context"]
            if "progress" in action_result:
                state["current_progress"] = action_result["progress"]
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield {"type": "error", "content": f"Desculpe, encontrei um erro ao processar sua pergunta: {str(e)}"}
            yield {"type": "complete", "content": "Streaming concluído com erro."}
            error_message = "Desculpe, encontrei um erro ao processar sua pergunta. Por favor, tente novamente."
            response = AIMessage(content=error_message)
            state["messages"] = list(state["messages"]) + [response]
            state["chat_history"] = list(state["chat_history"]) + [
                HumanMessage(content=latest_question),
                response
            ]
            state["error"] = str(e)
    
    return react_process, react_process_streaming_fixed

# Função de utilidade para verificar se uma interação merece progresso
def is_substantive_interaction(response_text, action_text=None):
    if not response_text or len(response_text) < 800:
        return False
    if not action_text:
        return False
    try:
        return "direct_teaching" in action_text.lower()
    except:
        return False

# Compilar expressões regulares
import re
_PENSAMENTO_PATTERN = re.compile(r"<pensamento>(.*?)</pensamento>", re.DOTALL)
_ACAO_PATTERN = re.compile(r"<ação>(.*?)</ação>", re.DOTALL)
_ACTION_PATTERN = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_OBSERVACAO_PATTERN = re.compile(r"<observação>(.*?)</observação>", re.DOTALL)
_OBSERVATION_PATTERN = re.compile(r"<observation>(.*?)</observation>", re.DOTALL)
_NUMBERED_PATTERN = re.compile(r"^(\d+\.\s*)+", re.DOTALL)
_EMPTY_LINES_PATTERN = re.compile(r"\n\s*\n", re.DOTALL)

def extract_section(text: str, section_name: str) -> str:
    if section_name == "pensamento":
        matches = _PENSAMENTO_PATTERN.findall(text)
    elif section_name == "ação":
        matches = _ACAO_PATTERN.findall(text)
        if not matches:
            matches = _ACTION_PATTERN.findall(text)
    elif section_name == "observação":
        matches = _OBSERVACAO_PATTERN.findall(text)
        if not matches:
            matches = _OBSERVATION_PATTERN.findall(text)
    else:
        pattern = f"<{section_name}>(.*?)</{section_name}>"
        matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return ""

def remove_react_tags(text: str) -> str:
    text = _PENSAMENTO_PATTERN.sub('', text)
    text = _ACAO_PATTERN.sub('', text)
    text = _OBSERVACAO_PATTERN.sub('', text)
    text = _ACTION_PATTERN.sub('', text)
    text = _OBSERVATION_PATTERN.sub('', text)
    text = _NUMBERED_PATTERN.sub('', text)
    text = _EMPTY_LINES_PATTERN.sub('\n\n', text)
    return text.strip()

async def process_action(action: str, state: AgentState, question: str, 
                        retrieval_tools: RetrievalTools, web_tools: WebSearchTools,
                        progress_manager: StudyProgressManager) -> Dict[str, Any]:
    action_lower = action.lower()
    print(f"[REACT_ACTION] Processing action: '{action_lower[:50]}...'")
    if "retrieval" in action_lower:
        return await process_retrieval_action(question, retrieval_tools, state)
    elif "websearch" in action_lower:
        return await process_websearch_action(question, web_tools)
    elif "progress_update" in action_lower:
        return await process_progress_action(action, state, progress_manager)
    elif "direct_teaching" in action_lower:
        if not state.get("extracted_context"):
            try:
                print(f"[REACT_ACTION] No context found for direct_teaching, retrieving...")
                context_result = await process_retrieval_action(question, retrieval_tools, state)
                return context_result
            except Exception as e:
                print(f"[REACT_ACTION] Failed to retrieve context: {e}")
        return {"type": "direct_teaching", "content": "Explicação estruturada fornecida"}
    elif "challenge_thinking" in action_lower:
        return {"type": "challenge_thinking", "content": "Desafio de pensamento apresentado"}
    elif "guided_discovery" in action_lower:
        return {"type": "guided_discovery", "content": "Descoberta guiada iniciada"}
    elif "socratic_questioning" in action_lower:
        return {"type": "socratic_questioning", "content": "Questionamento socrático aplicado"}
    elif "direct_response" in action_lower:
        return {"type": "direct", "content": "Resposta direta fornecida"}
    elif "analyze_response" in action_lower:
        return {"type": "analysis", "content": "Análise da resposta realizada"}
    elif "suggest_activity" in action_lower:
        return {"type": "activity", "content": "Nova atividade sugerida"}
    print(f"[REACT_ACTION] Using default action, attempting retrieval first")
    try:
        context_result = await process_retrieval_action(question, retrieval_tools, state)
        if context_result and "context" in context_result:
            return context_result
    except Exception as e:
        print(f"[REACT_ACTION] Error in default retrieval: {e}")
    return {"type": "direct_teaching", "content": "Explicação estruturada fornecida"}

async def process_retrieval_action(question: str, tools: RetrievalTools, state: AgentState) -> Dict[str, Any]:
    start_time = time.time()
    try:
        print(f"[REACT_ACTION] Starting retrieval for: '{question[:50]}...'")
        cache_key = question.strip().lower()[:100]
        if hasattr(tools, '_query_cache') and cache_key in tools._query_cache:
            print(f"[REACT_ACTION] Using cached retrieval results for similar query")
            context_results = tools._query_cache[cache_key]
        else:
            tools.state = state
            context_results = await tools.parallel_context_retrieval(question)
            if not hasattr(tools, '_query_cache'):
                tools._query_cache = {}
            tools._query_cache[cache_key] = context_results
        has_text = bool(context_results.get("text", ""))
        has_image = isinstance(context_results.get("image", {}), dict) and context_results["image"].get("type") == "image" and context_results["image"].get("image_bytes")
        has_table = isinstance(context_results.get("table", {}), dict) and context_results["table"].get("content")
        if "relevance_analysis" not in context_results or not isinstance(context_results["relevance_analysis"], dict):
            print(f"[REACT_ACTION] Creating default relevance analysis")
            context_results['relevance_analysis'] = tools._get_default_analysis()
        return {
            "type": "retrieval",
            "content": "Contexto recuperado com sucesso",
            "context": context_results,
            "has_image": has_image,
            "has_text": has_text,
            "has_table": has_table
        }
    except Exception as e:
        print(f"[REACT_ACTION] Retrieval error: {str(e)} ({time.time() - start_time:.2f}s)")
        return {
            "type": "error",
            "content": f"Erro na recuperação de contexto: {str(e)}"
        }

async def process_websearch_action(question: str, web_tools: WebSearchTools) -> Dict[str, Any]:
    start_time = time.time()
    try:
        print(f"[REACT_ACTION] Starting web search for: '{question[:50]}...'")
        wiki_time = time.time()
        wiki_result = web_tools.search_wikipedia(question)
        print(f"[REACT_ACTION] Wikipedia search completed in {time.time() - wiki_time:.2f}s")
        yt_time = time.time()
        youtube_result = web_tools.search_youtube(question)
        print(f"[REACT_ACTION] YouTube search completed in {time.time() - yt_time:.2f}s")
        resources = {
            "wikipedia": wiki_result,
            "youtube": youtube_result
        }
        extracted_context = {
            "text": f"Wikipedia:\n{wiki_result}\n\nYouTube:\n{youtube_result}",
            "image": {"type": "image", "content": None, "description": ""},
            "table": {"type": "table", "content": None},
            "relevance_analysis": {
                "text": {"score": 1.0, "reason": "Informação obtida da web"},
                "image": {"score": 0.0, "reason": "Nenhuma imagem disponível"},
                "table": {"score": 0.0, "reason": "Nenhuma tabela disponível"},
                "recommended_context": "text"
            }
        }
        print(f"[REACT_ACTION] Web search completed successfully in {time.time() - start_time:.2f}s")
        return {
            "type": "websearch",
            "content": "Busca web realizada com sucesso",
            "web_results": resources,
            "context": extracted_context
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[REACT_ACTION] Web search error: {str(e)} ({time.time() - start_time:.2f}s)")
        return {
            "type": "error",
            "content": f"Erro na busca web: {str(e)}"
        }

async def process_progress_action(action: str, state: AgentState, progress_manager: StudyProgressManager) -> Dict[str, Any]:
    start_time = time.time()
    try:
        print(f"[REACT_ACTION] Processing progress update")
        import re
        progress_match = re.search(r'progress_update.*?(\d+)', action)
        progress_value = int(progress_match.group(1)) if progress_match else 10
        print(f"[REACT_ACTION] Progress value extracted: {progress_value}%")
        session_id = state["session_id"]
        study_progress = await progress_manager.get_study_progress(session_id)
        if not study_progress:
            return {"type": "error", "content": "Dados de progresso não encontrados"}
        plano_execucao = study_progress['plano_execucao']
        current_step = None
        step_index = 0
        for idx, step in enumerate(plano_execucao):
            if step['progresso'] < 100:
                current_step = step
                step_index = idx
                break
        if not current_step:
            return {"type": "info", "content": "Todas as etapas completadas"}
        current_progress = current_step['progresso']
        new_progress = min(current_progress + progress_value, 100)
        update_success = await progress_manager.update_step_progress(
            session_id,
            step_index,
            new_progress
        )
        if update_success:
            study_summary = await progress_manager.get_study_summary(session_id)
            updated_progress = await progress_manager.get_study_progress(session_id)
            print(f"[REACT_ACTION] Progress update successful: {current_progress}% → {new_progress}% ({time.time() - start_time:.2f}s)")
            return {
                "type": "progress_update",
                "content": f"Progresso atualizado com sucesso: {new_progress}%",
                "progress": updated_progress,
                "summary": study_summary
            }
        else:
            print(f"[REACT_ACTION] Progress update failed ({time.time() - start_time:.2f}s)")
            return {"type": "error", "content": "Falha ao atualizar progresso"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[REACT_ACTION] Progress update error: {str(e)} ({time.time() - start_time:.2f}s)")
        return {
            "type": "error",
            "content": f"Erro na atualização de progresso: {str(e)}"
        }

def format_response_with_image(response: str, action_result: Dict[str, Any]) -> Any:
    print(f"[REACT_AGENT] Formatting response with action_result type: {action_result.get('type', 'unknown')}")
    has_image = action_result.get("has_image", False)
    image_bytes = None
    image_description = ""
    if action_result.get("type") == "retrieval" and "context" in action_result:
        context = action_result["context"]
        image_data = context.get("image", {})
        print(f"[REACT_AGENT] Image data keys: {image_data.keys() if isinstance(image_data, dict) else 'not a dict'}")
        if (isinstance(image_data, dict) and 
            image_data.get("type") == "image" and 
            image_data.get("image_bytes")):
            image_bytes = image_data["image_bytes"]
            image_description = image_data.get("description", "")
            print(f"[REACT_AGENT] Found valid image in context, size: {len(image_bytes) if image_bytes else 'unknown'} bytes")
            print(f"[REACT_AGENT] Image description: {image_description[:100]}...")
            if not image_bytes:
                has_image = False
                print(f"[REACT_AGENT] Image rejected: Missing image bytes")
            elif "relevance_analysis" in context:
                image_score = context["relevance_analysis"].get("image", {}).get("score", 0)
                image_reason = context["relevance_analysis"].get("image", {}).get("reason", "")
                print(f"[REACT_AGENT] Image relevance score: {image_score}, reason: {image_reason}")
                if image_score > 0.5:
                    has_image = True
                    print(f"[REACT_AGENT] Image considered highly relevant with score {image_score}")
                elif image_score > 0.3:
                    user_profile = state.get("user_profile", {})
                    learning_styles = user_profile.get("EstiloAprendizagem", {})
                    is_visual_learner = learning_styles.get("Entrada", "").lower() == "visual"
                    if is_visual_learner:
                        has_image = True
                        print(f"[REACT_AGENT] Image with medium relevance ({image_score}) included for visual learner")
                    else:
                        print(f"[REACT_AGENT] Image with medium relevance not included for non-visual learner")
                        has_image = False
                else:
                    print(f"[REACT_AGENT] Image relevance too low ({image_score}), not including image")
                    has_image = False
            else:
                if image_description and len(image_description) > 30:
                    has_image = True
                    print(f"[REACT_AGENT] Using image based on description without relevance score")
                else:
                    has_image = False
                    print(f"[REACT_AGENT] Image description too short or missing, not including image")
        else:
            has_image = False
            print(f"[REACT_AGENT] No valid image data found in context")
    if has_image and any(ref in response.lower() for ref in ["imagem", "figura", "ilustração", "visual", "diagrama"]):
        print(f"[REACT_AGENT] Response correctly contains references to the available image")
    elif has_image:
        print(f"[REACT_AGENT] WARNING: Image is available but response doesn't reference it explicitly")
    elif not has_image and any(ref in response.lower() for ref in ["imagem", "figura", "ilustração", "visual", "diagrama", 
                                                                    "como podemos ver", "conforme mostrado", "observe na"]):
        print(f"[REACT_AGENT] WARNING: Response mentions images but no valid images are available")
    if has_image and image_bytes:
        try:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            print(f"[REACT_AGENT] Successfully encoded image to base64")
            return {
                "type": "multimodal",
                "content": response,
                "image": f"data:image/jpeg;base64,{base64_image}"
            }
        except Exception as e:
            print(f"[REACT_AGENT] Error encoding image: {e}")
            return response
    else:
        print(f"[REACT_AGENT] No valid image found, returning text only")
        return response

class ReactTutorWorkflow:
    def __init__(
        self,
        qdrant_handler,
        student_email: str,
        disciplina: str,
        session_id: str,
        image_collection
    ):
        self.session_id = session_id
        self.student_email = student_email
        self.disciplina = disciplina
        self.progress_manager = StudyProgressManager()
        initial_state = AgentState(
            messages=[],
            current_plan="",
            user_profile={},
            extracted_context={},
            next_step=None,
            iteration_count=0,
            chat_history=[],
            needs_retrieval=True,
            evaluation_reason="",
            web_search_results={},
            current_progress=None,
            session_id=session_id,
            thoughts=""
        )
        self.retrieval_tools = RetrievalTools(
            qdrant_handler=qdrant_handler,
            student_email=student_email,
            disciplina=disciplina,
            session_id=session_id,
            image_collection=image_collection,
            state=initial_state
        )
        self.retrieval_tools._query_cache = {}
        self.web_tools = WebSearchTools()
        self.react_node, self.react_stream_node = create_react_node(
            self.retrieval_tools, 
            self.web_tools,
            self.progress_manager
        )
        self.workflow = self.create_workflow()
        
    def create_workflow(self) -> Graph:
        workflow = Graph()
        workflow.add_node("react", self.react_node)
        workflow.add_edge("react", END)
        workflow.set_entry_point("react")
        return workflow.compile()
    
    async def invoke(
        self, 
        query: str, 
        student_profile: dict, 
        current_plan=None, 
        chat_history=None
    ) -> dict:
        start_time = time.time()
        try:
            validated_profile = student_profile
            await self.progress_manager.sync_progress_state(self.session_id)
            current_progress = await self.progress_manager.get_study_progress(self.session_id)
            if chat_history is None:
                chat_history = []
            elif not isinstance(chat_history, list):
                chat_history = list(chat_history)
            recent_history = chat_history[-10:]
            initial_state = AgentState(
                messages=[HumanMessage(content=query)],
                current_plan=current_plan if current_plan else "",
                user_profile=validated_profile,
                extracted_context={},
                next_step=None,
                iteration_count=0,
                chat_history=recent_history,
                needs_retrieval=True,
                evaluation_reason="",
                web_search_results={},
                current_progress=current_progress,
                session_id=self.session_id,
                thoughts=""
            )
            result = await self.workflow.ainvoke(initial_state)
            study_summary = await self.progress_manager.get_study_summary(self.session_id)
            chat_history = result["chat_history"]
            print(f"[REACT_AGENT] Result contains {len(chat_history)} chat history messages")
            for i, msg in enumerate(chat_history):
                msg_type = type(msg).__name__
                content_preview = str(msg.content)[:30] + "..." if len(str(msg.content)) > 30 else str(msg.content)
                print(f"[REACT_AGENT] Result chat_history[{i}]: {msg_type} - {content_preview}")
            final_result = {
                "messages": result["messages"],
                "final_plan": result.get("current_plan", ""),
                "chat_history": result["chat_history"],
                "study_progress": study_summary,
                "thoughts": result.get("thoughts", "")
            }
            if "error" in result:
                final_result["error"] = result["error"]
            return final_result
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_response = {
                "error": f"Erro na execução do workflow: {str(e)}",
                "messages": [AIMessage(content="Desculpe, encontrei um erro ao processar sua pergunta. Por favor, tente novamente.")],
                "chat_history": recent_history
            }
            try:
                error_response["study_progress"] = await self.progress_manager.get_study_summary(self.session_id)
            except Exception as progress_error:
                print(f"Erro ao obter resumo de progresso: {progress_error}")
            return error_response
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Execução do workflow completada em {elapsed_time:.2f} segundos")
            
    async def invoke_streaming(
        self, 
        query: str, 
        student_profile: dict, 
        current_plan=None, 
        chat_history=None
    ):
        start_time = time.time()
        try:
            validated_profile = student_profile
            await self.progress_manager.sync_progress_state(self.session_id)
            current_progress = await self.progress_manager.get_study_progress(self.session_id)
            print(f"[REACT_AGENT] Progress retrieved: {current_progress}")
            if chat_history is None:
                chat_history = []
            elif not isinstance(chat_history, list):
                chat_history = list(chat_history)
            recent_history = chat_history[-10:]
            initial_state = AgentState(
                messages=[HumanMessage(content=query)],
                current_plan=current_plan if current_plan else "",
                user_profile=validated_profile,
                extracted_context={},
                next_step=None,
                iteration_count=0,
                chat_history=recent_history,
                needs_retrieval=True,
                evaluation_reason="",
                web_search_results={},
                current_progress=current_progress,
                session_id=self.session_id,
                thoughts=""
            )
            print("[REACT_AGENT] Starting streaming workflow")
            generator = self.react_stream_node(initial_state)
            async for chunk in generator:
                yield chunk
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Execução de streaming completada em {elapsed_time:.2f} segundos")
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield {
                "type": "error", 
                "content": f"Erro na execução do workflow: {str(e)}"
            }
            yield {
                "type": "complete",
                "content": "Streaming concluído com erro."
            }

# Fim do código
