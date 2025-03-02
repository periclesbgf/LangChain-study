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


class StudyProgressManager(MongoDatabaseManager):
    def __init__(self, db_name: str = "study_plans"):
        """
        Inicializa o gerenciador de progresso de estudos.

        Args:
            db_name: Nome do banco de dados MongoDB
        """
        super().__init__()
        self.collection_name = db_name

    async def sync_progress_state(self, session_id: str) -> bool:
        """
        Sincroniza o estado do progresso, garantindo consistência entre o banco de dados e o estado da aplicação.
        """
        try:
            collection = self.db[self.collection_name]
            plan = await collection.find_one({"id_sessao": session_id})

            if not plan:
                return False

            plano_execucao = plan.get("plano_execucao", [])
            modified = False

            # Validar e corrigir o progresso de cada etapa
            for step in plano_execucao:
                original_progress = step.get("progresso", 0)
                corrected_progress = min(max(float(original_progress), 0), 100)

                if original_progress != corrected_progress:
                    step["progresso"] = corrected_progress
                    modified = True

            if modified:
                # Recalcular e atualizar o progresso total
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

            # Validar e corrigir o progresso de cada etapa
            if "plano_execucao" in plan:
                for step in plan["plano_execucao"]:
                    if "progresso" not in step:
                        step["progresso"] = 0
                    else:
                        step["progresso"] = min(max(float(step["progresso"]), 0), 100)

                # Recalcular o progresso total
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
        """
        Atualiza o progresso de uma etapa específica do plano.

        Args:
            session_id: ID da sessão de estudo
            step_index: Índice da etapa no plano
            new_progress: Novo valor de progresso (0-100)

        Returns:
            bool indicando sucesso da operação
        """
        try:
            if not 0 <= new_progress <= 100:
                raise ValueError("Progresso deve estar entre 0 e 100")

            collection = self.db[self.collection_name]

            # Primeiro recupera o plano atual
            plan = await collection.find_one({"id_sessao": session_id})
            if not plan:
                return False

            # Atualiza o progresso da etapa específica
            plano_execucao = plan.get("plano_execucao", [])
            if step_index >= len(plano_execucao):
                raise ValueError(f"Índice de etapa inválido: {step_index}")

            plano_execucao[step_index]["progresso"] = new_progress

            # Calcula o progresso total
            total_steps = len(plano_execucao)
            progresso_total = sum(step["progresso"] for step in plano_execucao) / total_steps

            # Atualiza o documento
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
        """
        Recupera os detalhes de uma etapa específica do plano.

        Args:
            session_id: ID da sessão de estudo
            step_index: Índice da etapa no plano

        Returns:
            Dict contendo os detalhes da etapa ou None se não encontrado
        """
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
        """
        Marca uma etapa como concluída (100% de progresso).

        Args:
            session_id: ID da sessão de estudo
            step_index: Índice da etapa no plano

        Returns:
            bool indicando sucesso da operação
        """
        return await self.update_step_progress(session_id, step_index, 100)

    async def get_next_incomplete_step(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Encontra a próxima etapa incompleta do plano.

        Args:
            session_id: ID da sessão de estudo

        Returns:
            Dict contendo os detalhes da próxima etapa incompleta ou None se todas estiverem completas
        """
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

            return None  # Todas as etapas estão completas
        except Exception as e:
            return None

    async def get_study_summary(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Gera um resumo do progresso do plano de estudos.

        Args:
            session_id: ID da sessão de estudo

        Returns:
            Dict contendo o resumo do progresso
        """
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

def filter_chat_history(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Filtra o histórico do chat para remover conteúdo de imagem e manter apenas texto.
    """
    filtered_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            filtered_messages.append(msg)
        elif isinstance(msg, AIMessage):
            try:
                # Verifica se é uma mensagem multimodal (JSON)
                content = json.loads(msg.content)
                if isinstance(content, dict):
                    if content.get("type") == "multimodal":
                        # Remove o campo 'image' e mantém apenas o texto
                        filtered_content = {
                            "type": "multimodal",
                            "content": content["content"]
                        }
                        filtered_messages.append(AIMessage(content=filtered_content["content"]))
                    else:
                        # Mensagem JSON sem imagem
                        filtered_messages.append(msg)
                else:
                    # Não é um objeto JSON válido
                    filtered_messages.append(msg)
            except json.JSONDecodeError:
                # Mensagem regular sem JSON
                filtered_messages.append(msg)
    return filtered_messages

def format_chat_history(messages: List[BaseMessage], max_messages: int = 3) -> str:
    """
    Formata o histórico do chat filtrado para uso em prompts.
    """
    # Primeiro filtra o histórico
    filtered_messages = filter_chat_history(messages[-max_messages:])

    # Então formata as mensagens filtradas
    formatted_history = []
    for msg in filtered_messages:
        role = 'Aluno' if isinstance(msg, HumanMessage) else 'Tutor'
        content = msg.content
        if isinstance(content, str):
            formatted_history.append(f"{role}: {content}")

    return "\n".join(formatted_history)

class RetrievalTools:
    def __init__(self, qdrant_handler: QdrantHandler, student_email: str, disciplina: str, image_collection, session_id: str, state: AgentState):
        self.qdrant_handler = qdrant_handler
        self.student_email = student_email
        self.disciplina = disciplina
        self.session_id = session_id
        self.image_collection = image_collection
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.state = state

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
        """
        Transforma a pergunta para melhor recuperação de contexto baseado no tipo de busca.
        """
        # Usa a nova função de formatação do histórico
        formatted_history = format_chat_history(self.state["chat_history"], max_messages=4)

        prompt = ChatPromptTemplate.from_template(self.QUESTION_TRANSFORM_PROMPT)
        response = await self.model.ainvoke(prompt.format(
            chat_history=formatted_history,
            question=question,
        ))

        transformed_question = response.content.strip()
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
        """
        Recupera a imagem e sua descrição de forma assíncrona.
        """
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
                
            return {
                "type": "image",
                "image_bytes": processed_bytes,
                "description": results[0].page_content
            }
        except Exception as e:
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

                required_fields = ["text", "image", "table", "recommended_context"]
                if not all(field in analysis for field in required_fields):
                    raise ValueError("Missing required fields in analysis")

                return analysis

            except json.JSONDecodeError as e:
                return self._get_default_analysis()
            except ValueError as e:
                return self._get_default_analysis()

        except Exception as e:
            return self._get_default_analysis()

    def _get_default_analysis(self) -> Dict[str, Any]:
        """Retorna uma análise de relevância padrão para uso em caso de erro."""
        # Corrige a estrutura da análise para garantir compatibilidade completa
        return {
            "text": {"score": 0.5, "reason": "Default text relevance score"},
            "image": {"score": 0.5, "reason": "Default image relevance score"},
            "table": {"score": 0.5, "reason": "Default table relevance score"},
            "recommended_context": "combined"
        }

class WebSearchTools:
    def __init__(self):
        pass

    def search_youtube(self, query: str) -> str:
        """
        Realiza uma pesquisa no YouTube e retorna o link do vídeo mais relevante.
        """
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

                return response
            else:
                return "Nenhum vídeo encontrado."
        except Exception as e:
            return "Ocorreu um erro ao buscar no YouTube."

    def search_wikipedia(self, query: str) -> str:
        """
        Realiza uma pesquisa no Wikipedia e retorna o resumo da página.
        """
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

1. PENSAMENTO: Desenvolva seu raciocínio pedagógico sobre como ensinar ativamente o aluno.
   - Identifique o tópico central da etapa atual do plano de estudos
   - Analise o conhecimento prévio do aluno com base no histórico
   - Planeje uma explicação estruturada do tópico seguindo o plano de estudos
   - Formule perguntas de pensamento crítico para envolver o aluno
   - Identifique possíveis equívocos ou dificuldades de aprendizado
   - Identifique o que o aluno está tentando alcançar com a pergunta

2. AÇÃO: Escolha a ação pedagógica mais apropriada:
   - retrieval: PRIORIZE ESTA AÇÃO para buscar contexto relevante do material de estudo, incluindo imagens, tabelas ou textos que possam ajudar na explicação
   - websearch: Procurar recursos adicionais para ENRIQUECER sua explicação
   - direct_teaching: Fornecer uma explicação estruturada e aprofundada do tópico atual
   - challenge_thinking: Apresentar um problema ou cenário que desafie o entendimento do aluno
   - guided_discovery: Guiar o aluno a descobrir conceitos por meio de perguntas sequenciais
   - socratic_questioning: Fazer perguntas que estimulem o raciocínio crítico
   - suggest_activity: Propor uma atividade prática alinhada com o tópico atual
   - progress_update: Atualizar o progresso do aluno (use apenas quando identificar avanço significativo)

3. OBSERVAÇÃO: Avalie o impacto da sua ação:
   - Reflita sobre como seu ensino se alinhou com os objetivos de aprendizagem
   - Identifique ajustes necessários com base na resposta ou compreensão do aluno
   - Planeje os próximos passos para aprofundar o entendimento

## PRINCÍPIOS PEDAGÓGICOS FUNDAMENTAIS:
- ENSINE ATIVAMENTE: Não apenas responda perguntas, conduza o aluno através do conteúdo do plano de estudos
- ESTIMULE O PENSAMENTO CRÍTICO: Faça perguntas que desafiem o aluno a analisar, sintetizar e avaliar
- SIGA O PLANO: Mantenha o foco no tópico atual do plano de estudos e nos objetivos da etapa
- PERSONALIZE O ENSINO: Adapte suas explicações ao estilo de aprendizagem do aluno
- CONSTRUA SOBRE O CONHECIMENTO PRÉVIO: Faça conexões explícitas com conteúdos já estudados
- USE EXEMPLOS CONCRETOS: Forneça exemplos práticos e cenários do mundo real
- VERIFIQUE A COMPREENSÃO: Regularmente faça perguntas para confirmar o entendimento
- CORRIJA EQUÍVOCOS: Identifique e corrija gentilmente mal-entendidos
- USE MATERIAIS VISUAIS: Incorpore imagens quando disponíveis, especialmente para alunos com perfil visual

## ESTRUTURA DE RESPOSTA:
1. <pensamento>Seu raciocínio pedagógico detalhado</pensamento>
2. <ação>Estratégia de ensino escolhida e detalhes. MUITO IMPORTANTE: Se o material de estudo puder conter imagens ou tabelas, escolha 'retrieval' para buscá-las.</ação>
3. <observação>Reflexão sobre o processo de ensino</observação>
4. Explicação educacional para o aluno (clara, estruturada e adaptada ao contexto)
   * Introduza o tópico atual e conecte-o ao plano de estudos
   * Explique os conceitos fundamentais com clareza e profundidade
   * Forneça exemplos relevantes e aplicações práticas
   * Inclua perguntas de reflexão para estimular o pensamento crítico
   * Conclua com uma síntese e aponte para o próximo tópico do plano

IMPORTANTE:
- ESCOLHA 'retrieval' SEMPRE QUE JULGAR NECESSARIO integrar conteúdo visual e contextual do material de estudo
- SEU PAPEL É ENSINAR ATIVAMENTE, não apenas responder ou sugerir
- Mantenha seu raciocínio invisível para o aluno (use as tags apenas para organizar seu processo)
- Quando o aluno estiver desviando do plano, gentilmente redirecione-o ao tópico atual
- Priorize ensinar os conteúdos da etapa atual do plano de estudos, mesmo quando a pergunta do aluno for vaga
- Não espere que o aluno faça perguntas específicas; ensine proativamente o conteúdo da etapa atual
"""
    
    return ChatPromptTemplate.from_template(REACT_PROMPT)

def identify_current_step(plano_execucao: List[Dict]) -> ExecutionStep:
    """
    Identifica a etapa atual do plano de estudos com base no progresso.
    Garante que todos os dados necessários para o ensino proativo estejam disponíveis.
    """
    if not plano_execucao:
        # Fornecer uma etapa padrão quando não há plano, para que o agente possa ao menos dar orientações gerais
        return ExecutionStep(
            titulo="Introdução ao Tema",
            duracao="30 minutos",
            descricao="Introdução aos conceitos básicos do tema e exploração inicial dos objetivos de aprendizagem",
            conteudo=["Conceitos fundamentais", "Objetivos de aprendizagem", "Contextualização do tema"],
            recursos=[{"tipo": "texto", "titulo": "Material introdutório"}],
            atividade={"tipo": "reflexão", "descrição": "Reflexão inicial sobre o tema"},
            progresso=0
        )

    # Validar e normalizar o progresso de cada etapa
    for step in plano_execucao:
        if "progresso" not in step:
            step["progresso"] = 0
        else:
            # Garantir que o progresso é um número entre 0 e 100
            step["progresso"] = min(max(float(step["progresso"]), 0), 100)
        
        # Garantir que os campos obrigatórios existam
        for campo in ["conteudo", "recursos", "atividade"]:
            if campo not in step or not step[campo]:
                if campo == "conteudo":
                    step[campo] = ["Conceito principal", "Aplicações práticas"]
                elif campo == "recursos":
                    step[campo] = [{"tipo": "texto", "titulo": "Material de apoio"}]
                elif campo == "atividade":
                    step[campo] = {"tipo": "exercício", "descrição": "Praticar o conceito aprendido"}

    # Primeiro tenta encontrar uma etapa em progresso (entre 1% e 99%)
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
    
    # Se não encontrar etapa em progresso, procura a primeira etapa não iniciada
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

    # Se todas as etapas estiverem concluídas, retorna a última etapa com indicação para revisão
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

def create_react_node(retrieval_tools: RetrievalTools, web_tools: WebSearchTools, progress_manager: StudyProgressManager):
    """
    Cria um nó ReAct que implementa o processo de raciocínio, ação e observação.
    Suporta streaming de resposta.
    """
    react_prompt = create_react_prompt()
    model = ChatOpenAI(model="gpt-4o", temperature=0.2)
    model_stream = ChatOpenAI(model="gpt-4o", temperature=0.2, streaming=True)
    
    async def react_process(state: AgentState) -> AgentState:
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        chat_history = format_chat_history(state["chat_history"], max_messages=5)
        
        try:
            # Processar o plano de execução atual
            current_plan = state.get("current_plan", "{}")
            plano_execucao = []
            
            try:
                # Verificar se o plano já está em formato de dicionário (não precisa de json.loads)
                if isinstance(current_plan, dict) and "plano_execucao" in current_plan:
                    print(f"[REACT_AGENT] Plano encontrado em formato de dicionário")
                    plano_execucao = current_plan.get("plano_execucao", [])
                # Caso contrário, tenta converter de string JSON
                else:
                    print(f"[REACT_AGENT] Tentando converter plano de JSON")
                    plano_execucao = json.loads(current_plan).get("plano_execucao", [])
                
                # Verificar um progresso atual disponível
                current_progress = state.get("current_progress", {})
                if isinstance(current_progress, dict) and "plano_execucao" in current_progress:
                    print(f"[REACT_AGENT] Usando plano do current_progress")
                    plano_execucao = current_progress.get("plano_execucao", plano_execucao)
                    
                # Verificar se temos plano
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
            
            # Extrair e validar perfil do usuário
            user_profile = state.get("user_profile", {})
            estilos = user_profile.get("EstiloAprendizagem", {})
            
            # Preencher estilos faltantes
            required_styles = [("Percepcao", "não especificado"), 
                              ("Entrada", "não especificado"), 
                              ("Processamento", "não especificado"), 
                              ("Entendimento", "não especificado")]
            
            for style, default in required_styles:
                if style not in estilos:
                    estilos[style] = default
            
            # Verificar se é início de conversa ou mensagem vaga para iniciar ensino proativo
            is_first_message = len(state["chat_history"]) == 0
            is_vague_message = len(latest_question.split()) < 5 or any(term in latest_question.lower() 
                                                                      for term in ["oi", "olá", "começar", "iniciar", 
                                                                                   "continuar", "seguir", "ok", "bom", 
                                                                                   "entendi", "compreendi"])
            
            # Enriquecer a mensagem do aluno para estimular ensino proativo quando necessário
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
            
            # Preparar parâmetros do prompt
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
            
            # Invocar o modelo ReAct
            reaction = await model.ainvoke(react_prompt.format(**params))
            react_content = reaction.content
            
            # Extrair pensamento, ação e observação
            thoughts = extract_section(react_content, "pensamento")
            action = extract_section(react_content, "ação")
            observation = extract_section(react_content, "observação")
            
            # Imprimir pensamento completo para depuração
            print("\n[REACT_AGENT] ==== INÍCIO DO PENSAMENTO ====")
            print(thoughts)
            print("[REACT_AGENT] ==== FIM DO PENSAMENTO ====\n")
            
            print(f"[REACT_AGENT] Ação escolhida: {action[:100]}...")
            
            print("\n[REACT_AGENT] ==== INÍCIO DA OBSERVAÇÃO ====")
            print(observation)
            print("[REACT_AGENT] ==== FIM DA OBSERVAÇÃO ====\n")
            
            # Remover as seções de ReAct para obter a resposta final
            final_response = remove_react_tags(react_content)
            
            # Verificar se a resposta inclui explicação do conteúdo da etapa atual
            if current_step.titulo.lower() not in final_response.lower() and len(final_response) < 1000:
                print(f"[REACT_AGENT] Resposta não contém menção ao tópico atual - adicionando contexto")
                final_response = (
                    f"Vamos continuar explorando o tópico '{current_step.titulo}'.\n\n{final_response}"
                )
            
            # Processar a ação escolhida
            action_result = await process_action(
                action, 
                state, 
                latest_question,  # Usamos a pergunta original aqui, não a enriquecida
                retrieval_tools, 
                web_tools, 
                progress_manager
            )
            
            # Formatar resposta com possível conteúdo de imagem
            response_content = format_response_with_image(final_response, action_result)
            
            # Atualizar o estado com formato padrão do LangChain
            new_state = state.copy()
            
            # Criar mensagem do usuário em formato padrão
            user_message = HumanMessage(content=latest_question)
            
            # Criar mensagem do assistente em formato padrão
            if isinstance(response_content, dict) and "type" in response_content and response_content["type"] == "multimodal":
                # Para exibição usamos o conteúdo completo serializado
                response = AIMessage(content=json.dumps(response_content))
                
                # Para compatibilidade com o armazenamento MongoDB, também serializar para histórico
                # Esta abordagem corresponde ao formato esperado pelo chat_controller._save_chat_history()
                history_message = AIMessage(content=json.dumps(response_content))
            else:
                # Para texto simples, usar diretamente
                response = AIMessage(content=response_content)
                history_message = response
            
            # Atualizar messages (para exibição) e chat_history (para contexto)
            new_state["messages"] = list(state["messages"]) + [response]
            new_state["chat_history"] = list(state["chat_history"]) + [
                user_message,
                history_message
            ]
            
            print(f"[REACT_AGENT] Adicionado ao histórico: Pergunta ({type(user_message).__name__}) e Resposta ({type(history_message).__name__})")
            print(f"[REACT_AGENT] Conteúdo da pergunta: {user_message.content[:50]}...")
            print(f"[REACT_AGENT] Conteúdo da resposta: {history_message.content[:50]}...")
            new_state["thoughts"] = thoughts
            
            # Atualizar contexto se disponível no resultado da ação
            if "context" in action_result:
                new_state["extracted_context"] = action_result["context"]
            
            # Atualizar progresso se disponível no resultado da ação
            if "progress" in action_result:
                new_state["current_progress"] = action_result["progress"]
            
            # Se a interação for substantiva, automaticamente atualize um pouco o progresso
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
    
    # Versão com streaming
    async def react_process_streaming(state: AgentState):
        """Versão do processo ReAct que suporta streaming de resposta"""
        start_time = time.time()
        step_time = time.time()
        
        print(f"\n[REACT_AGENT] Starting streaming process at {datetime.now().strftime('%H:%M:%S')}")
        
        latest_question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
        chat_history = format_chat_history(state["chat_history"], max_messages=5)
        
        print(f"[REACT_AGENT] Processing question: '{latest_question[:50]}...'")
        print(f"[REACT_AGENT] Chat history has {len(state['chat_history'])} messages")
        
        try:
            # Yield mensagem de processamento inicial
            yield {"type": "processing", "content": "Analisando sua pergunta..."}
            print(f"[REACT_AGENT] Sent initial processing message ({time.time() - step_time:.2f}s)")
            
            # Processar o plano de execução atual
            current_plan = state.get("current_plan", "{}")
            plano_execucao = []
            
            try:
                # Verificar se o plano já está em formato de dicionário (não precisa de json.loads)
                if isinstance(current_plan, dict) and "plano_execucao" in current_plan:
                    print(f"[REACT_AGENT] Plano encontrado em formato de dicionário")
                    plano_execucao = current_plan.get("plano_execucao", [])
                # Caso contrário, tenta converter de string JSON
                else:
                    print(f"[REACT_AGENT] Tentando converter plano de JSON")
                    plano_execucao = json.loads(current_plan).get("plano_execucao", [])
                
                # Verificar um progresso atual disponível
                current_progress = state.get("current_progress", {})
                if isinstance(current_progress, dict) and "plano_execucao" in current_progress:
                    print(f"[REACT_AGENT] Usando plano do current_progress")
                    plano_execucao = current_progress.get("plano_execucao", plano_execucao)
                    
                # Verificar se temos plano
                print(f"[REACT_AGENT] Plano de execução encontrado com {len(plano_execucao)} etapas")
                current_step = identify_current_step(plano_execucao)
                
            except Exception as plan_error:
                print(f"[REACT_AGENT] Erro ao processar plano: {str(plan_error)}")
                current_step = ExecutionStep(
                    titulo="Etapa não encontrada",
                    duracao="N/A",
                    descricao="Informações do plano de execução não disponíveis",
                    conteudo=[],
                    recursos=[],
                    atividade={},
                    progresso=0
                )
            
            # Extrair e validar perfil do usuário
            user_profile = state.get("user_profile", {})
            estilos = user_profile.get("EstiloAprendizagem", {})
            
            # Preencher estilos faltantes
            required_styles = [("Percepcao", "não especificado"), 
                              ("Entrada", "não especificado"), 
                              ("Processamento", "não especificado"), 
                              ("Entendimento", "não especificado")]
            
            for style, default in required_styles:
                if style not in estilos:
                    estilos[style] = default
            
            # Verificar se é início de conversa ou mensagem vaga para iniciar ensino proativo
            is_first_message = len(state["chat_history"]) == 0
            is_vague_message = len(latest_question.split()) < 5 or any(term in latest_question.lower() 
                                                                      for term in ["oi", "olá", "começar", "iniciar", 
                                                                                   "continuar", "seguir", "ok", "bom", 
                                                                                   "entendi", "compreendi"])
            
            # Enriquecer a mensagem do aluno para estimular ensino proativo quando necessário
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
            
            # Preparar parâmetros do prompt
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
            
            # Yield mensagem de processamento sobre pensamento
            yield {"type": "processing", "content": "Elaborando uma resposta..."}
            step_time = time.time()
            print(f"[REACT_AGENT] Starting model streaming")
            
            # Iniciar stream - o método astream já retorna um async generator, não precisa await
            stream = model_stream.astream(react_prompt.format(**params))
            
            # Coleta completa para processamento
            full_reaction = ""
            chunk_count = 0
            async for chunk in stream:
                if chunk.content:
                    full_reaction += chunk.content
                    chunk_count += 1
            
            print(f"[REACT_AGENT] Collected {chunk_count} chunks, total length: {len(full_reaction)} chars ({time.time() - step_time:.2f}s)")
            step_time = time.time()
            
            # Extrair pensamento, ação e observação do conteúdo completo
            thoughts = extract_section(full_reaction, "pensamento")
            action = extract_section(full_reaction, "ação")
            observation = extract_section(full_reaction, "observação")
            
            # Imprimir pensamento completo para depuração
            print("\n[REACT_AGENT] ==== INÍCIO DO PENSAMENTO (STREAMING) ====")
            print(thoughts)
            print("[REACT_AGENT] ==== FIM DO PENSAMENTO (STREAMING) ====\n")
            
            print(f"[REACT_AGENT] Ação escolhida (streaming): {action[:100]}...")
            
            print("\n[REACT_AGENT] ==== INÍCIO DA OBSERVAÇÃO (STREAMING) ====")
            print(observation)
            print("[REACT_AGENT] ==== FIM DA OBSERVAÇÃO (STREAMING) ====\n")
            print(f"[REACT_AGENT] Extraction completed in {time.time() - step_time:.2f}s")
            step_time = time.time()
            
            # Yield mensagem sobre o processamento da ação
            yield {"type": "processing", "content": "Executando ações necessárias..."}
            
            # Processar a ação escolhida
            print(f"[REACT_AGENT] Processing action: {action[:30]}...")
            step_time = time.time()
            action_result = await process_action(
                action, 
                state, 
                latest_question, 
                retrieval_tools, 
                web_tools, 
                progress_manager
            )
            print(f"[REACT_AGENT] Action processing completed: {action_result.get('type', 'unknown')} ({time.time() - step_time:.2f}s)")
            
            # Remover as seções de ReAct para obter a resposta final
            step_time = time.time()
            final_response = remove_react_tags(full_reaction)
            print(f"[REACT_AGENT] Removed React tags, final response length: {len(final_response)} chars ({time.time() - step_time:.2f}s)")
            step_time = time.time()
            
            # Verificar se a resposta inclui explicação do conteúdo da etapa atual
            if current_step.titulo.lower() not in final_response.lower() and len(final_response) < 1000:
                print(f"[REACT_AGENT] Resposta não contém menção ao tópico atual - adicionando contexto")
                final_response = (
                    f"Vamos continuar explorando o tópico '{current_step.titulo}'.\n\n{final_response}"
                )
            
            # Formatar resposta com possível conteúdo de imagem
            response_content = format_response_with_image(final_response, action_result)
            has_image = isinstance(response_content, dict) and response_content.get("type") == "multimodal"
            print(f"[REACT_AGENT] Response formatting completed, has image: {has_image} ({time.time() - step_time:.2f}s)")
            step_time = time.time()
            
            # Se a interação for substantiva, automaticamente atualize um pouco o progresso
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
            
            # Stream a resposta final formatada em chunks
            if isinstance(response_content, dict) and response_content.get("type") == "multimodal":
                # Se for resposta com imagem
                text_content = response_content["content"]
                chunk_size = 30  # Tamanho de cada chunk
                text_chunks = [text_content[i:i+chunk_size] for i in range(0, len(text_content), chunk_size)]
                print(f"[REACT_AGENT] Streaming multimodal response with {len(text_chunks)} text chunks")
                
                # Enviar pedaços de texto
                chunks_sent = 0
                for chunk in text_chunks:
                    yield {"type": "chunk", "content": chunk}
                    chunks_sent += 1
                    if chunks_sent % 5 == 0:  # Log a cada 5 chunks
                        print(f"[REACT_AGENT] Sent {chunks_sent}/{len(text_chunks)} text chunks")
                
                # Enviar a imagem como chunk final
                print(f"[REACT_AGENT] Sending image chunk")
                yield {
                    "type": "image", 
                    "content": response_content["content"],
                    "image": response_content["image"]
                }
                print(f"[REACT_AGENT] Image chunk sent")
                
                # Preparar mensagens para o histórico de chat no formato correto
                # Para visualização, usamos o JSON completo
                response = AIMessage(content=json.dumps(response_content))
                
                # Para o histórico de chat, usamos também o JSON completo para compatibilidade com MongoDB
                # Esta abordagem corresponde ao formato esperado pelo chat_controller._save_chat_history()
                history_message = AIMessage(content=json.dumps(response_content))
                print(f"[REACT_AGENT] Created multimodal history message (serialized as JSON)")
            else:
                # Se for apenas texto
                text_content = response_content
                chunk_size = 30  # Tamanho de cada chunk
                text_chunks = [text_content[i:i+chunk_size] for i in range(0, len(text_content), chunk_size)]
                print(f"[REACT_AGENT] Streaming text-only response with {len(text_chunks)} chunks")
                
                chunks_sent = 0
                for chunk in text_chunks:
                    yield {"type": "chunk", "content": chunk}
                    chunks_sent += 1
                    if chunks_sent % 10 == 0:  # Log a cada 10 chunks
                        print(f"[REACT_AGENT] Sent {chunks_sent}/{len(text_chunks)} text chunks")
                
                # Criar mensagem de usuário e assistente para o histórico
                # Usamos o mesmo formato para ambos response e history_message para consistência
                response = AIMessage(content=text_content)
                history_message = AIMessage(content=text_content)
                print(f"[REACT_AGENT] Finished streaming text response ({time.time() - step_time:.2f}s)")
            
            # Yield mensagem de conclusão
            total_time = time.time() - start_time
            yield {"type": "complete", "content": f"Resposta completa em {total_time:.2f}s."}
            print(f"[REACT_AGENT] Full processing completed in {total_time:.2f}s")
            
            # Atualizar estado para uso posterior com formato padrão do LangChain
            step_time = time.time()
            
            # Criar mensagem do usuário em formato padrão
            user_message = HumanMessage(content=latest_question)
            
            # Atualizar state com as mensagens formatadas corretamente
            state["messages"] = list(state["messages"]) + [response]
            state["chat_history"] = list(state["chat_history"]) + [
                user_message,
                history_message
            ]
            state["thoughts"] = thoughts
            
            print(f"[REACT_AGENT] State updated, saved {len(thoughts)} chars of thoughts")
            print(f"[REACT_AGENT] Added to chat_history: user message ({type(user_message).__name__}) and assistant message ({type(history_message).__name__})")
            print(f"[REACT_AGENT] User message: {user_message.content[:50]}...")
            print(f"[REACT_AGENT] Assistant message: {history_message.content[:50]}...")
            
            # Atualizar contexto se disponível no resultado da ação
            if "context" in action_result:
                state["extracted_context"] = action_result["context"]
            
            # Atualizar progresso se disponível no resultado da ação
            if "progress" in action_result:
                state["current_progress"] = action_result["progress"]
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            # Enviar mensagem de erro em caso de falha
            yield {"type": "error", "content": f"Desculpe, encontrei um erro ao processar sua pergunta: {str(e)}"}
            
            # Atualizar história com mensagem de erro
            error_message = "Desculpe, encontrei um erro ao processar sua pergunta. Por favor, tente novamente."
            response = AIMessage(content=error_message)
            
            state["messages"] = list(state["messages"]) + [response]
            state["chat_history"] = list(state["chat_history"]) + [
                HumanMessage(content=latest_question),
                response
            ]
            state["error"] = str(e)
    
    # Retorna ambas as funções para uso conforme necessário
    return react_process, react_process_streaming

def extract_section(text: str, section_name: str) -> str:
    """Extrai a seção especificada do texto ReAct."""
    import re
    pattern = f"<{section_name}>(.*?)</{section_name}>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return ""

def remove_react_tags(text: str) -> str:
    """Remove todas as tags ReAct do texto."""
    import re
    # Remove pensamento, ação e observação
    text = re.sub(r'<pensamento>.*?</pensamento>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ação>.*?</ação>', '', text, flags=re.DOTALL)
    text = re.sub(r'<observação>.*?</observação>', '', text, flags=re.DOTALL)
    text = re.sub(r'<action>.*?</action>', '', text, flags=re.DOTALL)
    text = re.sub(r'<observation>.*?</observation>', '', text, flags=re.DOTALL)
    
    # Limpar linhas em branco extras
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

async def process_action(action: str, state: AgentState, question: str, 
                        retrieval_tools: RetrievalTools, web_tools: WebSearchTools,
                        progress_manager: StudyProgressManager) -> Dict[str, Any]:
    """Processa a ação escolhida pelo modelo ReAct."""
    action_lower = action.lower()
    
    # Log para debug
    print(f"[REACT_ACTION] Processing action: '{action_lower[:50]}...'")
    
    # Ação de busca em material de estudo (o caso mais importante para correção)
    if "retrieval" in action_lower:
        return await process_retrieval_action(question, retrieval_tools, state)
    
    # Ação de busca na web
    elif "websearch" in action_lower:
        return await process_websearch_action(question, web_tools)
    
    # Ação de atualização de progresso
    elif "progress_update" in action_lower:
        return await process_progress_action(action, state, progress_manager)
    
    # Novas ações pedagógicas proativas
    elif "direct_teaching" in action_lower:
        # Para ensino direto, vamos verificar se há contexto já carregado em state
        if not state.get("extracted_context"):
            # Se não houver contexto, podemos tentar recuperar
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
    
    # Ações originais mantidas para compatibilidade
    elif "direct_response" in action_lower:
        return {"type": "direct", "content": "Resposta direta fornecida"}
    
    elif "analyze_response" in action_lower:
        return {"type": "analysis", "content": "Análise da resposta realizada"}
    
    elif "suggest_activity" in action_lower:
        return {"type": "activity", "content": "Nova atividade sugerida"}
    
    # Ação padrão - tentar fazer retrieval para garantir
    print(f"[REACT_ACTION] Using default action, attempting retrieval first")
    try:
        context_result = await process_retrieval_action(question, retrieval_tools, state)
        if context_result and "context" in context_result:
            return context_result
    except Exception as e:
        print(f"[REACT_ACTION] Error in default retrieval: {e}")
        
    return {"type": "direct_teaching", "content": "Explicação estruturada fornecida"}

async def process_retrieval_action(question: str, tools: RetrievalTools, state: AgentState) -> Dict[str, Any]:
    """Processa a ação de busca em material de estudo."""
    start_time = time.time()
    try:
        print(f"[REACT_ACTION] Starting retrieval for: '{question[:50]}...'")
        
        # Usar as ferramentas de recuperação para buscar contexto
        tools.state = state
        context_results = await tools.parallel_context_retrieval(question)
        
        # Logging de resultados detalhados
        has_text = bool(context_results.get("text", ""))
        has_image = isinstance(context_results.get("image", {}), dict) and context_results["image"].get("type") == "image"
        has_table = isinstance(context_results.get("table", {}), dict) and context_results["table"].get("content")
        
        # Detalhamento do conteúdo recuperado para debug
        if has_text:
            text_preview = context_results["text"][:100] + "..." if len(context_results["text"]) > 100 else context_results["text"]
            print(f"[REACT_ACTION] Text context preview: {text_preview}")
        
        if has_image:
            image_info = context_results["image"]
            has_image_bytes = bool(image_info.get("image_bytes"))
            desc_preview = image_info.get("description", "")[:100] + "..." if len(image_info.get("description", "")) > 100 else image_info.get("description", "")
            print(f"[REACT_ACTION] Image found: has_bytes={has_image_bytes}, type={image_info.get('type')}")
            print(f"[REACT_ACTION] Image description: {desc_preview}")
        
        if has_table:
            table_preview = str(context_results["table"].get("content", ""))[:100] + "..." if len(str(context_results["table"].get("content", ""))) > 100 else str(context_results["table"].get("content", ""))
            print(f"[REACT_ACTION] Table context preview: {table_preview}")
        
        # Análise de relevância
        if "relevance_analysis" in context_results:
            print(f"[REACT_ACTION] Relevance analysis: {context_results['relevance_analysis']}")
            # Garantir que a análise de relevância tenha todos os campos necessários
            analysis = context_results['relevance_analysis']
            if not all(field in analysis for field in ["text", "image", "table", "recommended_context"]):
                print(f"[REACT_ACTION] Fixing incomplete relevance analysis")
                analysis = tools._get_default_analysis()
                context_results['relevance_analysis'] = analysis
        
        print(f"[REACT_ACTION] Retrieval completed in {time.time() - start_time:.2f}s")
        print(f"[REACT_ACTION] Found: text={has_text}, image={has_image}, table={has_table}")
        
        return {
            "type": "retrieval",
            "content": "Contexto recuperado com sucesso",
            "context": context_results
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[REACT_ACTION] Retrieval error: {str(e)} ({time.time() - start_time:.2f}s)")
        return {
            "type": "error",
            "content": f"Erro na recuperação de contexto: {str(e)}"
        }

async def process_websearch_action(question: str, web_tools: WebSearchTools) -> Dict[str, Any]:
    """Processa a ação de busca na web."""
    start_time = time.time()
    try:
        print(f"[REACT_ACTION] Starting web search for: '{question[:50]}...'")
        
        # Otimizar a query manualmente já que não temos o modelo disponível aqui
        wiki_time = time.time()
        wiki_result = web_tools.search_wikipedia(question)
        print(f"[REACT_ACTION] Wikipedia search completed in {time.time() - wiki_time:.2f}s")
        
        yt_time = time.time()
        youtube_result = web_tools.search_youtube(question)
        print(f"[REACT_ACTION] YouTube search completed in {time.time() - yt_time:.2f}s")
        
        # Formatar recursos para retorno
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
    """Processa a ação de atualização de progresso."""
    start_time = time.time()
    try:
        print(f"[REACT_ACTION] Processing progress update")
        
        # Tentar extrair o valor de progresso da ação (formato: "progress_update: 10")
        import re
        progress_match = re.search(r'progress_update.*?(\d+)', action)
        progress_value = int(progress_match.group(1)) if progress_match else 10
        print(f"[REACT_ACTION] Progress value extracted: {progress_value}%")
        
        session_id = state["session_id"]
        
        # Obter informações atualizadas de progresso
        study_progress = await progress_manager.get_study_progress(session_id)
        if not study_progress:
            return {"type": "error", "content": "Dados de progresso não encontrados"}
        
        # Identificar a etapa atual
        plano_execucao = study_progress['plano_execucao']
        current_step = None
        step_index = 0
        
        # Encontra a primeira etapa não concluída
        for idx, step in enumerate(plano_execucao):
            if step['progresso'] < 100:
                current_step = step
                step_index = idx
                break
        
        if not current_step:
            return {"type": "info", "content": "Todas as etapas completadas"}
        
        # Calcular novo progresso
        current_progress = current_step['progresso']
        new_progress = min(current_progress + progress_value, 100)
        
        # Atualizar o progresso
        update_success = await progress_manager.update_step_progress(
            session_id,
            step_index,
            new_progress
        )
        
        if update_success:
            # Obter o resumo atualizado do estudo
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
    """Formata a resposta final, possivelmente incluindo uma imagem."""
    # Log detalhado para debug
    print(f"[REACT_AGENT] Formatting response with action_result type: {action_result.get('type', 'unknown')}")
    
    # Verificar se temos uma imagem no contexto recuperado
    has_image = False
    image_bytes = None
    
    if action_result.get("type") == "retrieval" and "context" in action_result:
        context = action_result["context"]
        image_data = context.get("image", {})
        
        # Log detalhado da estrutura da imagem
        print(f"[REACT_AGENT] Image data keys: {image_data.keys() if isinstance(image_data, dict) else 'not a dict'}")
        
        if (isinstance(image_data, dict) and 
            image_data.get("type") == "image" and 
            image_data.get("image_bytes")):
            
            has_image = True
            image_bytes = image_data["image_bytes"]
            print(f"[REACT_AGENT] Found valid image in context, size: {len(image_bytes) if image_bytes else 'unknown'} bytes")
            
            # Verificar se a imagem está em análise de relevância
            if "relevance_analysis" in context:
                image_score = context["relevance_analysis"].get("image", {}).get("score", 0)
                print(f"[REACT_AGENT] Image relevance score: {image_score}")
                # Se score for muito baixo, talvez não valha a pena incluir
                if image_score < 0.2:
                    print(f"[REACT_AGENT] Image relevance too low, not including image")
                    has_image = False
    
    # Formatar a resposta com imagem se necessário
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
        
        self.web_tools = WebSearchTools()
        
        # Obter as funções de processamento primeiro
        self.react_node, self.react_stream_node = create_react_node(
            self.retrieval_tools, 
            self.web_tools,
            self.progress_manager
        )
        
        # Depois criar o workflow usando o react_node
        self.workflow = self.create_workflow()
        
    def create_workflow(self) -> Graph:
        workflow = Graph()
        
        # Adicionar o nó ReAct único
        workflow.add_node("react", self.react_node)
        
        # Configurar o fluxo simples
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
            # Validar perfil do usuário
            validated_profile = student_profile
            
            # Sincronizar e recuperar progresso atual
            await self.progress_manager.sync_progress_state(self.session_id)
            current_progress = await self.progress_manager.get_study_progress(self.session_id)
            
            # Preparar histórico de chat
            if chat_history is None:
                chat_history = []
            elif not isinstance(chat_history, list):
                chat_history = list(chat_history)
            
            recent_history = chat_history[-10:]
            
            # Preparar estado inicial
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
            
            # Executar o workflow
            result = await self.workflow.ainvoke(initial_state)
            
            # Recuperar o resumo atualizado do estudo
            study_summary = await self.progress_manager.get_study_summary(self.session_id)
            
            # Log the chat history for debugging
            chat_history = result["chat_history"]
            print(f"[REACT_AGENT] Result contains {len(chat_history)} chat history messages")
            for i, msg in enumerate(chat_history):
                msg_type = type(msg).__name__
                content_preview = str(msg.content)[:30] + "..." if len(str(msg.content)) > 30 else str(msg.content)
                print(f"[REACT_AGENT] Result chat_history[{i}]: {msg_type} - {content_preview}")
            
            # Preparar o resultado final
            final_result = {
                "messages": result["messages"],
                "final_plan": result.get("current_plan", ""),
                "chat_history": result["chat_history"],
                "study_progress": study_summary,
                "thoughts": result.get("thoughts", "")
            }
            
            # Adicionar informações de debug se necessário
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
            
            # Tenta adicionar o progresso mesmo em caso de erro
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
        """Versão streaming do invoke que retorna um gerador assíncrono"""
        start_time = time.time()
        
        try:
            # Validar perfil do usuário
            validated_profile = student_profile
            
            # Sincronizar e recuperar progresso atual
            await self.progress_manager.sync_progress_state(self.session_id)
            current_progress = await self.progress_manager.get_study_progress(self.session_id)
            print(f"[REACT_AGENT] Progress retrieved: {current_progress}")
            
            # Preparar histórico de chat
            if chat_history is None:
                chat_history = []
            elif not isinstance(chat_history, list):
                chat_history = list(chat_history)
            
            recent_history = chat_history[-10:]
            
            # Preparar estado inicial
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
            print(f"[REACT_AGENT] Initial state: {initial_state}")
            
            # Executar o nó de streaming diretamente (sem o workflow)
            generator = self.react_stream_node(initial_state)
            
            # Passar os chunks diretamente
            async for chunk in generator:
                yield chunk
                
            # Registrar o tempo de conclusão
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Execução de streaming completada em {elapsed_time:.2f} segundos")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            # Retornar erro como chunk
            yield {
                "type": "error", 
                "content": f"Erro na execução do workflow: {str(e)}"
            }