from typing import TypedDict, List, Sequence, Dict, Any, Optional
from datetime import datetime, timezone
from langgraph.graph import END, Graph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import motor.motor_asyncio
from pymongo import errors
import json
import time
from tavily import TavilyClient

from agent.tools import DatabaseUpdateTool
from logg import logger
from utils import TAVILY_API_KEY

class ActivityResource(BaseModel):
    tipo: str
    descricao: str
    url: str = Field(default="")

class Activity(BaseModel):
    descricao: str
    tipo: str
    formato: str

class PlanStep(BaseModel):
    titulo: str
    duracao: str
    descricao: str
    conteudo: List[str]
    recursos: List[ActivityResource]
    atividade: Activity
    progresso: int

class StudyPlan(BaseModel):
    plano_execucao: List[PlanStep]
    duracao_total: str
    progresso_total: int
    created_at: datetime
    horario_sugerido: Dict[str, str]

class PlanningState(TypedDict):
    messages: Sequence[BaseMessage]
    study_plan: dict
    user_profile: dict
    next_step: str | None
    review_feedback: str | None
    id_sessao: str
    update_status: bool | None
    scheduled_time: Dict[str, str] | None
    revision_count: int
    web_resources: List[Dict[str, str]]

class SessionPlanWorkflow:
    def __init__(self, db_tool: DatabaseUpdateTool):
        self.db_tool = db_tool
        self.workflow = self.create_workflow()

    def create_workflow(self) -> Graph:
        #print("[DEBUG]: Creating workflow")
        workflow = Graph()
        
        # Definir todos os nós
        workflow.add_node("initial_web_search", self.create_initial_web_search_node())
        workflow.add_node("generate_plan", self.create_planning_node())
        workflow.add_node("review_plan", self.create_review_node())
        workflow.add_node("update_plan", self.create_update_plan_node())
        workflow.add_node("update_plan_links", self.create_update_links_node())
        workflow.add_node("update_database", self.create_database_node())
        
        # Configurar o fluxo principal
        workflow.add_edge("initial_web_search", "generate_plan")
        workflow.add_edge("generate_plan", "review_plan")
        
        # Conditional edge com verificação de limite de ciclos e rotas específicas
        workflow.add_conditional_edges(
            "review_plan",
            self.route_after_review,
            {
                "needs_revision": "update_plan",
                "update_links": "update_plan_links",
                "approved": "update_database",
                "max_revisions_reached": "update_database"  # Se atingir o limite, vai direto para o banco
            }
        )
        
        # Ciclos de feedback
        workflow.add_edge("update_plan", "review_plan")  # Ciclo para revisar novamente após atualização geral
        workflow.add_edge("update_plan_links", "review_plan")  # Ciclo para revisar novamente após atualização de links
        workflow.add_edge("update_database", END)
        
        # Definir o ponto de entrada
        workflow.set_entry_point("initial_web_search")
        
        compiled_workflow = workflow.compile()
        #print("[DEBUG]: Workflow compiled")
        return compiled_workflow
        
    def route_after_review(self, state: PlanningState) -> str:
        # Verificar o contador de revisões (máximo de 3 ciclos)
        revision_count = state.get("revision_count", 0)
        MAX_REVISIONS = 2
        
        if revision_count >= MAX_REVISIONS:
            logger.warning(f"[PLAN_AGENT]: Máximo de {MAX_REVISIONS} revisões atingido, forçando aprovação")
            return "max_revisions_reached"
        
        # Processar o feedback do revisor
        review_feedback = state.get("review_feedback", {})
        status = review_feedback.get("status", "")
        
        # Verificar se o revisor indicou problemas específicos com links
        links_problem = review_feedback.get("links_problem", False)
        
        # Determinar a rota com base na indicação explícita do revisor
        if status == "needs_revision" and links_problem:
            logger.info(f"[PLAN_AGENT]: Revisor indicou problemas com links, redirecionando para ajuste específico de links")
            return "update_links"
        elif status == "needs_revision":
            logger.info(f"[PLAN_AGENT]: Plano precisa de revisão geral, redirecionando para atualização (revisão #{revision_count+1})")
            return "needs_revision"
        else:
            logger.info(f"[PLAN_AGENT]: Plano aprovado, prosseguindo para o banco de dados")
            return "approved"

    def create_planning_node(self):
        PLANNING_PROMPT = """Você é um especialista em planejamento educacional que cria planos de estudo detalhados e personalizados.
        O plano deve conter instruções claras e específicas para guiar uma LLM a ensinar um tópico específico.
        A LLM vai seguir o plano para criar uma sessão de estudo eficaz para um aluno.

        Tema da Sessão: {topic}
        Objetivo Geral da Disciplina: {objetivo_geral}

        Estilo de Aprendizagem do Aluno:
        {learning_style}

        Informações de Horário:
        Data do Encontro: {encounter_date}
        Horário do Encontro: {encounter_start} até {encounter_end}
        Preferência de Horário do Aluno: {study_preference}

        Recursos da Web disponíveis para o tópico (inclua pelo menos 3 destes recursos no plano):
        {web_resources}

        Crie um plano de estudo detalhado que:
        1. Seja adaptado ao estilo de aprendizagem específico do aluno
        2. Divida o conteúdo em etapas claras e gerenciáveis
        3. Inclua atividades e recursos apropriados ao estilo do aluno
        4. Foque em ensinar o tópico de forma eficaz e envolvente
        5. Seja específico para o tema da programação, incluindo exemplos práticos e exercícios de código
        6. SEMPRE FOQUE EM EXPLICAR O ASSUNTO ANTES DE FAZER QUALQUER ATIVIDADE
        7. IMPORTANTE: Agende a sessão de estudo para o mesmo dia do encontro ou para um dia anterior ao encontro
        8. Na descrição, explique detalhadamente as tecnologias, por exemplo. 
        9. Utilize o objetivo geral da disciplina para saber o que é a disciplina, mas foque no tema da sessão.
        10. UTILIZE OS RECURSOS DA WEB FORNECIDOS - inclua pelo menos 3 deles no plano, em seções relevantes

        Regras ESTRITAS para Agendamento:
        1. SEMPRE use a data do encontro ({encounter_date}) como referência
        2. A sessão DEVE ocorrer no mesmo dia do encontro, NUNCA depois
        3. Se agendar para o mesmo dia, a sessão DEVE terminar pelo menos 1 hora antes do encontro ({encounter_start})
        5. Respeite a preferência de horário do aluno ({study_preference}) quando possível
        6. Garanta pelo menos 1 hora de intervalo entre o fim da sessão e o início do encontro
        7. A duração da sessão deve ser compatível com o plano de estudo

        O plano deve seguir EXATAMENTE esta estrutura JSON:

            "plano_execucao": [
                    "titulo": "Título da seção"
                    "duracao": "XX minutos",
                    "descricao": "Descrição detalhada sobre qual assunto abordar",
                    "conteudo": ["Item 1", "Item 2"],
                    "recursos": [
                            "tipo": "Tipo do recurso",
                            "descricao": "Descrição do recurso",
                            "url": "URL opcional"
                    ],
                    "atividade":
                        "descricao": "Descrição da atividade",
                        "tipo": "Tipo da atividade",
                        "formato": "Formato da atividade"

                    "progresso": 0
            ],
            "duracao_total": "60 minutos",
            "progresso_total": 0,
            "horario_sugerido":
                "data": "YYYY-MM-DD",
                "inicio": "HH:MM",
                "fim": "HH:MM",
                "justificativa": "Explique a escolha do horário em relação à data do encontro"

        Adapte as atividades e recursos ao estilo de aprendizagem:
        - Para aluno Visual: Priorize recursos visuais como videos ou busca no banco de dados vetorial de imagens
        - Para aluno Verbal: Foque em explicações escritas e discussões
        - Para aluno Ativo: Inclua exercícios práticos e experimentação
        - Para aluno Reflexivo: Adicione momentos de análise e reflexão
        - Para aluno Sequencial: Organize em passos pequenos e conectados
        - Para aluno Global: Forneça visão geral e conexões com outros temas

        ATENÇÃO: 
        - TODOS OS LINKS DEVEM SER ACESSÍVEIS E RELEVANTES AO TEMA DA SESSÃO. NÃO USE LINKS DE EXEMPLOS GENÉRICOS.
        - INCLUA PELO MENOS 3 DOS RECURSOS WEB FORNECIDOS, selecionando os mais relevantes.

        NOTA: LIMITE o TITULO e DESCRICAO a 100 caracteres.

        Retorne APENAS o JSON do plano, sem explicações adicionais."""

        prompt = ChatPromptTemplate.from_template(PLANNING_PROMPT)
        model = ChatOpenAI(model="gpt-4o")

        def generate_plan(state: PlanningState) -> PlanningState:
            #print("[DEBUG]: Generating plan with state", state)

            topic = state["messages"][0].content if state["messages"] else "Tema não especificado"
            learning_style = state["user_profile"].get("EstiloAprendizagem", {})

            # Extrair informações de horário de forma mais estruturada
            encounter_info = state["user_profile"].get("horarios", {}).get("encontro", {})
            encounter_date = encounter_info.get("data", "Data não especificada")
            encounter_start = encounter_info.get("inicio", "Horário não especificado")
            encounter_end = encounter_info.get("fim", "Horário não especificado")
            study_preference = state["user_profile"].get("horarios", {}).get("preferencia")
            objetivo_geral = state["objetivo_geral"]
            
            # Formatar recursos web para o prompt
            web_resources_list = state.get("web_resources", [])
            formatted_web_resources = ""
            
            for i, resource in enumerate(web_resources_list):
                resource_info = f"Recurso {i+1}:\n"
                resource_info += f"- Título: {resource.get('title', 'Sem título')}\n"
                resource_info += f"- URL: {resource.get('url', 'Sem URL')}\n"
                resource_info += f"- Descrição: {resource.get('content', 'Sem descrição')[:150]}...\n\n"
                formatted_web_resources += resource_info
            
            if not formatted_web_resources:
                formatted_web_resources = "Nenhum recurso web disponível."
            
            logger.info(f"[PLAN_AGENT]: Gerando plano com {len(web_resources_list)} recursos web")
            
            response = model.invoke(prompt.format(
                learning_style=learning_style,
                topic=topic,
                encounter_date=encounter_date,
                encounter_start=encounter_start,
                encounter_end=encounter_end,
                study_preference=study_preference,
                objetivo_geral=objetivo_geral,
                web_resources=formatted_web_resources
            ))

            print("[GENERATE]: Model response", response)

            cleaned_content = response.content.strip("```json\n").strip("```").strip()

            try:
                plan_dict = json.loads(cleaned_content)
                #print("[DEBUG]: Parsed plan", plan_dict)
                new_state = state.copy()
                new_state["study_plan"] = plan_dict
                new_state["scheduled_time"] = plan_dict.get("horario_sugerido")
                return new_state
            except json.JSONDecodeError as e:
                logger.error(f"[PLAN_AGENT]: JSON decode error: {e}")
                raise
            except Exception as e:
                logger.error(f"[PLAN_AGENT]: Unexpected error: {e}")
                raise

        return generate_plan

    def create_review_node(self):
        REVIEW_PROMPT = """Você é um especialista em educação responsável por revisar e validar planos de estudo.
        O plano de estudos que você receberá será implementado por uma LLM para criar uma sessão de estudo eficaz e interativa.

        Analise criticamente o seguinte plano:
        {plan}

        Considerando:
        - Estilo de aprendizagem do aluno: {learning_style}
        - Horário do encontro: {encounter_time}
        - Preferência de horário: {study_preference}
        - Objetivo geral da disciplina: {objetivo_geral}

        Verifique:
        1. Se os títulos são claros e bem estruturados
        2. Se as descrições são informativas e adequadas ao conteudo
        3. Se as atividades estão alinhadas com o estilo de aprendizagem
        4. Se a distribuição do tempo é apropriada
        5. Se os recursos são relevantes e bem escolhidos
        6. Se contém instruções de ensino claras e detalhadas
        7. Se os exemplos e exercícios de programação são apropriados
        8. Se o plano está profundo tecnicamente
        9. Se não há links genéricos ou irrelevantes como example.com
        10. Avalie os recursos web adicionados ao plano - eles devem ser relevantes, educativos e úteis

        O feedback deve seguir EXATAMENTE esta estrutura JSON:
            "status": "approved" ou "needs_revision",
            "links_problem": true ou false,  # Defina como true se houver problemas ESPECÍFICOS com links/recursos
            "feedback": "Seu feedback detalhado aqui",
            "suggestions": [
                "Sugestão 1",
                "Sugestão 2"
            ],
            "horario_validado": 
                "adequado": true/false,
                "comentario": "Comentário sobre o horário sugerido"

        IMPORTANTE: O campo "links_problem" deve ser definido como true APENAS quando:
        - Há links genéricos, quebrados ou irrelevantes no plano
        - Faltam recursos/links em seções importantes
        - Os links não são adequados ao tópico de estudo
        - Os recursos não são suficientemente educativos ou úteis
        - Não seja tão crítico, mas forneça sugestões construtivas

        Se você definir "links_problem" como true, inclua sugestões específicas sobre quais recursos melhorar e em quais seções.

        Retorne APENAS o JSON, sem explicações adicionais.

        NOTA: Use o objetivo geral da disciplina para saber o que é a disciplina, mas foque no tema da sessão.
        """

        prompt = ChatPromptTemplate.from_template(REVIEW_PROMPT)
        model = ChatOpenAI(model="gpt-4o", temperature=0.3)

        def review_plan(state: PlanningState) -> PlanningState:
            #print("[DEBUG]: Reviewing plan", state["study_plan"])

            encounter_time = state["user_profile"].get("horarios", {}).get("encontro", {})
            study_preference = state["user_profile"].get("horarios", {}).get("preferencia")
            objetivo_geral = state["objetivo_geral"]
            response = model.invoke(prompt.format(
                plan=state["study_plan"],
                learning_style=state["user_profile"].get("EstiloAprendizagem", {}),
                encounter_time=encounter_time,
                study_preference=study_preference,
                objetivo_geral=objetivo_geral
            ))

            print("[REVIEW]: Model response", response)

            cleaned_content = response.content.strip("```json\n").strip("```").strip()

            #print("[DEBUG]: Review response", response)
            new_state = state.copy()

            try:
                feedback_dict = json.loads(cleaned_content)
                new_state["review_feedback"] = feedback_dict

                # Garantir que o status esteja presente e seja válido
                if "status" not in feedback_dict:
                    feedback_dict["status"] = "needs_revision"
                    logger.warning("[PLAN_AGENT]: Status não encontrado no feedback do revisor, definindo como 'needs_revision'")

                # Log do status da revisão
                logger.info(f"[PLAN_AGENT]: Status da revisão: {feedback_dict['status']}")

                # Verificar e atualizar o horário sugerido se for validado
                if feedback_dict.get("horario_validado", {}).get("adequado", False):
                    new_state["scheduled_time"] = state["study_plan"].get("horario_sugerido")
                    logger.info(f"[PLAN_AGENT]: Horário validado com sucesso")
                else:
                    logger.info(f"[PLAN_AGENT]: Horário não validado ou ausente no feedback")
            except Exception as e:
                logger.error(f"[PLAN_AGENT]: Erro ao processar feedback: {str(e)}")
                new_state["review_feedback"] = {"status": "needs_revision", "feedback": cleaned_content}

            return new_state

        return review_plan

    def create_web_search_node(self):
        def search_web_resources(state: PlanningState) -> PlanningState:
            try:
                # Inicializar o cliente Tavily
                client = TavilyClient(api_key=TAVILY_API_KEY)
                new_state = state.copy()
                
                if not state.get("study_plan") or "plano_execucao" not in state["study_plan"]:
                    logger.error("[PLAN_AGENT]: Plano de estudo incompleto para pesquisa web")
                    return new_state
                
                topic = state["messages"][0].content if state["messages"] else ""
                objetivo_geral = state["objetivo_geral"]
                
                # Buscar recursos para cada etapa do plano
                plano_atualizado = state["study_plan"].copy()
                execution_plan = plano_atualizado.get("plano_execucao", [])
                
                for i, step in enumerate(execution_plan):
                    # Construir query para pesquisa
                    search_query = f"{step['titulo']} {step['descricao']} {topic} {objetivo_geral}"
                    logger.info(f"[TAVILY_SEARCH]: Pesquisando: {search_query[:100]}...")
                    
                    # Fazer pesquisa com Tavily
                    search_results = client.search(
                        query=search_query,
                        search_depth="advanced",
                        max_results=3
                    )
                    
                    # Adicionar recursos encontrados
                    for result in search_results.get("results", []):
                        if "url" in result and "title" in result:
                            new_resource = {
                                "tipo": "Artigo Web",
                                "descricao": result["title"][:100],
                                "url": result["url"]
                            }
                            
                            # Verificar se o recurso já existe
                            resource_exists = False
                            for existing_resource in step.get("recursos", []):
                                if existing_resource.get("url") == new_resource["url"]:
                                    resource_exists = True
                                    break
                            
                            if not resource_exists:
                                execution_plan[i]["recursos"].append(new_resource)
                                logger.info(f"[TAVILY_SEARCH]: Adicionado recurso: {new_resource['descricao']}")
                
                # Atualizar o plano com os novos recursos
                plano_atualizado["plano_execucao"] = execution_plan
                new_state["study_plan"] = plano_atualizado
                
                logger.info(f"[TAVILY_SEARCH]: Recursos adicionados com sucesso para o tópico: {topic[:100]}")
                return new_state
            
            except Exception as e:
                logger.error(f"[TAVILY_SEARCH_ERROR]: {str(e)}")
                # Retornar estado original em caso de falha
                return state
        
        return search_web_resources

    def create_update_plan_node(self):
        UPDATE_PROMPT = """Você é um especialista em planejamento educacional. 
        Sua tarefa é atualizar um plano de estudo com base no feedback do revisor.

        Plano original:
        {original_plan}

        Feedback do revisor:
        {feedback}

        Realize as modificações necessárias no plano original de acordo com as sugestões do revisor.
        Mantenha todos os elementos que não precisam de ajustes e modifique apenas o que foi criticado.

        Quaisquer recursos web (URLs) já adicionados devem ser mantidos, a menos que tenham sido especificamente criticados.

        Retorne o plano corrigido seguindo EXATAMENTE a mesma estrutura JSON do plano original, sem explicações adicionais.
        O plano deve seguir EXATAMENTE esta estrutura JSON:

            "plano_execucao": [
                    "titulo": "Título da seção"
                    "duracao": "XX minutos",
                    "descricao": "Descrição detalhada sobre qual assunto abordar",
                    "conteudo": ["Item 1", "Item 2"],
                    "recursos": [
                            "tipo": "Tipo do recurso",
                            "descricao": "Descrição do recurso",
                            "url": "URL"
                    ],
                    "atividade":
                        "descricao": "Descrição da atividade",
                        "tipo": "Tipo da atividade",
                        "formato": "Formato da atividade"

                    "progresso": 0
            ],
            "duracao_total": "60 minutos",
            "progresso_total": 0,
            "horario_sugerido":
                "data": "YYYY-MM-DD",
                "inicio": "HH:MM",
                "fim": "HH:MM",
                "justificativa": "Explique a escolha do horário em relação à data do encontro"
        """

        prompt = ChatPromptTemplate.from_template(UPDATE_PROMPT)
        model = ChatOpenAI(model="gpt-4o")
        
        def update_plan(state: PlanningState) -> PlanningState:
            try:
                logger.info(f"[PLAN_AGENT]: Atualizando plano conforme feedback")
                new_state = state.copy()
                
                if not state.get("study_plan") or not state.get("review_feedback"):
                    logger.error("[PLAN_AGENT]: Dados insuficientes para atualização do plano")
                    return state
                
                # Obter os componentes necessários
                original_plan = state["study_plan"]
                feedback = state["review_feedback"]
                
                # Incrementar contador de revisões
                current_count = state.get("revision_count", 0)
                new_state["revision_count"] = current_count + 1
                logger.info(f"[PLAN_AGENT]: Incrementando contador de revisões para {new_state['revision_count']}")
                
                # Invocar o modelo para atualizar o plano
                response = model.invoke(prompt.format(
                    original_plan=original_plan,
                    feedback=feedback
                ))
                
                # Processar a resposta
                cleaned_content = response.content.strip("```json\n").strip("```").strip()
                
                try:
                    updated_plan = json.loads(cleaned_content)
                    logger.info(f"[PLAN_AGENT]: Plano atualizado com sucesso (revisão #{new_state['revision_count']})")
                    
                    # Atualizar o estado com o plano corrigido
                    new_state["study_plan"] = updated_plan
                    
                    # Se tiver um novo horário sugerido, atualizar
                    if "horario_sugerido" in updated_plan:
                        new_state["scheduled_time"] = updated_plan["horario_sugerido"]
                    
                    return new_state
                    
                except json.JSONDecodeError as e:
                    logger.error(f"[PLAN_AGENT]: Erro ao decodificar JSON do plano atualizado: {e}")
                    return state
                    
            except Exception as e:
                logger.error(f"[PLAN_AGENT]: Erro ao atualizar plano: {e}")
                return state
                
        return update_plan

    def create_database_node(self):
        async def update_database(state: PlanningState) -> PlanningState:
            #print("[DEBUG]: Updating database with state", state)
            new_state = state.copy()

            if not state.get("study_plan"):
                print("[DEBUG]: No study plan found, skipping update")
                new_state["update_status"] = False
                return new_state

            plan_with_schedule = state["study_plan"].copy()
            if state.get("scheduled_time"):
                plan_with_schedule["horario_agendado"] = state["scheduled_time"]

            success = await self.db_tool.update_study_plan(
                state["id_sessao"],
                plan_with_schedule
            )

            #print("[DEBUG]: Database update success", success)
            new_state["update_status"] = success
            return new_state

        return update_database

    def create_initial_web_search_node(self):
        def initial_web_search(state: PlanningState) -> PlanningState:
            try:
                new_state = state.copy()
                # Inicializar recursos web vazios, se não existirem
                if "web_resources" not in new_state:
                    new_state["web_resources"] = []
                
                # Inicializar contador de revisões
                if "revision_count" not in new_state:
                    new_state["revision_count"] = 0
                
                # Obter o tópico e objetivo da disciplina
                topic = state["messages"][0].content if state["messages"] else ""
                objetivo_geral = state.get("objetivo_geral", "")
                if isinstance(objetivo_geral, list) and objetivo_geral:
                    objetivo_str = objetivo_geral[0] if objetivo_geral else ""
                else:
                    objetivo_str = str(objetivo_geral)[:50]
                
                if not topic:
                    logger.warning("[TAVILY_INITIAL_SEARCH]: Tópico não fornecido para pesquisa web")
                    return new_state
                
                # Extrair palavras-chave do tópico (remover preposições e palavras comuns)
                topic_words = topic.split()
                if len(topic_words) > 3:
                    topic_summary = " ".join(topic_words[:3])
                else:
                    topic_summary = topic
                
                # Criar consultas específicas e focadas para o Tavily
                search_queries = [
                    f"guia de aprendizado {topic_summary}",
                    f"tutorial {topic_summary} exemplos",
                    f"recursos educacionais {topic_summary}"
                ]
                
                logger.info(f"[TAVILY_INITIAL_SEARCH]: Iniciando pesquisa com {len(search_queries)} consultas específicas")
                
                # Inicializar cliente Tavily
                client = TavilyClient(api_key=TAVILY_API_KEY)
                web_resources = []
                
                # Fazer pesquisas para cada consulta, com tratamento de erros por consulta
                for idx, query in enumerate(search_queries):
                    try:
                        logger.info(f"[TAVILY_INITIAL_SEARCH]: Consulta {idx+1}/{len(search_queries)}: '{query}'")
                        
                        # Usar pesquisa rápida com poucos resultados para economizar tokens e tempo
                        search_results = client.search(
                            query=query,
                            search_depth="basic",  
                            max_results=2,
                            include_answer=False,  # Desabilitar síntese para economizar tokens
                            include_domains=["edu", "org", "github.com", "stackoverflow.com", "medium.com", "dev.to"]  # Focar em domínios educacionais
                        )
                        
                        # Processar resultados individualmente
                        for result in search_results.get("results", []):
                            if "url" in result and "title" in result:
                                # Criar recurso com dados mínimos necessários
                                resource = {
                                    "title": result.get("title", "")[:80],
                                    "url": result.get("url", ""),
                                    "content": result.get("content", "")[:150]  # Reduzir tamanho do snippet
                                }
                                
                                # Verificar duplicação por URL
                                is_duplicate = False
                                for existing in web_resources:
                                    if existing.get("url") == resource["url"]:
                                        is_duplicate = True
                                        break
                                
                                if not is_duplicate:
                                    web_resources.append(resource)
                                    logger.info(f"[TAVILY_INITIAL_SEARCH]: Adicionado: {resource['title'][:40]}...")
                    
                    except Exception as e:
                        logger.error(f"[TAVILY_INITIAL_SEARCH]: Erro na consulta {idx+1}: {str(e)}")
                        # Continuar com próxima consulta mesmo com erro
                
                # Limitar número total de recursos para evitar tokens excessivos
                if len(web_resources) > 5:
                    web_resources = web_resources[:5]
                    logger.info("[TAVILY_INITIAL_SEARCH]: Limitando para 5 recursos principais")
                
                # Atualizar o estado com os recursos encontrados
                new_state["web_resources"] = web_resources
                logger.info(f"[TAVILY_INITIAL_SEARCH]: Total de recursos para plano: {len(web_resources)}")
                
                return new_state
                
            except Exception as e:
                logger.error(f"[TAVILY_INITIAL_SEARCH]: Erro geral: {str(e)}")
                # Em caso de erro completo, retornar estado inicial sem falhar o processo
                new_state = state.copy()
                new_state["web_resources"] = []
                return new_state
        
        return initial_web_search
        
    def create_update_links_node(self):
        QUERY_GENERATION_PROMPT = """Você é um especialista em pesquisas educacionais online.
        
        Com base no feedback recebido sobre um plano de estudo, gere consultas específicas para buscar melhores recursos web.
        
        Tópico: {topic}
        Objetivo: {objective}
        
        Feedback do revisor: {feedback}
        
        Sugestões para melhoria: {suggestions}
        
        Gere 3 consultas de pesquisa específicas e curtas (máximo 8 palavras cada) para encontrar recursos web educacionais 
        que atendam às necessidades apontadas no feedback.
        
        Formato da resposta (apenas JSON):
        {{"queries": ["consulta 1", "consulta 2", "consulta 3"]}}
        """
        
        query_prompt = ChatPromptTemplate.from_template(QUERY_GENERATION_PROMPT)
        # Usar modelo menor para gerar consultas
        query_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        def update_plan_links(state: PlanningState) -> PlanningState:
            try:
                logger.info(f"[LINK_UPDATE]: Iniciando atualização específica de links")
                new_state = state.copy()
                
                if not state.get("study_plan") or not state.get("review_feedback"):
                    logger.error("[LINK_UPDATE]: Dados insuficientes para atualização")
                    return state
                
                # Obter dados necessários
                topic = state["messages"][0].content if state["messages"] else ""
                feedback = state["review_feedback"].get("feedback", "")
                suggestions = state["review_feedback"].get("suggestions", [])
                objective = state.get("objetivo_geral", "")
                
                # Incrementar contador de revisões
                current_count = state.get("revision_count", 0)
                new_state["revision_count"] = current_count + 1
                logger.info(f"[LINK_UPDATE]: Revisão de links #{new_state['revision_count']}")
                
                # Gerar consultas usando GPT-4o-mini
                logger.info(f"[LINK_UPDATE]: Gerando consultas específicas com base no feedback")
                query_response = query_model.invoke(query_prompt.format(
                    topic=topic,
                    objective=objective,
                    feedback=feedback,
                    suggestions=suggestions
                ))
                
                # Extrair consultas da resposta
                try:
                    # Limpar a resposta e carregar como JSON
                    cleaned_content = query_response.content.strip().strip("```json").strip("```").strip()
                    queries_data = json.loads(cleaned_content)
                    search_queries = queries_data.get("queries", [])
                    
                    if not search_queries:
                        # Fallback para consultas básicas
                        search_queries = [
                            f"recursos educacionais {topic}",
                            f"exemplos {topic}",
                            f"tutorial {topic} prático"
                        ]
                    
                    logger.info(f"[LINK_UPDATE]: Geradas {len(search_queries)} consultas para pesquisa")
                    
                except Exception as e:
                    logger.error(f"[LINK_UPDATE]: Erro ao processar consultas: {str(e)}")
                    # Fallback
                    search_queries = [
                        f"recursos educacionais {topic}",
                        f"exemplos {topic}",
                        f"tutorial {topic} prático"
                    ]
                
                # Inicializar cliente Tavily
                client = TavilyClient(api_key=TAVILY_API_KEY)
                new_resources = []
                
                # Realizar pesquisas
                for idx, query in enumerate(search_queries):
                    try:
                        logger.info(f"[LINK_UPDATE]: Buscando recursos para consulta: '{query}'")
                        
                        # Pesquisa Tavily
                        search_results = client.search(
                            query=query,
                            search_depth="basic",  # Busca básica para economizar tokens
                            max_results=2,
                            include_answer=False,
                            include_domains=["edu", "org", "github.com", "stackoverflow.com", "medium.com", "dev.to"]
                        )
                        
                        # Processar resultados
                        for result in search_results.get("results", []):
                            if "url" in result and "title" in result:
                                new_resource = {
                                    "tipo": "Artigo Web",
                                    "descricao": result["title"][:100],
                                    "url": result["url"]
                                }
                                new_resources.append(new_resource)
                                logger.info(f"[LINK_UPDATE]: Encontrado recurso: '{new_resource['descricao'][:40]}...'")
                    
                    except Exception as e:
                        logger.error(f"[LINK_UPDATE]: Erro ao buscar para consulta {idx+1}: {str(e)}")
                
                # Atualizar o plano com os novos recursos
                if not new_resources:
                    logger.warning("[LINK_UPDATE]: Nenhum novo recurso encontrado")
                    return state
                
                # Obter o plano atual
                plan = state["study_plan"].copy()
                execution_steps = plan.get("plano_execucao", [])
                
                if not execution_steps:
                    logger.error("[LINK_UPDATE]: Plano de execução vazio")
                    return state
                
                # Verificar URLs já existentes (evitar duplicatas)
                existing_urls = set()
                for step in execution_steps:
                    for resource in step.get("recursos", []):
                        if "url" in resource and resource["url"]:
                            existing_urls.add(resource["url"])
                
                # Adicionar novos recursos ao plano
                resources_added = 0
                for i, step in enumerate(execution_steps):
                    for resource in new_resources:
                        # Pular recursos já existentes
                        if resource["url"] in existing_urls:
                            continue
                            
                        # Adicionar recurso
                        if "recursos" not in step:
                            step["recursos"] = []
                        
                        step["recursos"].append(resource)
                        existing_urls.add(resource["url"])
                        resources_added += 1
                        
                        # Limitar a 2 novos recursos por etapa
                        if resources_added >= 2:
                            break
                    
                    # Continuar até distribuir todos os recursos ou atingir o limite
                    if resources_added >= len(new_resources) or resources_added >= len(execution_steps) * 2:
                        break
                
                # Atualizar o plano no estado
                plan["plano_execucao"] = execution_steps
                new_state["study_plan"] = plan
                
                logger.info(f"[LINK_UPDATE]: Adicionados {resources_added} novos recursos ao plano")
                return new_state
                
            except Exception as e:
                logger.error(f"[LINK_UPDATE]: Erro geral: {str(e)}")
                return state
        
        return update_plan_links

    async def create_session_plan(self, topic: str, student_profile: dict, id_sessao: str, objetivo_geral: list) -> dict:
        start_time = time.time()
        student_email = student_profile.get("Email", "unknown")
        logger.info(f"[PLAN_START] Usuário={student_email} | sessão={id_sessao} | tópico={topic}")

        if not topic:
            logger.warning(f"[PLAN_ERROR] Usuário={student_email} | sessão={id_sessao} | erro=Tópico não fornecido")
            return {
                "error": "Tópico não fornecido",
                "plan": None,
                "feedback": None,
                "update_status": False,
                "id_sessao": id_sessao,
                "scheduled_time": None
            }

        initial_state = PlanningState(
            messages=[HumanMessage(content=topic)],
            study_plan={},
            user_profile=student_profile,
            next_step=None,
            review_feedback=None,
            id_sessao=id_sessao,
            update_status=None,
            scheduled_time=None,
            objetivo_geral=objetivo_geral,
            revision_count=0,
            web_resources=[]
        )

        try:
            result = await self.workflow.ainvoke(initial_state)
            # print("[DEBUG]: Workflow result", result)

            final_plan = result["study_plan"]
            final_plan["created_at"] = datetime.now(timezone.utc).isoformat()

            # Calcular métricas para o log
            end_time = time.time()
            elapsed_time = end_time - start_time
            steps_count = len(final_plan.get("plano_execucao", []))
            
            # Registrar sucesso no log
            logger.info(f"[PLAN_SUCCESS] Usuário={student_email} | sessão={id_sessao} | tempo={elapsed_time:.2f}s | etapas={steps_count}")

            return {
                "plan": final_plan,
                "feedback": result["review_feedback"],
                "update_status": result["update_status"],
                "id_sessao": id_sessao,
                "scheduled_time": result.get("scheduled_time"),
                "objetivo_geral": objetivo_geral
            }
        except Exception as e:
            # print(f"[DEBUG]: Error creating session plan: {e}")
            logger.error(f"[PLAN_ERROR] Usuário={student_email} | sessão={id_sessao} | erro={str(e)}")
            return {
                "error": f"Erro na criação do plano: {str(e)}",
                "plan": None,
                "feedback": None,
                "update_status": False,
                "id_sessao": id_sessao,
                "scheduled_time": None,
                "objetivo_geral": objetivo_geral
            }