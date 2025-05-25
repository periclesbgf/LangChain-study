# controllers/plan_controller.py
from typing import Dict, Any, List
from api.dispatchers.plan_dispatcher import PlanDispatcher
from agent.plan_agent import SessionPlanWorkflow
from datetime import datetime, timezone

class PlanController:
    def __init__(self):
        self.dispatcher = PlanDispatcher()  # Instancia o Dispatcher

    async def create_study_plan(self, plan: Dict[str, Any]) -> str:
        result = await self.dispatcher.save_plan(plan)
        return result

    async def get_study_plan(self, id_sessao: str) -> Dict[str, Any]:
        plan = await self.dispatcher.fetch_plan(id_sessao)
        return plan

    async def update_study_plan(self, id_sessao: str, updated_data: Dict[str, Any]) -> bool:
        success = await self.dispatcher.update_plan(id_sessao, updated_data)
        return success

    async def delete_study_plan(self, id_sessao: str) -> bool:
        success = await self.dispatcher.delete_plan(id_sessao)
        return success

    async def get_full_study_plan(self, id_sessao: str) -> Dict[str, Any]:
        plan = await self.dispatcher.fetch_plan(id_sessao)
        return plan

    async def get_sessions_without_plan(self, student_email: str, study_sessions) -> List[Dict[str, Any]]:
        """
        Recupera todas as sessões do estudante que não possuem plano de execução.
        """
        return await self.dispatcher.get_sessions_without_plan(student_email, study_sessions)

    async def verify_session_has_plan(self, id_sessao: str) -> bool:
        """
        Verifica se uma sessão específica já possui um plano de estudo.
        """
        return await self.dispatcher.verify_session_has_plan(id_sessao)

    async def create_automatic_plan(self, student_email: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates an automatic study plan using SessionPlanWorkflow.
        """
        try:
            # Get student profile
            student_profile = await self.db_manager.get_student_profile(student_email, 'student_learn_preference')
            if not student_profile:
                raise Exception("Student profile not found")

            # Initialize workflow
            workflow = SessionPlanWorkflow(self.db_manager)
            
            # Generate plan using the workflow
            result = await workflow.create_session_plan(
                topic=session_data.get('descricao', ''),
                student_profile=student_profile,
                id_sessao=session_data.get('session_id')
            )

            if result.get("error"):
                raise Exception(result["error"])

            # Extract only plano_execucao and duracao_total
            plan_data = {
                "plano_execucao": result["plan"]["plano_execucao"],
                "duracao_total": result["plan"]["duracao_total"]
            }

            # Update the plan in database
            await self.dispatcher.update_plan(session_data['session_id'], plan_data)

            return {
                "id": session_data['session_id'],
                "plan": result["plan"],
                "feedback": result["review_feedback"]
            }

        except Exception as e:
            print(f"Error in create_automatic_plan: {e}")
            raise

    async def update_step_progress(self, session_id: str, step_index: int, new_progress: int) -> bool:
        return await self.dispatcher.update_step_progress(session_id, step_index, new_progress)

    async def get_plan_progress(self, session_id: str) -> Dict[str, Any]:
        return await self.dispatcher.get_plan_progress(session_id)

    async def get_sessions_without_plan_by_discipline(
        self, 
        student_email: str, 
        discipline_id: int, 
        study_sessions: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Recupera todas as sessões de uma disciplina específica que não possuem plano de execução.

        Args:
            student_email (str): E-mail do estudante.
            discipline_id (int): ID da disciplina.
            study_sessions (Dict[str, List[Dict[str, Any]]]): Dicionário contendo uma lista de sessões filtradas.

        Returns:
            List[Dict[str, Any]]: Lista de sessões sem plano.
        """
        try:
            # Filtra sessões para a disciplina especificada
            sessions_for_discipline = [
                session for session in study_sessions
                if session["IdCurso"] == discipline_id
            ]

            # Busca sessões sem planos usando o dispatcher
            sessions_without_plan = await self.dispatcher.get_sessions_without_plan_by(
                student_email, sessions_for_discipline
            )

            return sessions_without_plan
        except Exception as e:
            print(f"Erro ao buscar sessões sem plano por disciplina: {e}")
            raise Exception(f"Erro ao buscar sessões sem plano por disciplina: {e}")
