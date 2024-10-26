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
