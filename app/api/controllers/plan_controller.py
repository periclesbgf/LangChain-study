# controllers/plan_controller.py
from typing import Dict, Any, List
from api.dispatchers.plan_dispatcher import PlanDispatcher

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