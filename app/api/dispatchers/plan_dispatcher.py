from database.mongo_database_manager import MongoDatabaseManager
from typing import Dict, Any, List

class PlanDispatcher:
    def __init__(self):
        self.mongo_manager = MongoDatabaseManager()  # Instancia o MongoDatabaseManager

    async def save_plan(self, plan: Dict[str, Any]) -> str:
        result = await self.mongo_manager.create_study_plan(plan)
        return result

    async def fetch_plan(self, id_sessao: str) -> Dict[str, Any]:
        plan = await self.mongo_manager.get_study_plan(id_sessao)
        return plan

    async def update_plan(self, id_sessao: str, updated_data: Dict[str, Any]) -> bool:
        success = await self.mongo_manager.update_study_plan(id_sessao, updated_data)
        return success

    async def delete_plan(self, id_sessao: str) -> bool:
        success = await self.mongo_manager.delete_study_plan(id_sessao)
        return success

    async def get_sessions_without_plan(self, student_email: str, study_sessions) -> List[Dict[str, Any]]:
        """
        Recupera todas as sessões sem plano através do MongoDB Manager.
        """
        return await self.mongo_manager.get_sessions_without_plan(student_email, study_sessions)

    async def verify_session_has_plan(self, id_sessao: str) -> bool:
        """
        Verifica se uma sessão específica já possui um plano de estudo.
        """
        return await self.mongo_manager.verify_session_has_plan(id_sessao)

    async def create_automatic_plan(self, student_email: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles the automatic plan creation process.
        """
        try:
            # Verify if session already has a plan
            existing_plan = await self.mongo_manager.get_study_plan(session_data['session_id'])
            if existing_plan and existing_plan.get('plano_execucao'):
                raise Exception("Session already has a plan")

            # Create the plan
            result = await self.mongo_manager.create_automatic_study_plan(student_email, session_data)
            return result
        except Exception as e:
            print(f"Error in dispatcher create_automatic_plan: {e}")
            raise