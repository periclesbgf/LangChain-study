from database.mongo_database_manager import MongoDatabaseManager
from typing import Dict, Any

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