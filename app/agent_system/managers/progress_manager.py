from datetime import datetime, timezone
from typing import Any, Dict, Optional

from database.mongo_database_manager import MongoDatabaseManager


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
