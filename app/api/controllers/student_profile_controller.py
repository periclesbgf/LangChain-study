from api.dispatchers.student_profile_dispatcher import StudentProfileDispatcher
from sqlalchemy.exc import IntegrityError
from datetime import datetime
from database.mongo_database_manager import MongoDatabaseManager
from api.models import StudentProfileCreate


class StudentProfileController:
    def __init__(self, sql_database_manager):
        self.dispatcher = StudentProfileDispatcher(sql_database_manager)

    def get_student_profile(self, email: str):
        user_id = self.dispatcher.get_user_id_by_email(email)
        if not user_id:
            raise ValueError("Usuário não encontrado.")
        return self.dispatcher.get_student_profile(user_id)

    def create_student_profile(self, email: str, perfil_aprendizado: dict, perfil_felder_silverman: dict):
        user_id = self.dispatcher.get_user_id_by_email(email)
        if not user_id:
            raise ValueError("Usuário não encontrado.")
        id_perfil_felder = self._create_felder_profile_if_exists(perfil_felder_silverman)
        perfil_data = {
            "IdUsuario": user_id,
            "DadosPerfil": perfil_aprendizado,
            "IdPerfilFelderSilverman": id_perfil_felder,
            "DataUltimaAtualizacao": datetime.utcnow()
        }
        self.dispatcher.create_student_learning_profile(perfil_data)

    def update_student_profile(self, email: str, perfil_aprendizado: dict, perfil_felder_silverman: dict):
        user_id = self.dispatcher.get_user_id_by_email(email)
        if not user_id:
            raise ValueError("Usuário não encontrado.")
        id_perfil_felder = self._create_felder_profile_if_exists(perfil_felder_silverman)
        updated_data = {
            "DadosPerfil": perfil_aprendizado,
            "IdPerfilFelderSilverman": id_perfil_felder,
            "DataUltimaAtualizacao": datetime.utcnow()
        }
        self.dispatcher.update_student_learning_profile(user_id, updated_data)

    def delete_student_profile(self, email: str):
        user_id = self.dispatcher.get_user_id_by_email(email)
        if not user_id:
            raise ValueError("Usuário não encontrado.")
        self.dispatcher.delete_student_learning_profile(user_id)

    def _create_felder_profile_if_exists(self, profile_data: dict):
        if profile_data:
            return self.dispatcher.create_felder_silverman_profile(profile_data)
        return None


class ProfileController:
    def __init__(self, mongo_manager: MongoDatabaseManager):
        self.mongo_manager = mongo_manager
        self.collection = mongo_manager.db["student_learn_preference"]

    async def create_profile(self, profile_data: StudentProfileCreate, current_user: dict):
        profile = {
            "user_email": current_user["sub"],
            "profile_data": profile_data.dict(),
            "created_at": datetime.now(timezone.utc)
        }
        result = await self.collection.insert_one(profile)
        return result

    async def get_profile(self, current_user: dict):
        profile = await self.collection.find_one({"user_email": current_user["sub"]})
        if not profile:
            raise HTTPException(status_code=404, detail="Perfil não encontrado")
        return profile

    async def update_profile(self, profile_data: StudentProfileCreate, current_user: dict):
        result = await self.collection.update_one(
            {"user_email": current_user["sub"]},
            {"$set": {"profile_data": profile_data.dict(), "updated_at": datetime.utcnow()}}
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Perfil não encontrado")

    async def delete_profile(self, current_user: dict):
        result = await self.collection.delete_one({"user_email": current_user["sub"]})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Perfil não encontrado")