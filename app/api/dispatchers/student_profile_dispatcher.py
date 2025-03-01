from sql_interface.sql_tables import tabela_perfil_aprendizado_aluno, tabela_perfis_felder_silverman

class StudentProfileDispatcher:
    def __init__(self, database_manager):
        self.database_manager = database_manager

    def get_user_id_by_email(self, email: str):
        return self.database_manager.get_user_id_by_email(email)

    def get_student_profile(self, user_id: int):
        return self.database_manager.get_one(tabela_perfil_aprendizado_aluno, {"IdUsuario": user_id})

    def create_felder_silverman_profile(self, profile_data: dict):
        return self.database_manager.inserir_dado(tabela_perfis_felder_silverman, profile_data)

    def create_student_learning_profile(self, profile_data: dict):
        self.database_manager.inserir_dado(tabela_perfil_aprendizado_aluno, profile_data)

    def update_student_learning_profile(self, user_id: int, updated_data: dict):
        self.database_manager.atualizar_dado(
            tabela_perfil_aprendizado_aluno,
            {"IdUsuario": user_id},
            updated_data
        )

    def delete_student_learning_profile(self, user_id: int):
        self.database_manager.deletar_dado(
            tabela_perfil_aprendizado_aluno,
            {"IdUsuario": user_id}
        )
