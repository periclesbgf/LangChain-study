from api.dispatchers.educator_dispatcher import EducatorDispatcher


class EducatorController:
    def __init__(self, dispatcher: EducatorDispatcher):
        self.dispatcher = dispatcher

    def get_all_educators(self, current_user: str):
        return self.dispatcher.get_all_educators(current_user)

    def create_educator(self, name: str, instituicao: str, especializacao_disciplina: str, current_user: str):
        # Organizar os dados do educador
        educator_data = {
            'Nome': name,
            'Instituicao': instituicao,
            'EspecializacaoDisciplina': especializacao_disciplina,
            'CriadoPor': current_user  # O e-mail do usuário é passado
        }
        # Chamar o dispatcher para criar o educador
        return self.dispatcher.create_educator(educator_data)

    def update_educator(self, educator_id: int, name: str = None, instituicao: str = None, especializacao_disciplina: str = None, current_user: str = None):
        # Coletar os dados a serem atualizados
        updated_data = {}
        if name:
            updated_data['Nome'] = name
        if instituicao:
            updated_data['Instituicao'] = instituicao
        if especializacao_disciplina:
            updated_data['EspecializacaoDisciplina'] = especializacao_disciplina

        # Chamar o dispatcher para atualizar o educador
        return self.dispatcher.update_educator(educator_id, updated_data, current_user)

    def delete_educator(self, educator_id: int, current_user: str):
        # Chamar o dispatcher para deletar o educador
        return self.dispatcher.delete_educator(educator_id, current_user)
