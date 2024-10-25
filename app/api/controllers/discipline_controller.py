# api/controllers/discipline_controller.py

from api.dispatchers.discipline_dispatcher import DisciplineDispatcher
import json
from api.controllers.plan_controller import PlanController
from chains.chain_setup import DisciplinChain
from datetime import datetime, timezone



class DisciplineController:
    def __init__(self, dispatcher: DisciplineDispatcher, disciplin_chain: DisciplinChain = None):
        self.dispatcher = dispatcher
        self.disciplin_chain = disciplin_chain
        self.plan_controller = PlanController()  # Instanciar PlanController


    def get_all_user_disciplines(self, current_user: str):
        return self.dispatcher.get_all_disciplines_for_student(current_user)

    def get_discipline_by_id(self, discipline_id: int, current_user: str):
        # Call dispatcher to get the discipline by ID
        return self.dispatcher.get_discipline_by_id(discipline_id, current_user)

    def create_discipline(self, nome_curso: str, ementa: str, objetivos: str, current_user: str):
        # Organize discipline data
        discipline_data = {
            'NomeCurso': nome_curso,
            'Ementa': ementa,
            'Objetivos': objetivos,
        }
        return self.dispatcher.create_discipline(discipline_data, current_user)

    def update_discipline(self, discipline_id: int, nome_curso: str = None, ementa: str = None, objetivos: str = None, current_user: str = None):
        # Collect data to update
        updated_data = {}
        if nome_curso:
            updated_data['NomeCurso'] = nome_curso
        if ementa:
            updated_data['Ementa'] = ementa
        if objetivos:
            updated_data['Objetivos'] = objetivos

        # Call dispatcher to update the discipline
        return self.dispatcher.update_discipline(discipline_id, updated_data, current_user)

    def delete_discipline(self, discipline_id: int, current_user: str):
        # Call dispatcher to delete the discipline
        return self.dispatcher.delete_discipline(discipline_id, current_user)

    async def create_discipline_from_pdf(self, text: str, user_email: str):
        try:
            data = self.disciplin_chain.create_discipline_from_pdf(text, user_email)
            print(data)
            data = json.loads(data)
            print("Conversão bem-sucedida.")

            # Chamar o dispatcher para criar a disciplina e obter os IDs
            course_id, session_ids = await self.dispatcher.create_discipline_from_pdf(data, user_email)

            # Após criar a disciplina e as sessões, criar planos de estudo vazios
            for session_id in session_ids:
                empty_plan = {
                    "id_sessao": str(session_id),
                    "disciplina_id": str(course_id),
                    "disciplina": data['curso'].get('nome', 'Sem Nome'),
                    "descricao": "",  # Descrição vazia
                    "objetivo_sessao": "",
                    "plano_execucao": [],
                    "duracao_total": "",
                    "progresso_total": 0,
                    "created_at": datetime.now(timezone.utc)
                }
                plan_result = await self.plan_controller.create_study_plan(empty_plan)
                if not plan_result:
                    print(f"Falha ao criar plano de estudo vazio para a sessão {session_id}")

            print("Disciplina e planos de estudo vazios criados com sucesso.")

        except Exception as e:
            print(f"Erro ao criar disciplina a partir do PDF: {e}")
            raise e