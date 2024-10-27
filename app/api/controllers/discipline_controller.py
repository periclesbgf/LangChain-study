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

    def create_discipline(self, nome_curso: str, ementa: str, objetivos: str, 
                         turno_estudo: str, horario_inicio: str, horario_fim: str, 
                         current_user: str):
        discipline_data = {
            'NomeCurso': nome_curso,
            'Ementa': ementa,
            'Objetivos': objetivos,
            'TurnoEstudo': turno_estudo,
            'HorarioInicio': horario_inicio,
            'HorarioFim': horario_fim,
        }
        return self.dispatcher.create_discipline(discipline_data, current_user)

    def update_discipline(self, discipline_id: int, nome_curso: str = None, 
                        ementa: str = None, objetivos: str = None, 
                        turno_estudo: str = None, horario_inicio: str = None, 
                        horario_fim: str = None, current_user: str = None):
        updated_data = {}
        if nome_curso:
            updated_data['NomeCurso'] = nome_curso
        if ementa:
            updated_data['Ementa'] = ementa
        if objetivos:
            updated_data['Objetivos'] = objetivos
        if turno_estudo:
            updated_data['TurnoEstudo'] = turno_estudo
        if horario_inicio:
            updated_data['HorarioInicio'] = horario_inicio
        if horario_fim:
            updated_data['HorarioFim'] = horario_fim

        return self.dispatcher.update_discipline(discipline_id, updated_data, current_user)

    def delete_discipline(self, discipline_id: int, current_user: str):
        # Call dispatcher to delete the discipline
        return self.dispatcher.delete_discipline(discipline_id, current_user)

    async def create_discipline_from_pdf(self, text: str, user_email: str, 
                                    turno_estudo: str, horario_inicio: str, 
                                    horario_fim: str):
        try:
            # Extrair dados do PDF usando o chain
            data = self.disciplin_chain.create_discipline_from_pdf(text, user_email)
            data = json.loads(data)
            
            # Chamar o dispatcher com todos os parâmetros necessários
            course_id, session_ids = await self.dispatcher.create_discipline_from_pdf(
                discipline_json=data,
                user_email=user_email,
                turno_estudo=turno_estudo,
                horario_inicio=horario_inicio,
                horario_fim=horario_fim
            )

            # Criar planos de estudo para cada sessão
            cronograma_data = data.get('cronograma', [])

            for i, session_id in enumerate(session_ids):
                encontro = cronograma_data[i]

                empty_plan = {
                    "id_sessao": str(session_id),
                    "disciplina_id": str(course_id),
                    "disciplina": data['curso'].get('nome', 'Sem Nome'),
                    "descricao": encontro.get('conteudo', ""),
                    "objetivo_sessao": encontro.get('estrategia', ""),
                    "plano_execucao": [],
                    "duracao_total": "",
                    "progresso_total": 0,
                    "created_at": datetime.now(timezone.utc)
                }

                plan_result = await self.plan_controller.create_study_plan(empty_plan)
                if not plan_result:
                    print(f"Falha ao criar plano de estudo para a sessão {session_id}")

            print("Disciplina e planos de estudo criados com sucesso.")
            return course_id, session_ids

        except Exception as e:
            print(f"Erro ao criar disciplina a partir do PDF: {e}")
            raise e