# api/controllers/discipline_controller.py

from api.dispatchers.discipline_dispatcher import DisciplineDispatcher
import json
from chains.chain_setup import DisciplinChain



class DisciplineController:
    def __init__(self, dispatcher: DisciplineDispatcher, disciplin_chain: DisciplinChain = None):
        self.dispatcher = dispatcher
        self.disciplin_chain = disciplin_chain

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

    def create_discipline_from_pdf(self, text: str, user_email: str):
        try:
            data = self.disciplin_chain.create_discipline_from_pdf(text, user_email)
            print(data)

            # saving data into json
            # with open('output_disciplina.json', 'w') as f:
            #     json.dump(data, f)
            # Ler o arquivo disciplin.json, economizando chamadas de API
            # with open('output_disciplina.json', 'r') as f:
            #     data = json.load(f)
            # print("Arquivo JSON lido com sucesso.")
            # print(data)

            # print("Tentando converter o texto do PDF em JSON...")
            data = json.loads(data)
            print("Conversão bem-sucedida.")
            # Obtenha o nome do curso diretamente do JSON
            #discipline_name = data['curso']['nome']
            #print(f"Disciplina '{discipline_name}' encontrada no PDF.")
            # Chamar a função create_discipline_from_pdf no dispatcher para inserir os dados no banco de dados
            self.dispatcher.create_discipline_from_pdf(data, user_email)
            #print(f"Disciplina '{discipline_name}' e sessões foram salvas com sucesso no banco de dados.")

        except Exception as e:
            print(f"Erro ao criar disciplina a partir do PDF: {e}")
