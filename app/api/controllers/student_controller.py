from api.dispatchers.student_dispatcher import StudentDispatcher


class StudentController:
    def __init__(self, dispatcher: StudentDispatcher):
        self.dispatcher = dispatcher

    def get_all_students(self, current_user: str):
        return self.dispatcher.get_all_students(current_user)

    def create_student(self, name: str, matricula: str, current_user: str):
        # Organizar os dados do estudante
        student_data = {
            'Nome': name,
            'Matricula': matricula,
            'CriadoPor': current_user  # O e-mail do usuário é passado
        }
        # Chamar o dispatcher para criar o estudante
        return self.dispatcher.create_student(student_data)

    def update_student(self, student_id: int, name: str = None, matricula: str = None, current_user: str = None):
        # Coletar os dados a serem atualizados
        updated_data = {}
        if name:
            updated_data['Nome'] = name
        if matricula:
            updated_data['Matricula'] = matricula

        # Chamar o dispatcher para atualizar o estudante
        return self.dispatcher.update_student(student_id, updated_data, current_user)

    def delete_student(self, student_id: int, current_user: str):
        # Chamar o dispatcher para deletar o estudante
        return self.dispatcher.delete_student(student_id, current_user)
