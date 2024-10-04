# controllers/calendar_controller.py

from api.dispatchers.calendar_dispatcher import CalendarDispatcher
from database.sql_database_manager import DatabaseManager


class CalendarController:
    def __init__(self, dispatcher: CalendarDispatcher):
        self.dispatcher = dispatcher

    def get_all_events(self, current_user: str):
        return self.dispatcher.get_all_events(current_user)

    def create_event(self, title: str, description: str, start_time, end_time, location: str, current_user: str):
        # Organizar os dados do evento
        print(f"Creating event for user: {current_user}")
        event_data = {
            'GoogleEventId': f"event-{current_user}-{title}",  # ID único com base no e-mail
            'Titulo': title,
            'Descricao': description,
            'Inicio': start_time,
            'Fim': end_time,
            'Local': location,
            # O 'CriadoPor' será definido no dispatcher com o user_id obtido
        }
        # Chamar o dispatcher para criar o evento
        return self.dispatcher.create_event(event_data, current_user)

    def update_event(self, event_id: int, title: str = None, description: str = None, start_time = None, end_time = None, location: str = None, current_user: str = None):
        # Coletar os dados a serem atualizados
        updated_data = {}
        if title:
            updated_data['Titulo'] = title
        if description:
            updated_data['Descricao'] = description
        if start_time:
            updated_data['Inicio'] = start_time
        if end_time:
            updated_data['Fim'] = end_time
        if location:
            updated_data['Local'] = location
        
        # Chama o dispatcher para atualizar o evento e passa o current_user
        return self.dispatcher.update_event(event_id, updated_data, current_user)

    def delete_event(self, event_id: int, current_user: str):
        # Chama o dispatcher para deletar o evento, passando o current_user
        return self.dispatcher.delete_event(event_id, current_user)