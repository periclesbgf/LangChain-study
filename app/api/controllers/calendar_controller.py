# controllers/calendar_controller.py

from fastapi import HTTPException
from api.dispatchers.calendar_dispatcher import CalendarDispatcher
from database.sql_database_manager import DatabaseManager
from datetime import datetime
import pytz

class CalendarController:
    def __init__(self, dispatcher: CalendarDispatcher):
        self.dispatcher = dispatcher

    def convert_to_brasilia_time(self, utc_time):
        brasilia_zone = pytz.timezone("America/Sao_Paulo")

        if isinstance(utc_time, datetime):
            brasilia_dt = utc_time.astimezone(brasilia_zone)
        else:
            utc_time = datetime.strptime(utc_time, "%Y-%m-%d %H:%M:%S%z")
            brasilia_dt = utc_time.astimezone(brasilia_zone)

        return brasilia_dt.strftime("%Y-%m-%d %H:%M:%S")

    def get_all_events(self, current_user: str):
        return self.dispatcher.get_all_events(current_user)

    def create_event(self, title: str, description: str, start_time, end_time, location: str, current_user: str, course_id: int = None, categoria: str = None, importancia: str = None, material: str = None, convert_timezone: bool = True):
        """
        Create a calendar event

        Args:
            title (str): Event title
            description (str): Event description
            start_time: Event start time
            end_time: Event end time
            location (str): Event location
            current_user (str): User email
            convert_timezone (bool): Whether to convert times to Brasilia timezone. Default False
            course_id (int, optional): Course ID if applicable
        """
        # if convert_timezone:
        #     start_time_final = self.convert_to_brasilia_time(start_time)
        #     end_time_final = self.convert_to_brasilia_time(end_time)
        # else:
        #     # Ensure we have string format without conversion
        #     if isinstance(start_time, datetime):
        #         start_time_final = start_time.strftime("%Y-%m-%d %H:%M:%S")
        #     else:
        #         start_time_final = start_time

        #     if isinstance(end_time, datetime):
        #         end_time_final = end_time.strftime("%Y-%m-%d %H:%M:%S")
        #     else:
        #         end_time_final = end_time

        # print("Start time: ", start_time_final)
        # print("End time: ", end_time_final)
        # print(f"Creating event for user: {current_user}")
        if end_time <= start_time:
            raise HTTPException(status_code=400, detail="End time must be after start time")

        event_data = {
            'GoogleEventId': f"event-{current_user}-{title}",
            'Titulo': title,
            'Descricao': description,
            'Inicio': start_time,
            'Fim': end_time,
            'Local': location,
            'Categoria': categoria,
            'Importancia': importancia,
            'Material': material,
            # 'IdCurso': course_id  # Descomente se o campo 'IdCurso' existir na tabela
        }
        return self.dispatcher.create_event(event_data, current_user)

    def update_event(self, event_id: int, title: str = None, description: str = None, start_time = None, end_time = None, location: str = None, current_user: str = None, categoria: str = None, importancia: str = None, material: str = None):
        if end_time <= start_time:
            raise HTTPException(status_code=400, detail="End time must be after start time")

        updated_data = {}
        if title:
            updated_data['Titulo'] = title
        if description:
            updated_data['Descricao'] = description
        if start_time:
            updated_data['Inicio'] = self.convert_to_brasilia_time(start_time)
        if end_time:
            updated_data['Fim'] = self.convert_to_brasilia_time(end_time)
        if location:
            updated_data['Local'] = location
        if categoria:
            updated_data['Categoria'] = categoria
        if importancia:
            updated_data['Importancia'] = importancia
        if material:
            updated_data['Material'] = material

        return self.dispatcher.update_event(event_id, updated_data, current_user)

    def delete_event(self, event_id: int, current_user: str):
        return self.dispatcher.delete_event(event_id, current_user)