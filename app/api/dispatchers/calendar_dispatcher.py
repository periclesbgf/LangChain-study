# dispatchers/calendar_dispatcher.py

from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException
from sql_interface.sql_tables import tabela_eventos_calendario
from database.sql_database_manager import DatabaseManager

class CalendarDispatcher:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def get_all_events(self, current_user: str):
        user_id = self.database_manager.get_user_id_by_email(current_user)
        print(f"Fetching calendar events for user: {current_user} (ID: {user_id})")

        try:
            # Buscar todos os eventos de calendário criados pelo usuário
            events = self.database_manager.get_all_events_by_user(tabela_eventos_calendario, user_id)

            # Transformar o resultado em uma lista de dicionários
            event_list = [
                {
                    "IdEvento": event.IdEvento,
                    "GoogleEventId": event.GoogleEventId,
                    "Titulo": event.Titulo,
                    "Descricao": event.Descricao,
                    "Inicio": event.Inicio.isoformat() if event.Inicio else None,
                    "Fim": event.Fim.isoformat() if event.Fim else None,
                    "Local": event.Local,
                    "CriadoPor": event.CriadoPor
                }
                for event in events
            ]
            return event_list

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching calendar events: {e}")

    def create_event(self, event_data: dict, current_user: str):
        try:
            # Obter o IdUsuario do current_user (email)
            user_id = self.database_manager.get_user_id_by_email(current_user)

            if not user_id:
                raise HTTPException(status_code=404, detail="Usuário não encontrado.")

            # Atualizar o 'CriadoPor' no event_data com o user_id correto
            event_data['CriadoPor'] = user_id

            # Criar um novo evento no banco de dados
            evento_id = self.database_manager.inserir_dado_evento(tabela_eventos_calendario, event_data)
            return {"message": "Evento criado com sucesso", "IdEvento": evento_id}
        except IntegrityError:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=400, detail="Event already exists.")
        except Exception as e:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=500, detail=f"Error creating calendar event: {e}")

    def update_event(self, event_id: int, updated_data: dict, current_user: str):
        try:
            # Obter o IdUsuario do current_user (email)
            user_id = self.database_manager.get_user_id_by_email(current_user)

            # Atualizar o evento no banco de dados (somente se o usuário for o criador)
            self.database_manager.atualizar_dado(
                tabela_eventos_calendario,
                (tabela_eventos_calendario.c.IdEvento == event_id) & (tabela_eventos_calendario.c.CriadoPor == user_id),
                updated_data
            )
            return True
        except HTTPException as e:
            # Repassar exceções HTTP
            raise e
        except Exception as e:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=500, detail=f"Error updating calendar event: {e}")

    def delete_event(self, event_id: int, current_user: str):
        try:
            # Obter o IdUsuario do current_user (email)
            user_id = self.database_manager.get_user_id_by_email(current_user)

            # Deletar o evento do banco de dados (somente se o usuário for o criador)
            self.database_manager.deletar_dado(
                tabela_eventos_calendario,
                (tabela_eventos_calendario.c.IdEvento == event_id) & (tabela_eventos_calendario.c.CriadoPor == user_id)
            )
            return True
        except HTTPException as e:
            # Repassar exceções HTTP
            raise e
        except Exception as e:
            self.database_manager.session.rollback()
            raise HTTPException(status_code=500, detail=f"Error deleting calendar event: {e}")
