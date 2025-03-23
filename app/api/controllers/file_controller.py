import asyncio
import logging
import uuid
import hashlib
from datetime import datetime
import traceback

from fastapi import HTTPException
from io import BytesIO

# Supondo que o AccessLevel seja um Enum definido em api.endpoints.models, se necessário
from api.endpoints.models import AccessLevel
from database.mongo_database_manager import MongoDatabaseManager, MongoPDFHandler
from database.vector_db import QdrantHandler

logger = logging.getLogger(__name__)

class FileController:
    def __init__(
        self,
        mongo_db: MongoDatabaseManager,
        pdf_handler: MongoPDFHandler,
        qdrant_handler: QdrantHandler,
        student_email: str,
        disciplina: str,
        session_id: str,
        access_level: str
    ):
        self.mongo_db = mongo_db
        self.pdf_handler = pdf_handler
        self._qdrant_handler = qdrant_handler
        self.student_email = student_email
        self.disciplina = disciplina
        self.session_id = session_id
        self.access_level = access_level

    async def process_files(self, files):
        """Processa os arquivos enviados de forma concorrente e retorna os resultados."""
        processing_tasks = []
        result_messages = []

        for file in files:
            logger.info(f"Processing file: {file.filename}")
            task = asyncio.create_task(self._process_single_file(file))
            processing_tasks.append(task)

        results = await asyncio.gather(*processing_tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Erro ao processar o arquivo '{files[i].filename}': {str(result)}"
                logger.error(error_msg)
                result_messages.append({"status": "error", "message": error_msg})
            elif isinstance(result, dict) and "error" in result:
                logger.warning(f"Arquivo rejeitado: {result['message']}")
                result_messages.append({"status": "rejected", "message": result["message"]})
            else:
                result_messages.append({"status": "success", "message": f"Arquivo '{files[i].filename}' processado com sucesso."})

        return result_messages

    async def _process_single_file(self, file):
        """Processa um único arquivo, verificando o conteúdo, gerando metadados e armazenando-o."""
        try:
            filename = file.filename
            content_type = file.content_type
            is_pdf = content_type == "application/pdf" or filename.lower().endswith(".pdf")

            try:
                content = await file.read()

                if not content:
                    logger.error(f"Arquivo {filename} vazio")
                    return {
                        "error": "FILE_EMPTY",
                        "message": f"O arquivo {filename} está vazio."
                    }

                file_size = len(content)
                logger.info(f"Arquivo {filename} lido com sucesso, tamanho: {file_size} bytes")

                if is_pdf:
                    ESTIMATED_PAGE_SIZE = 100 * 1024
                    MAX_PAGES = 50
                    MAX_PDF_SIZE = ESTIMATED_PAGE_SIZE * MAX_PAGES

                    estimated_pages = file_size / ESTIMATED_PAGE_SIZE
                    logger.info(f"PDF {filename} tem tamanho de {file_size} bytes, estimativa de {estimated_pages:.1f} páginas")

                    if file_size > MAX_PDF_SIZE:
                        logger.warning(f"PDF {filename} excede o limite estimado de páginas: {estimated_pages:.1f} > {MAX_PAGES}")
                        return {
                            "error": "PDF_TOO_LARGE",
                            "message": f"O arquivo PDF é muito grande. Por favor, envie um PDF com no máximo {MAX_PAGES} páginas."
                        }

                    logger.info(f"PDF {filename} aceito, estimativa de {estimated_pages:.1f} páginas")
            except Exception as e:
                logger.error(f"Erro ao ler o arquivo {filename}: {str(e)}", exc_info=True)
                return {
                    "error": "FILE_READ_ERROR",
                    "message": f"Erro ao processar o arquivo {filename}. O arquivo pode estar corrompido ou inacessível."
                }

            pdf_uuid = str(uuid.uuid4())
            content_hash = hashlib.md5(content).hexdigest()
            storage_tasks = []

            if is_pdf:
                pdf_task = asyncio.create_task(
                    self.pdf_handler.store_pdf(
                        pdf_uuid=pdf_uuid,
                        pdf_bytes=content,
                        student_email=self.student_email,
                        disciplina=self.disciplina,
                        session_id=self.session_id,
                        filename=filename,
                        content_hash=content_hash,
                        access_level=self.access_level
                    )
                )
                storage_tasks.append(pdf_task)

            vector_task = asyncio.create_task(
                self._qdrant_handler.process_file(
                    content=content,
                    filename=filename,
                    student_email=self.student_email,
                    session_id=self.session_id,
                    disciplina=self.disciplina,
                    access_level=self.access_level
                )
            )
            storage_tasks.append(vector_task)

            await asyncio.gather(*storage_tasks)
            logger.info(f"File '{filename}' processed successfully")
            return {"status": "success"}

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
            return {
                "error": "PROCESSING_ERROR",
                "message": f"Erro ao processar o arquivo {filename}. Detalhes: {str(e)}"
            }
