# app/database/vector_db.py

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from qdrant_client import QdrantClient
import uuid
from qdrant_client.http import models

from qdrant_client.http.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    VectorParams,
)
from utils import OPENAI_API_KEY, QDRANT_URL
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import (
    Filter, FieldCondition, MatchValue, VectorParams, Distance
)
from langchain_core.documents import Document
from typing import List, Optional
import hashlib

class QdrantHandler:
    def __init__(self, url: str, collection_name: str, embeddings):
        print(f"Inicializando QdrantHandler para a coleção '{collection_name}'...")
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.client = QdrantClient(url=url, prefer_grpc=False)
        print(f"QdrantClient inicializado para a URL '{url}'.")

        self.ensure_collection_exists()

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            validate_collection_config=True
        )
        print(f"QdrantHandler inicializado com sucesso para a coleção '{collection_name}'.")

    def ensure_collection_exists(self):
        collections = self.client.get_collections().collections
        print(f"Coleções disponíveis: {[col.name for col in collections]}")

        if self.collection_name not in [col.name for col in collections]:
            vector_size = len(self.embeddings.embed_query("test query"))
            print(f"Recriando coleção '{self.collection_name}' com tamanho de vetor {vector_size}.")
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Coleção '{self.collection_name}' criada com sucesso.")
        else:
            print(f"Coleção '{self.collection_name}' já existe.")

    def add_document(
        self,
        student_email: str,
        session_id: str,  # Obrigatório para associar o documento à sessão
        content: str,
        access_level: str = "session",  # Pode ser "global", "discipline", ou "session"
        disciplina_id: Optional[str] = None,  # Opcional, necessário se o acesso for por disciplina
        specific_file_id: Optional[str] = None,  # ID do arquivo, se aplicável
        metadata_extra: Optional[dict] = None,  # Metadados adicionais opcionais
        embedding: Optional[List[float]] = None  # Novo parâmetro para o embedding
    ):
        """
        Adiciona um documento no banco de vetores com metadados completos para filtragem.

        Args:
            student_email (str): E-mail do estudante dono do documento.
            session_id (str): ID da sessão associada ao documento.
            content (str): Conteúdo do documento.
            access_level (str): Nível de acesso ("global", "discipline", "session").
            disciplina_id (Optional[str]): ID da disciplina, necessário se for por disciplina.
            specific_file_id (Optional[str]): ID específico do arquivo, se houver.
            metadata_extra (Optional[dict]): Metadados adicionais opcionais.
            embedding (Optional[List[float]]): Embedding do conteúdo, se disponível.
        """
        print(f"Adicionando documento com metadados: student_email={student_email}, session_id={session_id}")
        print(f"Nível de acesso: {access_level}, Disciplina: {disciplina_id}, Arquivo: {specific_file_id}")

        # Gera um hash do conteúdo para evitar duplicações
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

        # Metadados principais do documento
        metadata = {
            "student_email": student_email,
            "session_id": session_id,
            "content_hash": content_hash,
            "access_level": access_level
        }

        # Adiciona o ID da disciplina, se aplicável
        if disciplina_id and access_level == "discipline":
            metadata["discipline_id"] = disciplina_id

        # Adiciona o ID do arquivo, se aplicável
        if specific_file_id:
            metadata["file_id"] = specific_file_id

        # Adiciona metadados extras, se fornecidos
        if metadata_extra:
            metadata.update(metadata_extra)

        # Cria o documento com o conteúdo e metadados
        document = Document(page_content=content, metadata=metadata)

        # Adiciona o documento no banco de vetores (Qdrant)
        try:
            # Se o embedding for fornecido, utilizá-lo ao adicionar o documento
            if embedding is not None:
                self.vector_store.add_documents([document], embeddings=[embedding])
            else:
                # Caso contrário, o vector_store gerará o embedding internamente
                self.vector_store.add_documents([document])
            print("Documento adicionado com sucesso.")
        except Exception as e:
            print(f"Erro ao adicionar documento: {e}")

    def similarity_search_with_filter(
        self,
        query: str,
        student_email: str,
        session_id: Optional[str] = None,
        disciplina_id: Optional[str] = None,
        k: int = 5,
        use_global: bool = True,
        use_discipline: bool = True,
        use_session: bool = True,
        specific_file_id: Optional[str] = None,
        specific_metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        Realiza busca por similaridade com filtros flexíveis usando a estrutura correta do Qdrant.
        """
        print(f"\n[SEARCH] Iniciando busca com filtros:")
        print(f"[SEARCH] Query: {query}")
        print(f"[SEARCH] Student: {student_email}")
        print(f"[SEARCH] Config: global={use_global}, discipline={use_discipline}, session={use_session}")
        print(f"[SEARCH] Disciplina: {disciplina_id}, Session: {session_id}")

        try:
            # Inicializa as condições must com o email do estudante
            must_conditions = [
                models.FieldCondition(
                    key="metadata.student_email",
                    match=models.MatchValue(value=student_email)
                )
            ]

            # Busca por ID específico
            if specific_file_id:
                print(f"[SEARCH] Buscando por ID específico: {specific_file_id}")
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.file_id",
                        match=models.MatchValue(value=specific_file_id)
                    )
                )
            
            # Adiciona metadados específicos se fornecidos
            if specific_metadata:
                print(f"[SEARCH] Adicionando metadados específicos: {specific_metadata}")
                for key, value in specific_metadata.items():
                    must_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchValue(value=str(value))
                        )
                    )

            # Lista para armazenar os níveis de acesso permitidos
            allowed_access_levels = []
            
            # Global access
            if use_global:
                print("[SEARCH] Adicionando acesso global aos níveis permitidos")
                allowed_access_levels.append("global")

            # Discipline access
            if use_discipline and disciplina_id:
                print(f"[SEARCH] Adicionando acesso de disciplina aos níveis permitidos: {disciplina_id}")
                allowed_access_levels.append("discipline")
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.discipline_id",
                        match=models.MatchValue(value=disciplina_id)
                    )
                )

            # Session access
            if use_session and session_id:
                print(f"[SEARCH] Adicionando acesso de sessão aos níveis permitidos: {session_id}")
                allowed_access_levels.append("session")
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.session_id",
                        match=models.MatchValue(value=session_id)
                    )
                )

            # Adiciona o filtro de níveis de acesso permitidos
            if allowed_access_levels:
                print(f"[SEARCH] Níveis de acesso permitidos: {allowed_access_levels}")
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.access_level",
                        match=models.MatchAny(any=allowed_access_levels)
                    )
                )

            # Cria o filtro final com todas as condições must
            search_filter = models.Filter(must=must_conditions)
            
            print(f"[SEARCH] Filtro final construído: {search_filter}")
            return self._execute_search(query, search_filter, k)

        except Exception as e:
            print(f"[ERROR] Erro durante a busca: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _execute_search(self, query: str, search_filter: models.Filter, k: int) -> List[Document]:
        """
        Executa a busca com o filtro construído e trata os resultados.
        """
        try:
            print(f"[SEARCH] Executando busca com filtro: {search_filter}")
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=search_filter
            )

            print(f"[SEARCH] Encontrados {len(results)} resultados")
            for i, doc in enumerate(results, 1):
                print(f"[SEARCH] Resultado {i}:")
                print(f"  - Nível de acesso: {doc.metadata.get('access_level')}")
                print(f"  - Preview: {doc.page_content[:100]}...")

            return results

        except Exception as e:
            print(f"[ERROR] Erro na execução da busca: {str(e)}")
            raise

    def similarity_search_without_filter(self, query: str, k: int = 5):
        print(f"Realizando busca sem filtro: query={query}")

        try:
            results = self.vector_store.similarity_search(query=query, k=k)
            print(f"{len(results)} documentos encontrados sem filtro.")
            return results
        except Exception as e:
            print(f"Erro na busca sem filtro: {e}")
            return []

    def debug_metadata(self):
        print("Depurando metadados...")

        try:
            # Recupera documentos da coleção com scroll
            results, _ = self.client.scroll(collection_name=self.collection_name, limit=10)
            print(f"{len(results)} documentos encontrados.")

            for result in results:
                # Exibe todo o payload para análise completa
                metadata = result.payload
                print(f"Metadados Recuperados: {metadata}")

                # Acessa o campo 'disciplina' corretamente
                if 'metadata' in metadata:
                    disciplina_value = metadata['metadata'].get('disciplina')
                    print(f"Disciplina: {disciplina_value} (Tipo: {type(disciplina_value)})")
                else:
                    print("⚠️ O campo 'metadata' está ausente ou mal formatado.")

                # Validar se 'disciplina' está correta
                if disciplina_value is None:
                    print("⚠️ O campo 'disciplina' está ausente ou vazio.")
                elif not isinstance(disciplina_value, str):
                    print("⚠️ O campo 'disciplina' não é uma string.")

        except Exception as e:
            print(f"Erro ao listar documentos: {e}")




    def compare_search_results(self, query: str, student_email: str, disciplina: str, k: int = 5):
        print("🔍 Comparando resultados de busca...")

        print("🔍 Buscando sem filtro...")
        no_filter_results = self.similarity_search_without_filter(query, k)

        print("🔍 Buscando com filtro...")
        filter_results = self.similarity_search_with_filter(query, student_email, disciplina, k)

        print(f"Sem filtro: {len(no_filter_results)} | Com filtro: {len(filter_results)}")
        if len(filter_results) == 0:
            print("⚠️ Nenhum resultado encontrado com filtro. Verifique os metadados e filtros.")

    def document_exists(self, content_hash: str, student_email: str, disciplina: int) -> bool:
        query_filter = Filter(
            must=[
                FieldCondition(key="content_hash", match=MatchValue(value=content_hash)),
                FieldCondition(key="student_email", match=MatchValue(value=student_email)),
                FieldCondition(key="disciplina", match=MatchValue(value=disciplina)),
            ]
        )
        try:
            results, _ = self.client.scroll(collection_name=self.collection_name, scroll_filter=query_filter, limit=1)
            exists = len(results) > 0
            print(f"Documento encontrado: {exists}")
            return exists
        except Exception as e:
            print(f"Erro ao verificar existência do documento: {e}")
            return False

    def fix_document_metadata(self):
        """
        Corrige os metadados dos documentos que possuem 'disciplina' como None.
        """
        try:
            results, _ = self.client.scroll(collection_name=self.collection_name, limit=10)
            for result in results:
                metadata = result.payload
                if metadata.get("disciplina") is None:
                    metadata["disciplina"] = "1"  # Corrigir para string

                    # Atualizar o documento no Qdrant
                    self.client.update_point(
                        collection_name=self.collection_name,
                        id=result.id,
                        payload=metadata
                    )
                    print(f"Metadados corrigidos para o documento {result.id}")
        except Exception as e:
            print(f"Erro ao corrigir metadados: {e}")




class TextSplitter:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 50):
        """
        Inicializa o TextSplitter com tamanho de chunk e sobreposição.
        
        :param chunk_size: Tamanho máximo de cada pedaço de texto
        :param chunk_overlap: Sobreposição entre pedaços
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Divide documentos em pedaços menores.
        
        :param documents: Lista de documentos a serem divididos
        :return: Lista de documentos divididos
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return text_splitter.split_documents(documents)


class Embeddings:
    def __init__(self):
        """
        Inicializa a classe de embeddings.
        """
        self.embeddings = self.load_embeddings()

    def load_embeddings(self) -> OpenAIEmbeddings:
        """
        Carrega as embeddings do OpenAI.
        
        :return: Instância de OpenAIEmbeddings
        """
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    def get_embeddings(self) -> OpenAIEmbeddings:
        """
        Retorna a instância de embeddings.
        
        :return: Instância de OpenAIEmbeddings
        """
        return self.embeddings