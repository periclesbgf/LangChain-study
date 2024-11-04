#!/bin/bash

# Nome do container
CONTAINER_NAME="qdrant_db"

# Diretório de armazenamento
STORAGE_DIR="$(pwd)/qdrant_storage"

# Função para verificar status do Qdrant
check_qdrant_status() {
    local max_attempts=30
    local attempt=1
    local wait_time=2

    echo "Verificando status do Qdrant..."
    while [ $attempt -le $max_attempts ]; do
        # Mudando para /healthz
        status_code=$(curl -o /dev/null -s -w "%{http_code}\n" http://localhost:6333/healthz)
        if [ "$status_code" -eq 200 ]; then
            echo "Qdrant está rodando e saudável em http://localhost:6333"
            return 0
        fi
        echo "Tentativa $attempt de $max_attempts. Aguardando serviço iniciar..."
        sleep $wait_time
        attempt=$((attempt + 1))
    done
    return 1
}

# Remove qualquer container existente com o mesmo nome
if docker ps -a | grep -q $CONTAINER_NAME; then
    echo "Removendo container existente..."
    docker rm -f $CONTAINER_NAME
fi

# Configura o diretório de armazenamento
echo "Configurando diretório de armazenamento..."
sudo mkdir -p "$STORAGE_DIR"
sudo chown -R $(whoami):$(groups | awk '{print $1}') "$STORAGE_DIR"
sudo chmod -R 755 "$STORAGE_DIR"

# Cria e inicia o container
echo "Criando e iniciando um novo container $CONTAINER_NAME..."
docker run --name $CONTAINER_NAME \
    -p 6333:6333 \
    -p 6334:6334 \
    -e QDRANT_ALLOW_GRPC=true \
    -v "$STORAGE_DIR:/qdrant/storage" \
    -d qdrant/qdrant

if [ $? -ne 0 ]; then
    echo "Erro ao criar e iniciar o container."
    exit 1
fi

# Verifica se o container está rodando
if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "Erro: Container não está rodando após tentativa de inicialização."
    echo "Logs do container:"
    docker logs $CONTAINER_NAME
    exit 1
fi

# Aguarda o serviço iniciar e verifica o status
if ! check_qdrant_status; then
    echo "Erro: Qdrant não iniciou corretamente após várias tentativas."
    echo "Logs do container:"
    docker logs $CONTAINER_NAME
    exit 1
fi

# Tenta criar uma coleção de teste
echo "Verificando API do Qdrant..."
curl -X PUT "http://localhost:6333/collections/test_collection" \
    -H "Content-Type: application/json" \
    -d '{
        "vectors": {
            "size": 1536,
            "distance": "Cosine"
        }
    }'

# Lista as coleções para verificar
echo -e "\nListando coleções:"
curl -s "http://localhost:6333/collections"

# Exibe informações do container
echo -e "\nInformações do container:"
docker inspect $CONTAINER_NAME | grep -E "IPAddress|State|Ports"

# Exibe os logs mais recentes
echo -e "\nÚltimas linhas do log:"
docker logs --tail 10 $CONTAINER_NAME

echo "Setup concluído com sucesso!"