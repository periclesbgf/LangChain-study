#!/bin/bash

# Nome do container
CONTAINER_NAME="qdrant_db"

# URL do Qdrant
QDRANT_URL="http://localhost:6333"

# Verifica se o container já está em execução
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "O container $CONTAINER_NAME já está em execução."
else
    # Verifica se o container existe mas não está em execução
    if [ "$(docker ps -aq -f status=exited -f name=$CONTAINER_NAME)" ]; then
        echo "Iniciando o container existente $CONTAINER_NAME..."
        docker start $CONTAINER_NAME
    else
        # Cria e inicia um novo container Qdrant
        echo "Criando e iniciando um novo container $CONTAINER_NAME..."
        docker run --name $CONTAINER_NAME -p 6333:6333 -p 6334:6334 -d qdrant/qdrant
    fi
fi

# Verifica se o Qdrant está acessível na URL especificada
status_code=$(curl -o /dev/null -s -w "%{http_code}\n" $QDRANT_URL)

if [ "$status_code" -eq 200 ]; then
    echo "Qdrant está rodando em $QDRANT_URL"
else
    echo "Erro: Qdrant não está acessível em $QDRANT_URL"
fi
