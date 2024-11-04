#!/bin/bash

# Nome do container
CONTAINER_NAME="qdrant_db"

# Verifica se o container está em execução
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Parando o container $CONTAINER_NAME..."
    docker stop $CONTAINER_NAME
fi

# Remove o container
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Removendo o container $CONTAINER_NAME..."
    docker rm $CONTAINER_NAME
    echo "Container removido com sucesso!"
else
    echo "Container $CONTAINER_NAME não encontrado."
fi

# Opcional: remover a imagem do Qdrant
read -p "Deseja remover também a imagem do Qdrant? (s/n) " remove_image
if [ "$remove_image" = "s" ]; then
    echo "Removendo a imagem qdrant/qdrant..."
    docker rmi qdrant/qdrant
    echo "Imagem removida com sucesso!"
fi

echo "Limpeza concluída!"