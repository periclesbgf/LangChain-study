#!/bin/bash

# Nome do container
CONTAINER_NAME="mongo_student"
VOLUME_NAME="mongo_student_data"

# Verifica se o container está em execução
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Parando o container $CONTAINER_NAME..."
    docker stop $CONTAINER_NAME
else
    echo "O container $CONTAINER_NAME não está em execução."
fi

# Remove o container e o volume se ele existir
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Removendo o container $CONTAINER_NAME..."
    docker rm $CONTAINER_NAME
else
    echo "O container $CONTAINER_NAME não existe."
fi

# Verifica se o volume existe
if [ "$(docker volume ls -q -f name=$VOLUME_NAME)" ]; then
    echo "Removendo o volume $VOLUME_NAME..."
    docker volume rm $VOLUME_NAME
else
    echo "O volume $VOLUME_NAME não existe."
fi
