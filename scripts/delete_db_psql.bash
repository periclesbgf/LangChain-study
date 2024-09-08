#!/bin/bash

# Nome do container
CONTAINER_NAME="psql_student"

# Verifica se o container está em execução
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Parando o container $CONTAINER_NAME..."
    docker stop $CONTAINER_NAME
else
    echo "O container $CONTAINER_NAME não está em execução."
fi

# Remove o container se ele existir
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Removendo o container $CONTAINER_NAME..."
    docker rm $CONTAINER_NAME
else
    echo "O container $CONTAINER_NAME não existe."
fi
