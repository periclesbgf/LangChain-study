#!/bin/bash

# Nome do container
CONTAINER_NAME="psql_student"

# Verifica se o container já está em execução
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "O container $CONTAINER_NAME já está em execução."
else
    # Verifica se o container existe mas não está em execução
    if [ "$(docker ps -aq -f status=exited -f name=$CONTAINER_NAME)" ]; then
        echo "Iniciando o container existente $CONTAINER_NAME..."
        docker start $CONTAINER_NAME
    else
        # Cria e inicia um novo container PostgreSQL
        echo "Criando e iniciando um novo container $CONTAINER_NAME..."
        docker run --name $CONTAINER_NAME -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=123456789 -e POSTGRES_DB=postgres -p 5432:5432 -d postgres
    fi
fi
