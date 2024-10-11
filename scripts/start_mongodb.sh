#!/bin/bash

# Nome do container
CONTAINER_NAME="mongo_student"

# Porta externa (host) e interna (container)
HOST_PORT=27017
CONTAINER_PORT=27017

# Credenciais de autenticação
MONGO_INITDB_ROOT_USERNAME=admin
MONGO_INITDB_ROOT_PASSWORD=123456789

# Diretório de dados no host
DATA_DIR="$(pwd)/mongo_data"

# Verifica se o diretório de dados existe, se não, cria
if [ ! -d "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR"
fi

# Verifica se o container já está em execução
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "O container $CONTAINER_NAME já está em execução."
else
    # Verifica se o container existe mas está parado
    if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
        echo "Iniciando o container existente $CONTAINER_NAME..."
        docker start $CONTAINER_NAME
    else
        # Cria e inicia um novo container MongoDB com autenticação e persistência de dados
        echo "Criando e iniciando um novo container $CONTAINER_NAME com autenticação e persistência de dados..."
        docker run --name $CONTAINER_NAME \
            -e MONGO_INITDB_ROOT_USERNAME=$MONGO_INITDB_ROOT_USERNAME \
            -e MONGO_INITDB_ROOT_PASSWORD=$MONGO_INITDB_ROOT_PASSWORD \
            -v $DATA_DIR:/data/db \
            -d -p $HOST_PORT:$CONTAINER_PORT mongo
    fi
fi
