#!/bin/bash

# Diretório atual
current_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Current directory: ${current_dir}"
# Script create_db_psql.sh
"${current_dir}/create_db_psql.sh"

# Script start_mongodb.sh
"${current_dir}/start_mongodb.sh"

# Script qdrant_create.sh (requer sudo)
sudo -E "${current_dir}/qdrant_create.sh"
