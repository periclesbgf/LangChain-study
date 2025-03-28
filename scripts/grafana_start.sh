#!/bin/bash
# Script para subir os containers do Grafana, Loki e Promtail

# Remove containers existentes (caso estejam rodando)
docker rm -f grafana loki promtail 2>/dev/null

# Inicia o container do Loki
echo "Subindo Loki..."
docker run -d --name loki -p 3100:3100 grafana/loki:3.4.1

# Inicia o container do Promtail
echo "Subindo Promtail..."
docker run -d --name promtail grafana/promtail:3.4.1

# Inicia o container do Grafana
echo "Subindo Grafana..."
docker run -d --name grafana -p 3000:3000 grafana/grafana

echo "Containers subidos com sucesso."
