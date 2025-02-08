# LangChain-study

Caso seja necessário atualizar, devido a novas depêndencias inseridas:

```sh
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Para rodar, siga estes passos:

```sh
./scripts/create_db_psql.sh
./scripts/qdrant_create.sh
./scripts/start_mongodb.sh
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Acesse a API em http://localhost:8000/docs#/
