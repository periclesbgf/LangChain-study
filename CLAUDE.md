# LangChain-Study Project Guidelines

## Setup Commands
- Create Python environment: `python3.12 -m venv venv`
- Activate environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

## Database Setup
- PostgreSQL (relational): `./scripts/create_db_psql.sh`
- Qdrant (vector DB): `./scripts/qdrant_create.sh`
- MongoDB (document store): `./scripts/start_mongodb.sh`

## Running the Application
- Start all databases (see above commands)
- Run the application: `cd app && uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- Access API docs: `http://localhost:8000/docs#/`

## Testing
- Run agent tests: `python app/agent/agent_test.py`
- Run RAG test: `python app/database/test_rag.py`
- Run image summary test: `python app/database/test_image_summary.py`

## Code Style
- Use snake_case for variables, functions, methods
- Use PascalCase for classes, TypedDict, and Pydantic models
- Add type hints for function parameters and return values
- Use async/await for database operations
- Handle errors with specific try/except blocks and logging
- Follow modular agent architecture (planner, analytics, interactive)

## Database Conventions
- PostgreSQL: Used for relational data (users, courses, sessions)
- MongoDB: Used for document storage and chat history
- Qdrant: Used for vector embeddings and semantic search

## Architecture Note
This project implements a multi-agent educational system with specialized agents
for personalized learning, as outlined in app/agent/idea.md.