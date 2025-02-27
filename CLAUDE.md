# LangChain-Study Project Guidelines

## Project Overview
This project implements an AI-powered educational platform using a multi-agent architecture to provide personalized learning experiences. The system consists of:

- **Interactive Agent**: 24/7 tutor that helps students with queries and provides learning resources
- **Analytical Agent**: Analyzes student interactions and updates learning profiles
- **Plan Generator Agent**: Creates personalized execution plans based on student needs

The platform supports multimodal content (text, images, videos, tables) and uses the Felder-Silverman learning style model for personalization.

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

## System Architecture
The application uses a 3-tier architecture:

### 1. Database Layer
- **PostgreSQL**: Relational data (users, educators, courses, study sessions, calendar events)
- **MongoDB**: Document storage (student profiles, chat history, study plans, images, PDFs)
- **Qdrant**: Vector database for embeddings and semantic search

### 2. API Layer
- **FastAPI**: Web framework for API endpoints
- **Controllers**: Business logic implementation (authentication, chat, calendar, etc.)
- **Dispatchers**: Interface between API endpoints and controllers
- **WebSockets**: Real-time communication

#### API Structure
The API follows a structured pattern:

1. **Endpoints (app/api/endpoints/)**: 
   - Define HTTP routes and handle request/response formatting
   - Use FastAPI for input validation with Pydantic models
   - Implement authentication via dependency injection
   - Key files: routes.py, student.py, chat.py, calendar.py, classroom.py

2. **Controllers (app/api/controllers/)**:
   - Implement business logic for each domain
   - Process requests from endpoints and call appropriate services
   - Coordinate interactions between various system components
   - Key files: controller.py, student_controller.py, chat_controller.py, auth.py

3. **Dispatchers (app/api/dispatchers/)**:
   - Handle data access operations
   - Translate controller requests into database operations
   - Manage exceptions and data formatting
   - Key files: student_dispatcher.py, login_dispatcher.py, chat_dispatcher.py

### 3. Agent System
- **LangGraph**: Agent orchestration and workflow management
- **RAG**: Retrieval-Augmented Generation for context-aware responses
- **Multiple Specialized Agents**: Planning, retrieval, teaching, analysis
- **Context Manager**: Manages conversation and user context

## Integration Components
- **MQTT Controller**: IoT device integration with secure TLS connections
- **Google Classroom**: Integration with Google's educational platform
- **Authentication**: Email/password and Google OAuth support
- **Vector Search**: Semantic search across educational materials

## Services Provided
1. **Authentication Services**:
   - Local user authentication
   - Google OAuth integration
   - Password reset functionality

2. **Educational Services**:
   - Student and educator management
   - Course discipline handling
   - Study plan creation and management
   - Calendar scheduling for educational activities

3. **AI Interaction Services**:
   - Chat-based tutoring via LangChain agents
   - SQL query generation and execution
   - Document processing and vector database integration
   - Image handling and analysis

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

## Key Features
- **Multimodal Content Processing**: Text, images, videos, tables
- **Personalized Learning**: Adapts to individual learning styles
- **Critical Thinking Focus**: Guides students rather than providing direct answers
- **Progress Tracking**: Analyzes and reports on student learning progress
- **Proactive Support**: Generates personalized study plans and recommendations