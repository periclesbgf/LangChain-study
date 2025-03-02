from typing import List, Dict, Any, Optional, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.tools import BaseTool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_react_agent
from langchain.pydantic_v1 import BaseModel, Field
from database.vector_db import QdrantHandler
from database.mongo_database_manager import MongoDatabaseManager, MongoImageHandler
import json
import asyncio
import base64
from datetime import datetime
import traceback

# Import existing tools
from agent.tools import search_youtube, search_wikipedia


class VectorDBSearchInput(BaseModel):
    """Inputs for vector database search."""
    query: str = Field(description="The search query to find relevant information in the vector database")
    student_email: str = Field(description="Email of the student making the query")
    session_id: str = Field(description="Current session ID")
    discipline_id: str = Field(description="Discipline ID for filtering")
    content_type: str = Field(description="Type of content to search for: 'text', 'image', or 'table'")
    limit: int = Field(default=3, description="Maximum number of results to return")
    
    class Config:
        """Configuration for this pydantic object."""
        schema_extra = {
            "example": {
                "query": "O que é uma matriz?",
                "student_email": "aluno@example.com",
                "session_id": "session123",
                "discipline_id": "matematica",
                "content_type": "text",
                "limit": 3
            }
        }


class RetrievalTools:
    """Tools for retrieving information from various sources."""
    
    def __init__(
        self,
        qdrant_handler: QdrantHandler,
        mongo_manager: MongoDatabaseManager,
        student_email: str,
        discipline_id: str,
        session_id: str
    ):
        self.qdrant_handler = qdrant_handler
        self.mongo_manager = mongo_manager
        self.student_email = student_email
        self.discipline_id = discipline_id
        self.session_id = session_id
        
        # Ensure mongo_manager has an image_handler
        if not hasattr(mongo_manager, 'image_handler'):
            self.mongo_manager.image_handler = MongoImageHandler(mongo_manager)

    def search_vector_db(self, input_data: VectorDBSearchInput) -> str:
        """
        Search the vector database for relevant information based on the query and filters.
        
        Args:
            input_data: An object containing search parameters including query, email, and filters
            
        Returns:
            A formatted string with search results from the vector database
        """
        try:
            # Convert if a dictionary was provided instead of a proper model
            if isinstance(input_data, dict):
                input_data = VectorDBSearchInput(**input_data)
                
            # For convenience, if student_email was not provided, use the one from initialization
            if not input_data.student_email or input_data.student_email in ["aluno@example.com", "student@example.com"]:
                input_data.student_email = self.student_email
                
            # For convenience, if session_id was not provided, use the one from initialization
            if not input_data.session_id or input_data.session_id in ["session123", "session-id", "session-123"]:
                input_data.session_id = self.session_id
                
            # For convenience, if discipline_id was not provided, use the one from initialization
            if not input_data.discipline_id or input_data.discipline_id in ["matematica", "cs101", "discipline-id"]:
                input_data.discipline_id = self.discipline_id
            
            print(f"[VECTOR_SEARCH] Searching for: {input_data.query}")
            print(f"[VECTOR_SEARCH] Content type: {input_data.content_type}")
            print(f"[VECTOR_SEARCH] Using student_email: {input_data.student_email}")
            print(f"[VECTOR_SEARCH] Using session_id: {input_data.session_id}")
            print(f"[VECTOR_SEARCH] Using discipline_id: {input_data.discipline_id}")
            
            # Map content type to specific metadata filter
            content_type_mapping = {
                "text": {"type": "text"},
                "image": {"type": "image"},
                "table": {"type": "table"}
            }
            
            # Default to text if no content type is provided
            if not input_data.content_type:
                input_data.content_type = "text"
                
            specific_metadata = content_type_mapping.get(input_data.content_type.lower(), None)
            if not specific_metadata:
                return "Invalid content type. Use 'text', 'image', or 'table'."
                
            # Perform the search
            results = self.qdrant_handler.similarity_search_with_filter(
                query=input_data.query,
                student_email=input_data.student_email,
                session_id=input_data.session_id,
                disciplina_id=input_data.discipline_id,
                k=input_data.limit,
                specific_metadata=specific_metadata
            )
            
            # Process and format results
            if not results:
                return f"No {input_data.content_type} content found matching your query."
                
            formatted_results = []
            for i, doc in enumerate(results, 1):
                content = doc.page_content
                metadata = doc.metadata
                
                # Format based on content type
                if input_data.content_type.lower() == "image":
                    # For images, just return the description (actual image retrieval handled separately)
                    image_uuid = metadata.get("image_uuid", "N/A")
                    formatted_results.append(f"Image {i}:\n- Description: {content}\n- ID: {image_uuid}")
                else:
                    # For text and tables
                    source = metadata.get("source", "Unknown source")
                    formatted_results.append(f"Result {i}:\n- Content: {content}\n- Source: {source}")
            
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            print(f"[ERROR] Vector DB search error: {str(e)}")
            traceback.print_exc()
            return f"Error searching vector database: {str(e)}"
    
    async def retrieve_image(self, image_uuid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an image from MongoDB by its UUID.
        """
        try:
            print(f"[IMAGE_RETRIEVAL] Retrieving image: {image_uuid}")
            
            # First, find the vector embedding with the image description
            doc_results = self.qdrant_handler.similarity_search_with_filter(
                query="",  # Empty query as we're filtering by ID
                student_email=self.student_email,
                session_id=self.session_id,
                disciplina_id=self.discipline_id,
                k=1,
                specific_metadata={"image_uuid": image_uuid, "type": "image"}
            )
            
            description = doc_results[0].page_content if doc_results else "No description available"
            
            # Then get the actual image data from MongoDB
            image_data = await self.mongo_manager.image_handler.get_image(image_uuid)
            
            if not image_data:
                print(f"[IMAGE_RETRIEVAL] Image not found: {image_uuid}")
                return None
                
            return {
                "image_data": image_data,
                "description": description,
                "image_uuid": image_uuid
            }
            
        except Exception as e:
            print(f"[ERROR] Image retrieval error: {str(e)}")
            traceback.print_exc()
            return None


class ReactAgent:
    """
    A ReAct agent implementation that can use vector database retrieval
    and other tools to answer educational questions for students.
    
    This agent uses a reasoning and acting (ReAct) approach to:
    1. Analyze student questions
    2. Determine the best tools to use
    3. Retrieve relevant educational context
    4. Provide personalized responses
    """
    
    def __init__(
        self,
        qdrant_handler: QdrantHandler,
        mongo_manager: MongoDatabaseManager,
        student_email: str,
        discipline_id: str,
        session_id: str,
        student_profile: Dict[str, Any]
    ):
        self.student_email = student_email
        self.discipline_id = discipline_id
        self.session_id = session_id
        self.student_profile = student_profile
        
        # Initialize retrieval tools
        self.retrieval_tools = RetrievalTools(
            qdrant_handler=qdrant_handler,
            mongo_manager=mongo_manager,
            student_email=student_email,
            discipline_id=discipline_id,
            session_id=session_id
        )
        
        # Initialize LLM with moderate temperature for educational responses
        # Use a reliable model for structured ReAct reasoning
        try:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
            print(f"[REACT_AGENT] Successfully initialized with gpt-4o-mini model")
        except Exception as e:
            print(f"[REACT_AGENT] Error loading preferred model, falling back to alternative: {str(e)}")
            # Fallback to another model if needed
            self.llm = ChatOpenAI(temperature=0.2)
        
        # Initialize agent
        self.agent_executor = self._setup_agent()
        
    def _setup_agent(self) -> AgentExecutor:
        """Setup the ReAct agent with tools and prompt."""
        
        # Define tools
        vector_search_tool = StructuredTool.from_function(
            func=self.retrieval_tools.search_vector_db,
            name="search_vector_db",
            description="""Search the vector database for information matching your query.
            
Required inputs:
- query: Your search query
- student_email: The student's email (use the value from the student profile)
- session_id: The current session ID (use the current session ID)
- discipline_id: The discipline ID (use the current discipline ID)
- content_type: The type of content to search for ('text', 'image', or 'table')
- limit: Maximum number of results to return (default: 3)

Example: Find information about matrices in the vector database."""
        )
        
        # Create YouTube and Wikipedia search tools
        youtube_tool = StructuredTool.from_function(
            func=search_youtube,
            name="search_youtube",
            description="""Search YouTube for educational videos related to the query.
            
This tool is useful when:
- You need to find visual explanations of concepts
- The student asks for video resources
- The concept is better explained with visual demonstrations
- You can't find sufficient information in the vector database

Example: Find videos about solving quadratic equations."""
        )
        
        wikipedia_tool = StructuredTool.from_function(
            func=search_wikipedia,
            name="search_wikipedia",
            description="""Search Wikipedia for factual information related to the query.
            
This tool is useful when:
- You need verified factual information about a topic
- The student asks for definitions or historical context
- You need background information not found in the vector database
- You want to provide a reliable external source

Example: Look up information about the Pythagorean theorem."""
        )
        
        tools = [
            # Vector DB search tool
            vector_search_tool,
            # Web search tools
            youtube_tool,
            wikipedia_tool
        ]
        
        # Create system prompt with educational context
        learning_style = self.student_profile.get("EstiloAprendizagem", {})
        
        system_message = f"""You are an AI tutor specialized in helping students learn effectively.

Student Profile:
- Name: {self.student_profile.get("Nome", "Student")}
- Learning Style:
  - Perception: {learning_style.get("Percepcao", "Not specified")}
  - Input: {learning_style.get("Entrada", "Not specified")}
  - Processing: {learning_style.get("Processamento", "Not specified")}
  - Understanding: {learning_style.get("Entendimento", "Not specified")}

Your task is to help the student by providing educational assistance, answering questions,
and retrieving relevant information from the available knowledge base.

IMPORTANT GUIDELINES:
1. Always prioritize retrieving information from the vector database before searching external sources
2. For educational questions, first check if the answer exists in the stored materials
3. Use YouTube or Wikipedia searches only when necessary information isn't in the vector DB
4. When suggesting learning materials, prioritize those already available in the system
5. Analyze the question carefully to determine the best source of information
6. Adapt your explanations to the student's learning style

Always think step by step to determine the best approach to the student's question.
"""
        
        # Use the default ReAct prompt to avoid template issues
        # This is much safer than trying to create a custom template
        # We'll rely on the built-in ReAct prompt format
        prompt_prefix = f"""You are an AI tutor helping students learn effectively. When students ask about specific course material, always search for it in their personal learning database.

For EVERY question, your first action must be to search the database using search_vector_db. If you get no results, try other search terms before using external tools.

When using the search_vector_db tool:
- student_email = "{self.student_email}"
- session_id = "{self.session_id}"
- discipline_id = "{self.discipline_id}"
- content_type = "text" (or "image" for visual content)
- limit = 3
"""

        # Create a custom template that follows the ReAct format but with our instructions
        system_message = f"""You are an AI tutor helping students learn effectively.

{prompt_prefix}

When using search_vector_db, make sure to include these required fields in your Action Input:
- query: The search query itself (what you're looking for)
- student_email: "{self.student_email}"
- session_id: "{self.session_id}"
- discipline_id: "{self.discipline_id}"
- content_type: "text" (or "image" for visual content)
- limit: 3
"""

        # Create the ReAct agent with the standard messages format and required variables
        react_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Add the required variables for ReAct prompt
        react_prompt = react_prompt.partial(
            tools="{tools}",
            tool_names="{tool_names}"
        )
        
        # Import formatters for ReAct agent
        from langchain.agents.format_scratchpad import format_to_openai_function_messages
        from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
        
        try:
            # Create the agent with our prompt and proper formatting
            agent = create_react_agent(
                llm=self.llm,
                tools=tools,
                prompt=react_prompt,
                output_parser=OpenAIFunctionsAgentOutputParser()
            )
            print("[REACT_AGENT] Successfully created agent with custom prompt")
        except Exception as e:
            # Fall back to a simpler prompt
            print(f"[REACT_AGENT] Error creating agent with custom prompt: {e}")
            print("[REACT_AGENT] Creating a simpler default prompt")
            
            # Create a simplified prompt with basic instruction
            simple_system = "You are a helpful AI tutor. First search for information in the vector database before answering."
            simple_prompt = ChatPromptTemplate.from_messages([
                ("system", simple_system),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            # Add the required variables for ReAct prompt
            simple_prompt = simple_prompt.partial(
                tools="{tools}",
                tool_names="{tool_names}"
            )
            
            agent = create_react_agent(
                llm=self.llm,
                tools=tools,
                prompt=simple_prompt,
                output_parser=OpenAIFunctionsAgentOutputParser()
            )
        
        # Create agent executor with increased iterations and better error handling
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,  # Limit iterations for responsiveness
            early_stopping_method="generate",  # On failure, try to generate a response anyway
            return_intermediate_steps=True  # Return the steps for debugging
        )
        
    async def process_query(
        self,
        query: str,
        chat_history: Optional[List[BaseMessage]] = None
    ) -> Dict[str, Any]:
        """
        Process a query from the student and return a response.
        """
        start_time = datetime.now()
        
        # Default empty chat history if none provided
        if chat_history is None:
            chat_history = []
            
        try:
            print(f"[REACT_AGENT] Processing query: {query}")
            
            # Execute the agent (format_scratchpad is already set in _setup_agent)
            result = await self.agent_executor.ainvoke({
                "input": query,
                "chat_history": chat_history
            })
            
            # Extract answer from the result
            answer = result.get("output", "I couldn't process your request.")
            
            # Check if we have valid steps or if the agent struggled
            steps = result.get("intermediate_steps", [])
            
            # Check for various error conditions that require fallback
            has_error = (
                not steps or  # No steps executed
                (len(steps) > 0 and "is not a valid tool" in str(steps[-1])) or  # Tool usage error
                "Invalid Format" in answer or  # Format error
                "Action Input:" in answer or  # Incomplete ReAct format
                "Missing 'Action Input:'" in answer  # Another ReAct error message
            )
            if has_error:
                print(f"[REACT_AGENT] Agent couldn't use tools properly, attempting direct answer")
                # Attempt to get a direct answer using simple LLM call
                # Use a fallback education-focused prompt for direct answers
                fallback_system_message = f"""You are an educational assistant helping a student. 
The question is related to the course topic '{self.discipline_id}'. 
Provide a concise, helpful answer based on your knowledge. 
If you don't know specific course material, explain general concepts.

Student profile information:
- Email: {self.student_email}
- Discipline: {self.discipline_id}
"""
                direct_response = await self.llm.ainvoke([
                    {"role": "system", "content": fallback_system_message},
                    {"role": "user", "content": query}
                ])
                answer = direct_response.content
            
            # Check if the answer contains image information (UUID)
            image_uuid = None
            if "image_uuid" in answer or "Image ID:" in answer:
                # Extract the image UUID using simple pattern matching
                import re
                uuid_match = re.search(r'(Image ID:|image_uuid)[:=]?\s*([a-f0-9-]+)', answer)
                if uuid_match:
                    image_uuid = uuid_match.group(2).strip()
                    
            # If we have an image UUID, retrieve the image
            image_data = None
            if image_uuid:
                image_info = await self.retrieval_tools.retrieve_image(image_uuid)
                if image_info and image_info.get("image_data"):
                    # Convert image bytes to base64 for transmission
                    try:
                        image_bytes = image_info["image_data"]
                        base64_image = base64.b64encode(image_bytes).decode('utf-8')
                        image_data = f"data:image/jpeg;base64,{base64_image}"
                        
                        # Clean up the answer by removing the image ID references
                        answer = re.sub(r'(Image ID:|image_uuid)[:=]?\s*([a-f0-9-]+)', '', answer)
                        
                        # Add image description if not already included
                        if "description" in image_info and image_info["description"] not in answer:
                            answer += f"\n\nImage description: {image_info['description']}"
                    except Exception as e:
                        print(f"[ERROR] Error processing image: {str(e)}")
                        traceback.print_exc()
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"[REACT_AGENT] Processed query in {processing_time:.2f} seconds")
            
            # Format the response
            response = {
                "answer": answer,
                "image_data": image_data,
                "processing_time": processing_time,
                "tool_calls": result.get("intermediate_steps", [])
            }
            
            return response
            
        except Exception as e:
            print(f"[ERROR] Agent execution error: {str(e)}")
            traceback.print_exc()
            return {
                "answer": f"I encountered an error while processing your request: {str(e)}",
                "image_data": None,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "error": str(e)
            }
            
    async def format_response(self, response: Dict[str, Any]) -> AIMessage:
        """
        Format the response dictionary into an AIMessage object.
        This is needed for chat controller integration.
        """
        try:
            answer = response.get("answer", "")
            image_data = response.get("image_data")
            
            if image_data:
                # Create a multimodal response with image
                content = {
                    "type": "multimodal",
                    "content": answer,
                    "image": image_data
                }
                return AIMessage(content=json.dumps(content))
            else:
                # Return plain text response
                return AIMessage(content=answer)
        except Exception as e:
            print(f"[ERROR] Error formatting response: {str(e)}")
            traceback.print_exc()
            return AIMessage(content=response.get("answer", "Error formatting response"))