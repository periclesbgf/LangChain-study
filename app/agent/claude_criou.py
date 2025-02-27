import hashlib
from typing import Dict, List, Any, Optional, Union, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, Graph
import asyncio
import json
import time
import base64
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class AgentState(TypedDict):
    messages: List[BaseMessage]
    profile: Dict[str, Any]
    context: Dict[str, Any]
    plan: Dict[str, Any]
    analysis: Dict[str, Any]
    progress: Dict[str, Any]
    session_id: str
    fast_path: bool  # Flag for simple questions to bypass heavy processing

class FastTutor:
    """
    Optimized educational multi-agent system implementing the architecture in idea.md
    with a focus on performance and responsiveness.
    """
    def __init__(
        self,
        session_id: str,
        student_email: str,
        discipline_id: str,
        vector_db_handler,
        mongo_db_handler,
        student_profile: Dict[str, Any]
    ):
        self.session_id = session_id
        self.student_email = student_email
        self.discipline_id = discipline_id
        self.vector_db = vector_db_handler
        self.mongo_db = mongo_db_handler
        self.profile = student_profile

        # Tiered model approach - use different models for different complexity tasks
        self.heavy_model = ChatOpenAI(model="gpt-4o", temperature=0.2)
        self.medium_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.light_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Content cache for repeated questions
        self.context_cache = {}
        self.plan_cache = {}

        # Initialize the workflow
        self.workflow = self.create_workflow()

        # Pre-load common resources to memory
        self.preload_resources()

    def preload_resources(self):
        """Pre-load frequently used resources to reduce database access"""
        # Implement resource pre-loading logic here
        pass

    def create_workflow(self):
        """Create an optimized workflow with parallel processing where possible"""
        workflow = Graph()

        # Add nodes with clear separation of concerns
        workflow.add_node("classifier", self.question_classifier_node())
        workflow.add_node("fast_response", self.fast_response_node())
        workflow.add_node("planner", self.planner_node())
        workflow.add_node("retriever", self.retriever_node())
        workflow.add_node("teacher", self.teacher_node())
        workflow.add_node("analyzer", self.analyzer_node())

        # Add conditional edges for optimized routing
        workflow.add_conditional_edges(
            "classifier",
            self.route_by_complexity,
            {
                "fast": "fast_response",
                "complex": "planner"
            }
        )
        workflow.add_edge("planner", "retriever")
        workflow.add_edge("retriever", "teacher")
        workflow.add_edge("fast_response", "analyzer")
        workflow.add_edge("teacher", "analyzer")
        workflow.add_edge("analyzer", END)

        workflow.set_entry_point("classifier")
        return workflow.compile()

    def question_classifier_node(self):
        """Quickly classify question complexity to choose processing path"""
        CLASSIFIER_PROMPT = """
        Analyze this question and determine if it's a simple direct question that can be answered
        immediately or a complex question requiring deeper context and planning.
        Question: {question}
        
        Return ONLY "simple" or "complex" without explanation.
        """
        prompt = ChatPromptTemplate.from_template(CLASSIFIER_PROMPT)

        async def classify_question(state: AgentState) -> AgentState:
            question = state["messages"][-1].content

            # Check cache for known questions
            question_key = self._generate_cache_key(question)
            if question_key in self.plan_cache:
                new_state = state.copy()
                new_state["fast_path"] = True
                new_state["plan"] = self.plan_cache[question_key]
                return new_state

            # For very short questions, use fast path
            if len(question.split()) < 8 and "?" in question:
                new_state = state.copy()
                new_state["fast_path"] = True
                return new_state

            # Use lightweight model for classification
            try:
                response = await self.light_model.ainvoke(
                    prompt.format(question=question)
                )
                is_simple = response.content.strip().lower() == "simple"

                new_state = state.copy()
                new_state["fast_path"] = is_simple
                return new_state
            except Exception:
                # Default to complex path on error
                new_state = state.copy()
                new_state["fast_path"] = False
                return new_state

        return classify_question

    def fast_response_node(self):
        """Generate quick responses for simple questions"""
        FAST_RESPONSE_PROMPT = """
        You are an educational assistant providing a quick response to a simple question.
        
        Student profile: {profile}
        
        Respond to the question directly and concisely.
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(FAST_RESPONSE_PROMPT)

        async def generate_fast_response(state: AgentState) -> AgentState:
            start_time = time.time()

            question = state["messages"][-1].content

            response = await self.medium_model.ainvoke(
                prompt.format(
                    profile=state["profile"],
                    question=question
                )
            )

            new_state = state.copy()
            new_state["messages"] = list(state["messages"]) + [
                AIMessage(content=response.content)
            ]

            # Track metrics
            processing_time = time.time() - start_time
            new_state["analysis"] = {
                "processing_time": processing_time,
                "path": "fast",
                "complexity": "simple"
            }

            return new_state

        return generate_fast_response

    def planner_node(self):
        """Generate an educational response plan"""
        PLANNER_PROMPT = """
        Create a concise educational response plan for this student question.
        
        Student profile: {profile}
        Question: {question}
        
        Your plan should include:
        1. Key concepts to explain
        2. Approach tailored to the student's learning style
        3. A structured response outline
        
        Return a JSON object with these fields.
        """
        prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)

        async def generate_plan(state: AgentState) -> AgentState:
            question = state["messages"][-1].content

            # Check cache for similar questions
            question_key = self._generate_cache_key(question)
            if question_key in self.plan_cache:
                new_state = state.copy()
                new_state["plan"] = self.plan_cache[question_key]
                return new_state

            response = await self.medium_model.ainvoke(
                prompt.format(
                    profile=state["profile"],
                    question=question
                )
            )

            try:
                # Extract the JSON response
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                plan = json.loads(content.strip())

                # Cache the plan for future similar questions
                self.plan_cache[question_key] = plan

                new_state = state.copy()
                new_state["plan"] = plan
                return new_state
            except Exception:
                # Default plan if JSON parsing fails
                default_plan = {
                    "concepts": ["Based on your question"],
                    "approach": "direct explanation",
                    "structure": ["introduction", "explanation", "conclusion"]
                }

                new_state = state.copy()
                new_state["plan"] = default_plan
                return new_state

        return generate_plan

    def retriever_node(self):
        """Retrieve relevant context for the question"""
        async def retrieve_context(state: AgentState) -> AgentState:
            question = state["messages"][-1].content

            # Check cache for context
            context_key = self._generate_cache_key(question)
            if context_key in self.context_cache:
                new_state = state.copy()
                new_state["context"] = self.context_cache[context_key]
                return new_state

            # Get embeddings asynchronously - vector search and web search in parallel
            try:
                # Combine retrieval into a single operation
                context_results = await self._parallel_retrieval(question)

                # Cache for future use
                self.context_cache[context_key] = context_results

                new_state = state.copy()
                new_state["context"] = context_results
                return new_state
            except Exception:
                # Default empty context on error
                new_state = state.copy()
                new_state["context"] = {"text": "", "images": [], "web": ""}
                return new_state

        return retrieve_context

    async def _parallel_retrieval(self, question: str) -> Dict[str, Any]:
        """Perform parallel retrieval operations for maximum speed"""
        # Optimize by avoiding repeated question transformations
        # Instead of transforming the question three times, do it once

        # Simultaneously retrieve text, images and web results
        text_results, image_results, web_results = await asyncio.gather(
            self._retrieve_text(question),
            self._retrieve_images(question),
            self._retrieve_web_content(question)
        )

        return {
            "text": text_results,
            "images": image_results,
            "web": web_results
        }

    async def _retrieve_text(self, question: str) -> str:
        """Retrieve text content from vector database"""
        try:
            results = await self.vector_db.similarity_search_async(
                query=question,
                student_email=self.student_email,
                session_id=self.session_id,
                discipline_id=self.discipline_id,
                k=3,  # Limit to top 3 results for speed
                filter={"type": "text"}
            )
            return "\n".join([doc.page_content for doc in results]) if results else ""
        except Exception:
            return ""

    async def _retrieve_images(self, question: str) -> List[Dict[str, Any]]:
        """Retrieve image content from vector database"""
        try:
            results = await self.vector_db.similarity_search_async(
                query=question,
                student_email=self.student_email,
                session_id=self.session_id,
                discipline_id=self.discipline_id,
                k=1,  # Just one image for speed
                filter={"type": "image"}
            )

            if not results:
                return []

            image_results = []
            for doc in results:
                image_uuid = doc.metadata.get("image_uuid")
                if image_uuid:
                    # Get image with minimal processing
                    image_data = await self.mongo_db.get_image(image_uuid)
                    if image_data:
                        image_results.append({
                            "description": doc.page_content,
                            "image_data": image_data
                        })

            return image_results
        except Exception:
            return []

    async def _retrieve_web_content(self, question: str) -> str:
        """Retrieve content from web sources if needed"""
        # This would integrate with your existing WebSearchTools
        # Simplified for this example
        return ""

    def teacher_node(self):
        """Generate educational response based on context and plan"""
        TEACHING_PROMPT = """
        You are an educational tutor providing a personalized response.
        
        Student profile: {profile}
        Response plan: {plan}
        
        Context:
        {context}
        
        Question: {question}
        
        Respond following the plan structure and tailored to the student's learning style.
        Be concise and focused.
        """
        prompt = ChatPromptTemplate.from_template(TEACHING_PROMPT)

        async def generate_response(state: AgentState) -> AgentState:
            start_time = time.time()

            question = state["messages"][-1].content
            profile = state["profile"]
            plan = state["plan"]
            context = state["context"]

            # Extract and prepare the most relevant context
            text_context = context.get("text", "")

            # Limit context size for performance
            if len(text_context) > 2000:
                text_context = text_context[:2000] + "..."

            # Add image descriptions if available
            image_descriptions = ""
            for image in context.get("images", []):
                image_descriptions += f"Image: {image.get('description', '')}\n"

            # Combine contexts
            combined_context = text_context + "\n" + image_descriptions

            # Generate response with the heavy model
            response = await self.heavy_model.ainvoke(
                prompt.format(
                    profile=profile,
                    plan=plan,
                    context=combined_context,
                    question=question
                )
            )

            # Check if we need to include images in response
            images = context.get("images", [])
            if images and len(images) > 0:
                # We have images to include
                image_data = images[0].get("image_data")
                if image_data:
                    # Create multimodal response
                    try:
                        # Convert bytes to base64
                        base64_image = base64.b64encode(image_data).decode('utf-8')
                        multimodal_content = {
                            "type": "multimodal",
                            "content": response.content,
                            "image": f"data:image/jpeg;base64,{base64_image}"
                        }
                        response = AIMessage(content=json.dumps(multimodal_content))
                    except Exception:
                        # Fallback to text-only on error
                        pass

            # Update state
            new_state = state.copy()
            new_state["messages"] = list(state["messages"]) + [response]
            new_state["analysis"] = {
                "processing_time": time.time() - start_time,
                "path": "complex",
                "complexity": "complex"
            }

            return new_state

        return generate_response

    def analyzer_node(self):
        """Analyze interaction and update student progress"""
        async def analyze_interaction(state: AgentState) -> AgentState:
            # Skip heavy analysis for fast path responses
            if state.get("fast_path", False):
                return state

            question = state["messages"][-2].content  # User's question
            answer = state["messages"][-1]  # System's answer

            # Simplified progress tracking - could be expanded
            # Store progress update asynchronously to avoid blocking
            asyncio.create_task(
                self._update_progress(
                    session_id=self.session_id,
                    question=question,
                    answer=answer,
                    processing_time=state.get("analysis", {}).get("processing_time", 0)
                )
            )

            return state

        return analyze_interaction

    async def _update_progress(self, session_id: str, question: str, answer: Any, processing_time: float):
        """Update progress data asynchronously"""
        try:
            await self.mongo_db.update_progress({
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "interaction": {
                    "question": question,
                    "response_type": "complex" if processing_time > 1.0 else "simple",
                    "processing_time": processing_time
                }
            })
        except Exception:
            # Log error but don't block main workflow
            pass

    def route_by_complexity(self, state: AgentState) -> str:
        """Route based on question complexity"""
        return "fast" if state.get("fast_path", False) else "complex"

    def _generate_cache_key(self, question: str) -> str:
        """Generate a simple cache key from the question"""
        # Simple normalization and hashing
        normalized = question.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    async def invoke(self, question: str) -> Dict[str, Any]:
        """Main entry point to invoke the tutor"""
        start_time = time.time()

        # Create initial state
        initial_state = AgentState(
            messages=[HumanMessage(content=question)],
            profile=self.profile,
            context={},
            plan={},
            analysis={},
            progress={},
            session_id=self.session_id,
            fast_path=False
        )

        # Run the workflow
        try:
            result = await self.workflow.ainvoke(initial_state)

            # Add timing information
            execution_time = time.time() - start_time
            print(f"[TUTOR] Total execution time: {execution_time:.2f} seconds")

            return {
                "messages": result["messages"],
                "execution_time": execution_time
            }
        except Exception as e:
            # Error handling with fallback response
            print(f"[TUTOR] Error in workflow: {str(e)}")
            return {
                "messages": [
                    HumanMessage(content=question),
                    AIMessage(content="I apologize, but I encountered an error processing your request. Could you please try again?")
                ],
                "error": str(e)
            }