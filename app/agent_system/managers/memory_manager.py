from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

class MemoryManager:
    def __init__(self):
        self.memory = InMemoryStore()

    def get_memory(self, session_id: str) -> Memory:
        return self.memory.get(session_id)

    def set_memory(self, session_id: str, memory: Memory):
        self.memory.set(session_id, memory)