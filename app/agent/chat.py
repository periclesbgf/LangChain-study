from langchain_openai import ChatOpenAI
import json
from typing import Sequence, List

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from openai.types.chat import ChatCompletionMessageToolCall

# import nest_asyncio

# nest_asyncio.apply()
class ConversationHistory:
    def __init__(self):
        self.history = [
            {
                "role": "system",
                "content": """ """
            }
        ]

    def add_user_message(self, message):
        self.history.append({"role": "user", "content": message})

    def add_assistant_message(self, message):
        self.history.append({"role": "assistant", "content": message})

    def get_history(self):
        return self.history

    def remove_last_two_messages(self):
        if len(self.history) >= 2:
            self.history.pop()
            self.history.pop()


# def multiply(a: int, b: int) -> int:
#     """Multiple two integers and returns the result integer"""
#     print(a, b)
#     print("multiply")
#     return a + b

# def add(a: int, b: int) -> int:
#     """Add two integers and returns the result integer"""
#     print(a, b)
#     return a + b


# add_tool = FunctionTool.from_defaults(fn=add)
# multiply_tool = FunctionTool.from_defaults(fn=multiply)

# class YourOpenAIAgent:
#     def __init__(
#         self,
#         tools: Sequence[BaseTool] = [],
#         llm: OpenAI = OpenAI(temperature=0, model="gpt-4o-mini"),
#         chat_history: List[ChatMessage] = [],
#     ) -> None:
#         self._llm = llm
#         self._tools = {tool.metadata.name: tool for tool in tools}
#         self._chat_history = chat_history

#     def reset(self) -> None:
#         self._chat_history = []

#     def chat(self, message: str) -> str:
#         chat_history = self._chat_history
#         chat_history.append(ChatMessage(role="user", content=message))
#         tools = [
#             tool.metadata.to_openai_tool() for _, tool in self._tools.items()
#         ]

#         ai_message = self._llm.chat(chat_history, tools=tools).message
#         additional_kwargs = ai_message.additional_kwargs
#         chat_history.append(ai_message)

#         tool_calls = additional_kwargs.get("tool_calls", None)
#         # parallel function calling is now supported
#         if tool_calls is not None:
#             for tool_call in tool_calls:
#                 function_message = self._call_function(tool_call)
#                 chat_history.append(function_message)
#                 ai_message = self._llm.chat(chat_history).message
#                 chat_history.append(ai_message)
#         print(self._chat_history)
#         return ai_message.content

#     def _call_function(
#         self, tool_call: ChatCompletionMessageToolCall
#     ) -> ChatMessage:
#         id_ = tool_call.id
#         function_call = tool_call.function
#         tool = self._tools[function_call.name]
#         output = tool(**json.loads(function_call.arguments))
#         return ChatMessage(
#             name=function_call.name,
#             content=str(output),
#             role="tool",
#             additional_kwargs={
#                 "tool_call_id": id_,
#                 "name": function_call.name,
#             },
#         )

# if __name__ == "__main__":
#     agent = YourOpenAIAgent(tools=[multiply_tool, add_tool])
#     print(agent.chat("Hi"))
#     print(agent.chat("What is 2 * 4"))