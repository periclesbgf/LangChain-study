from langchain_openai import ChatOpenAI

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
