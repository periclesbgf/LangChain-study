import os
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

def setup_chain(text):
    """configuring chain and prompt template for the chain"""

    summary_template = """
       1. Você é um assistente virtual chamado Éden, você é capaz de receber perguntas, comandos ou afirmações.
       2. Seu papel é responder perguntas de maneira amigável.
       3. Diferencie se um texto vindo do usuário é uma pergunta, comando ou afirmação.
       4. Você possui uma lista de comandos disponíveis para serem executados.
       5. Os comandos incluem: "ligar luminária", "desligar luminária", "ligar luz", "desligar luz", "travar porta", "destravar porta",\
        "checar bomba de água", "checar sensor de temperatura", "ligar válvula", "desligar válvula", "ligar bomba de água", "desligar bomba de água".
       6. Se a entrada do usuário for um comando: Sua tarefa é determinar se a entrada de um usuário é um desses comandos específicos\
        ou algo que se relacione com esses comandos. Se for um comando,\
        retorne exatamente o comando que você entendeu que o usuário quer executar, sem alterar a estrutura e nem adicionar texto a mais.
       7. Se a entrada do usuário for uma pergunta: Sua tarefa é responder a pergunta de maneira amigável e informativa.
       8. Se a entrada do usuário for uma afirmação: Sua tarefa é responder a afirmação de maneira amigável e informativa.
       9. Se a entrada do usuário for algo que não faz sentido: Sua tarefa é responder: 'Desculpe, não entendi.'
       10. Você deve utilizar no máximo 70 palavras para responder a cada pergunta.

       EXEMPLO_1:
        USUÁRIO: "Ligue a luminária."
        ÉDEN: "ligar a luminária"

       EXEMPLO_2:
        USUÁRIO: "Qual é a temperatura atual?"
        ÉDEN: "checar sensor de temperatura"

       EXEMPLO_3:
        USUÁRIO: "Acenda a luz."
        ÉDEN: "ligar luz"

       EXEMPLO_4:
        USUÁRIO: "Quanto é 1 + 1?"
        ÉDEN: "Um mais um é igual a dois."

       EXEMPLO_4:
        USUÁRIO: "Abra a porta."
        ÉDEN: "destravar porta"

       Dado o contexto acima, responda o texto a seguir: {text}
    """

    summary_prompt_template = PromptTemplate(input_variables=["text"], template=summary_template)

    llm = ChatOpenAI(
        model="gpt-4-turbo",
        api_key=os.environ["OPENAI_API_KEY"],
        #max_tokens=70,
    )

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    print("Chain setup complete")
    return chain, text

def invoke_chain(chain, text):
    result = chain.invoke(input={"text": text})
    return result

def context_generator(question: str):
    context = """
        You are an AI language model assistant. Your task is to generate five different versions of the given user question to \
        retrieve relevant documents from a vector database. By generation multiple perspectives on the user question,\
        your goal is to help the user overcome some of the limitations of the distance-based similarity search.\
        Provide these alternative questions separated by newlines. Original question: {question}
    """
    return context

def prompt_template(question: str, context: str):
    template = """Answer the following questions based on this context:

    {context}

    Question: {question}
    """
