from chains.chain_setup import CommandChain, SQLChain, AnswerChain, ClassificationChain, SQLSchoolChain, DefaultChain
from database.query import execute_query
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


from fastapi.logger import logger



from utils import OPENAI_API_KEY, CODE

def code_confirmation(code):
    if code == CODE:
        return True
    else:
        return False

def build_chain(text):
    chain = CommandChain(api_key=OPENAI_API_KEY)
    response = chain.setup_chain(text=text)
    return response

def build_sql_chain(text):
    chain = SQLChain(api_key=OPENAI_API_KEY)
    response = chain.setup_sql_chain(text=text)
    query_prepared = prepare_query(response)
    print(query_prepared)
    data = execute_query(query_prepared)
    print("data: ", data)
    # if len(data) >= 300:
    #     return "Data too large to process"
    awnser = AnswerChain(api_key=OPENAI_API_KEY)
    response = awnser.setup_chain(user_question=text, data=data, query=query_prepared)

    return response, query_prepared

def route_chooser(info, isFile):
    if "consultarbancodedados" in info.lower():
        sql_chain = SQLChain(api_key=OPENAI_API_KEY)
        return sql_chain
    elif "comando" in info.lower():
        command_chain = CommandChain(api_key=OPENAI_API_KEY)
        return command_chain
    elif "outros" in info.lower():
        default_chain = DefaultChain(api_key=OPENAI_API_KEY)
        return default_chain

def route_request(text, file=None):
    try:
        if file:
            file_content = file.file.read()
            text = text + " " + file_content.decode("utf-8")
            


        classification_chain = ClassificationChain(api_key=OPENAI_API_KEY)
        route = classification_chain.setup_chain(text=text)

        logger.critical(f"Route: {route}")

        if route == "ConsultarBancoDeDados":
            logger.critical("Setting up SQL chain DB.")

            sql_chain = SQLSchoolChain(api_key=OPENAI_API_KEY)
            sql_query, important_tables = sql_chain.setup_chain(text=text)

            logger.critical(f"Generated SQL query: {sql_query}")
            logger.critical(f"Important tables: {important_tables}")
            logger.critical("Executing SQL query.")


            data = execute_query(sql_query)


            logger.critical(f"Data from query execution: {data}")
            logger.critical("Generating final response.")

            response = sql_chain.output_chain(
                user_question=text,
                data=data,
                query=sql_query,
                importantTables=important_tables
            )
            logger.critical(f"Final response: {response}")
            return response, sql_query

        elif route == "Comando":
            logger.critical("Setting up Language chain.")
            command_chain = CommandChain(api_key=OPENAI_API_KEY)
            response = command_chain.setup_chain(text=text)
            logger.critical(f"Generated response: {response}")
            return response

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return str(e), None

