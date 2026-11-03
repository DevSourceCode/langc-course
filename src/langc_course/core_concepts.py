from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

# Load Environment Variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")


def demo_chain_process() -> str:
    # Create a prompt template
    prompt_template = ChatPromptTemplate.from_template(
        "You are an helpful assistant. Please respond to the following question: {question}")

    # Create an output parser
    output_parser = StrOutputParser()

    # Create a ChatOpenAI instance
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Langchain LCEL
    chain = prompt_template | llm | output_parser

    # Generate a response using the prompt template and output parser
    llm_response = chain.invoke({"question": "What is the capital of France?"})
    print(f"LLM Response: {llm_response}")

    return llm_response


def demo_chain_batch_process() -> list:
    # Create a prompt template
    prompt_template = ChatPromptTemplate.from_template(
        "You are an helpful assistant. Please respond to the following question: {question}")

    # Create an output parser
    output_parser = StrOutputParser()

    # Create a ChatOpenAI instance
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Langchain LCEL
    chain = prompt_template | llm | output_parser

    # Generate responses for a batch of questions
    questions = [
        {"question": "What is the capital of France?"},
        {"question": "What is the largest ocean on Earth?"},
        {"question": "Who wrote 'Pride and Prejudice'?"}
    ]

    llm_responses = chain.batch(questions)
    print(f"LLM Batch Responses: {llm_responses}")

    return llm_responses


def demo_streaming() -> None:
    # Create a prompt template
    prompt_template = ChatPromptTemplate.from_template(
        "You are an helpful assistant. Please respond to the following question: {question}")

    # Create an output parser
    output_parser = StrOutputParser()

    # Create a ChatOpenAI instance with streaming enabled
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)

    # Langchain LCEL
    chain = prompt_template | llm | output_parser

    # Generate a response using the prompt template and output parser
    for chunk in chain.stream({"question": "What is the capital of France?"}):
        print(f"LLM Streaming Response: {chunk}")


def new_way_model_init() -> None:
    # Create a prompt template
    prompt_template = ChatPromptTemplate.from_template(
        "You are an helpful assistant. Please respond to the following question: {question}")

    # Create an output parser
    output_parser = StrOutputParser()

    # Create a ChatOpenAI instance with streaming enabled
    model = init_chat_model(model_name="gpt-4o-mini",
                            temperature=0.7)

    # Langchain LCEL
    chain = prompt_template | model | output_parser

    # Generate a response using the prompt template and output parser
    for chunk in chain.stream({"question": "What is the capital of France?"}):
        print(f"LLM Streaming Response: {chunk}")


if __name__ == "__main__":
    # demo_chain_process()
    # demo_chain_batch_process()
    demo_streaming()
