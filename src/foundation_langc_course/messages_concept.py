from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")


def demo_messages_concept() -> str:
    # Create a list of messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?")
    ]

    # Create a ChatOpenAI instance
    llm = init_chat_model(model="gpt-4o-mini", temperature=0.7)

    # Generate a response using the messages
    llm_response = llm.invoke(messages)
    print(f"LLM Response: {llm_response}")

    return llm_response


if __name__ == "__main__":
    demo_messages_concept()
