from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from langchain.messages import SystemMessage, HumanMessage, AIMessage

# Load Environment Variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")


def demo_multi_model_setup(question: str, models: list[str]) -> list[AIMessage]:
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=question)
    ]

    responses = []
    for model_name in models:
        llm = init_chat_model(model=model_name, temperature=0.7)
        response = llm.invoke(messages)
        responses.append(response)

    return responses


if __name__ == "__main__":
    question = "What is the capital of France?"
    models = ["gpt-4o-mini", "gpt-4o"]
    responses = demo_multi_model_setup(question, models)
    for model_name, response in zip(models, responses):
        print(f"Response from {model_name}: {response}")
