# imports
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")


def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    llm_response = llm.invoke("Say, setup complete in one word,")
    print(llm_response)


if __name__ == "__main__":
    main()
