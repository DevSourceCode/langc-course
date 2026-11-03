from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
import os

# load environment variables from .env file
load_dotenv()

# LangSmith Configuration
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT_NAME", "Smart Q&A Bot Project")
    print(
        f"LangSmith is configured. - Project: {os.environ['LANGSMITH_PROJECT_NAME']}")

# Schema Defination


class QAResponse (BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    confidence: str = Field(
        description="The confidence level of the answer, e.g., high, medium, low.")
    reasoning: str = Field(
        description="The reasoning or explanation behind the answer provided.")
    follow_up_questions: List[str] = Field(
        description="A list of follow-up questions related to the topic.", default_factory=list)
    sources_needed: bool = Field(
        description="Indicates whether sources are needed for the answer.", default=False)

# Bot Implementation


class SmartQABot:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        self.model = ChatOpenAI(
            model=model_name, temperature=temperature).with_structured_output(QAResponse)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are a knowledgeable and helpful assistant.
                Your Guidelines:
                    - Answer questions accurately and concisely
                    - Be honest about uncertainty - set confidence to 'low' if unsure
                    - Provide clear reasoning for your answers
                    - Suggest relevant follow-up questions
                    - Indicate if external sources would help

                    Always respond with accurate, helpful information."""),
                ("user", "{question}")
            ]
        )
        self.chain = self.prompt | self.model

    def ask_question(self, question: str) -> QAResponse:
        try:
            response = self.chain.invoke({"question": question})
            return QAResponse.model_validate(response)
        except Exception as e:
            return QAResponse(
                answer="An error occurred while processing your question.",
                confidence="low",
                reasoning=str(e),
                follow_up_questions=[],
                sources_needed=False
            )
