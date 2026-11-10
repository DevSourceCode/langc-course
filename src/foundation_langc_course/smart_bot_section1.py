from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Literal
from dotenv import load_dotenv
from langsmith import traceable
import os

# load environment variables from .env file
load_dotenv()

# LangSmith Configuration
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "Smart Q&A Bot Project")
    print(
        f"LangSmith tracing enabled. - Project: {os.environ['LANGSMITH_PROJECT']}")

# Schema Defination


class QAResponse (BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    confidence: Literal["high", "medium", "low"] = Field(
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
            return self._error_response(e)

    @staticmethod
    def _error_response(error: Exception) -> QAResponse:
        return QAResponse(
            answer="An error occurred while processing your question.",
            confidence="low",
            reasoning=str(error),
            follow_up_questions=[],
            sources_needed=False
        )

    @traceable(name="ask_batch_questions", run_type="chain")
    def ask_batch_questions(self, questions: List[str]) -> List[QAResponse]:
        inputs = [{"question": q} for q in questions]
        responses = self.chain.batch(inputs, return_exceptions=True)
        results: List[QAResponse] = []
        for response in responses:
            if isinstance(response, Exception):
                results.append(self._error_response(response))
                continue
            try:
                results.append(QAResponse.model_validate(response))
            except Exception as e:
                results.append(self._error_response(e))
        return results


def demo_qa_bot():
    bot = SmartQABot()
    questions = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "How does photosynthesis work?",
        "What is the meaning of life?"
    ]
    responses = bot.ask_batch_questions(questions)
    for question, response in zip(questions, responses):
        print(f"Question: {question}")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Follow-up Questions: {response.follow_up_questions}")
        print(f"Sources Needed: {response.sources_needed}")
        print("-" * 40)


if __name__ == "__main__":
    demo_qa_bot()
