from langchain_core.runnables import RunnableBranch
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


def demo_basic_chain():
    """
    Demonstrates a basic chain using LangChain.
    """
    # Create a prompt template
    prompt_template = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer the following question: {question}"
    )

    # Initialize the chat model
    chat_model = init_chat_model(model="gpt-4o-mini", temperature=0.7)

    parser = StrOutputParser()

    # Define the input for the chain
    input_data = {"question": "What is the capital of France?"}

    chain = prompt_template | chat_model | parser

    response = chain.invoke({"question": input_data["question"]})

    print("Response:", response)


def demo_paralle_chain():
    """
    Demonstrates a parallel chain using LangChain.
    """
    # Create a prompt template
    summerize_prompt = ChatPromptTemplate.from_template(
        "Summerize the following text: {text}"
    )

    keywords_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Extract keywords from the following text: {text}"
    )

    sentiment_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Analyze the sentiment of the following text: {text}"
    )

    # Initialize the chat model
    chat_model = init_chat_model(model="gpt-4o-mini", temperature=0.7)

    parser = StrOutputParser()

    # Define the input for the chain
    input_data = {
        "text": "Yesterday, I started wacthing a nextflix season Lupin. It was a great experience. I really enjoyed it."}

    analyze_chains = RunnableParallel(

        summary=summerize_prompt | chat_model | parser,
        keywords=keywords_prompt | chat_model | parser,
        sentiment=sentiment_prompt | chat_model | parser,
    )

    results = analyze_chains.invoke({"text": input_data["text"]})
    print("Analysis Results")
    print(f" Summary: {results['summary']}")
    print(f" Keywords: {results['keywords']}")
    print(f" Sentiment: {results['sentiment']}")


def demo_passthrough_chain():
    """
    Demonstrates a passthrough chain using LangChain.
    """

    def fake_context(input_dict):
        return "LangChain was created by Harrison Chase. It is a framework for developing applications powered by language models."

    # Create a prompt template
    prompt_template = ChatPromptTemplate.from_template(
        "Answer the question using this context:\n{context}\n\n"
        "Question: {question}"
    )

    # Initialize the chat model
    chat_model = init_chat_model(model="gpt-4o-mini", temperature=0.7)

    parser = StrOutputParser()

    # Define the input for the chain
    input_data = {"question": "Is it great framework to learn in 2026?"}

    chain = (
        RunnableParallel(
            context=RunnableLambda(fake_context),
            question=RunnablePassthrough()
        )

        | RunnableLambda
        (
            lambda x: {"context": x["context"],
                       "question": x["question"]["question"]}
        )

        | prompt_template | chat_model | parser

    )
    response = chain.invoke({"question": input_data["question"]})

    print("Response:", response)


def demo_runnablebranch_chains():
    """
    Demonstrates a runnable branch chain using LangChain.
    """

    llm = ChatOpenAI(model="gpt-4.1-mini")
    parser = StrOutputParser()

    python_chain = (
        ChatPromptTemplate.from_template(
            "You are a Python expert. Answer: {question}"
        )
        | llm
        | parser
    )

    spark_chain = (
        ChatPromptTemplate.from_template(
            "You are an Apache Spark expert. Answer: {question}"
        )
        | llm
        | parser
    )

    default_chain = (
        ChatPromptTemplate.from_template(
            "Answer the following question: {question}"
        )
        | llm
        | parser
    )

    branch = RunnableBranch(
        (
            lambda x: x["topic"] == "python",
            python_chain
        ),
        (
            lambda x: x["topic"] == "spark",
            spark_chain
        ),
        default_chain
    )

    result = branch.invoke({
        "topic": "spark",
        "question": "What is a shuffle operation?"
    })

    print(result)


if __name__ == "__main__":
    # demo_basic_chain()
    # demo_paralle_chain()
    # demo_passthrough_chain()
    demo_runnablebranch_chains()
