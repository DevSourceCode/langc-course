from langchain_core.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_template(
    "Tell me a {adjective} joke about {topic}.")

messages = prompt.format_messages(adjective="funny", topic="programming")

print(messages)

prompt = ChatPromptTemplate.from_messages(messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "human", "content": "Tell me a {adjective} joke about {topic}."}
])
