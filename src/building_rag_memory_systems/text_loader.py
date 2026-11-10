from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

load_dotenv()


def load_text_file(file_path: str):
    """
    Loads a text file and returns its content as a string.
    """
    loader = TextLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from {file_path}.")
    # Print the content of the first document
    print(f"Content of the first document: {documents[0].page_content}")
    print(f"Metadata of the first document: {documents[0].metadata}")
    # Print first 100 characters
    print(
        f"Content of the first document (truncated): {documents[0].page_content[:100]}...")
    print(documents)


if __name__ == "__main__":
    # Example usage
    # Replace with your text file path
    file_path = "C:\\Working Folder\\Repositories\\langc-course\\src\\building_rag_memory_systems\\sample_text.txt"
    load_text_file(file_path)
