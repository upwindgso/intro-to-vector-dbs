import os
from boilerplate import load_env_files

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_env_files()

if __name__ == "__main__":
    print("Ingesting data...")
    
    loader = TextLoader("./mediumblog1.txt", autodetect_encoding=True)
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    print("done!")





