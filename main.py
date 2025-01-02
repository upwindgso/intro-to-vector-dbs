import os

from groq.types import embedding
from langchain_core.runnables import RunnablePassthrough

from boilerplate import load_env_files

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.schema import StrOutputParser

load_env_files()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print(" Retrieving...")

    embedding_model = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4o-mini")
                     
    query = "What is Pinecone in machine learning?"

    chain = PromptTemplate.from_template(template=query) | llm | StrOutputParser() 

    result = chain.invoke(input={})

    print(result)
    print("*****************")
    print("=================")
    print("*****************")


    # Initialize a Pinecone vector store with the specified index name and embedding model.# Initialize a Pinecone vector store with the specified index name and embedding model.
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embedding_model
    )

    # Load the retrieval QA chat prompt from the LangChain hub.
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Create a chain that combines documents using the specified language learning model (llm) and the retrieved QA chat prompt.
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    # Construct a retrieval chain by integrating the vector store retriever and the document combination chain.
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), # Use the vector store's retriever method to fetch relevant documents
        combine_docs_chain=combine_docs_chain, # Specify how retrieved documents should be combined (using the previously created chain)
    ) 
    
    # Invoke the retrieval chain with a query input to get the final result.
    result = retrieval_chain.invoke(input={"input": query})

    print(result["answer"])
    print("*****************")
    print("=================")
    print("*****************")

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end.

    {context}

    Question: {question}

    Helpful Answer:
    """

    # in the below example, the RunnablePassthrough is used to pass the chain without and transformation
    #this allows us to invoke with a string:
    # in that case of passing a string, it just passes the string like a kwarg whereever the runnablepassthrough is used.
    #
    # if we passed a dict like:
    # Prepare a dictionary with both 'context' and 'question'
    #input_data = {
    #   "context": some_context_value,
    #   "question": some_question_value
    #}
    # Invoke the chain with the prepared input
    #res = rag_chain.invoke(input_data)
    #
    #then it would pick the relvant key....ie, if we said "question": RunnablePassthrough() then it would match the "question" from the dict and pass that value to the chain.
    


    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}  
        | custom_rag_prompt
        | llm
    )

    res = rag_chain.invoke(query)  
    print(res["content"])
    print("*****************")
    print("=================")
    print("*****************")



