from llama_index.readers.azstorage_blob import AzStorageBlobReader
from azure.storage.blob import BlobServiceClient,generate_account_sas, ResourceTypes, AccountSasPermissions
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex,StorageContext,Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer

import streamlit as st
import logging
import sys
import nest_asyncio
nest_asyncio.apply()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

api_key = "c09f91126e51468d88f57cb83a63ee36"
azure_endpoint = "https://chat-gpt-a1.openai.azure.com/"
api_version = "2023-03-15-preview"


connect_str = 'DefaultEndpointsProtocol=https;AccountName=shreemirrahrag;AccountKey=UlbcKqbBtcs0J+2esmr8AznpwdV1dF5XI+v13kY07sjJ2U8rHmWbtCUqLjb12lD7Bn9k17mWgqjd+AStB4xXcw==;EndpointSuffix=core.windows.net'
container_name = 'docs'
account_url='https://shreemirrahrag.blob.core.windows.net/docs'
blob_service_client = BlobServiceClient(account_url=account_url, credentials='UlbcKqbBtcs0J+2esmr8AznpwdV1dF5XI+v13kY07sjJ2U8rHmWbtCUqLjb12lD7Bn9k17mWgqjd+AStB4xXcw==')

llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="DanielChatGPT16k",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-3-small",
    deployment_name="text-embedding-3-small",
    api_key="c09f91126e51468d88f57cb83a63ee36",
    azure_endpoint="https://chat-gpt-a1.openai.azure.com/",
    api_version="2023-03-15-preview",
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1000

vector_store = PGVectorStore.from_params(
    database='citus',
    host='c-shreemirrah-final.6c2zuyv2zdqcrp.postgres.cosmos.azure.com',
    password='Admin123!',
    port=5432,
    user='citus',
    table_name="rag_v1",
    embed_dim=1536)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

context_str="""You're an assistant who fetches answers from the documents. Add all the necessary details."""

template = """
You are a knowledgeable and precise assistant specialized in question-answering tasks, 
particularly from documented sources. 
Your goal is to provide accurate, concise, and contextually relevant answers based on the given information.

Instructions:
Dates: When it comes to retirieving information pertaining to dates in the document(when there's discrepency), give preference to choosing the dates that are more recent, the date with the day.month, and year as well as the ones which occur more frequently in the relevant documents
Comprehension and Accuracy: Carefully read and comprehend the provided context from the documents to ensure accuracy in your response.
Truthfulness: If the context does not provide enough information to answer the question, clearly state, "I don't know."
Contextual Relevance: Ensure your answer is well-supported by the retrieved context and does not include any information beyond what is provided.

Remember if no context is provided please say you don't know the answer
Here is the question and context for you to work with:

\nQuestion: {question} \nContext: {context} \nAnswer:"""


prompt_tmpl = PromptTemplate(
    template=template,
    template_var_mappings={"query_str": "question", "context_str": "context"},
)
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5
)

rerank = SentenceTransformerRerank(
    top_n = 5, 
    model = "BAAI/bge-reranker-base"
)
postproc = MetadataReplacementPostProcessor(target_metadata_key="window")

response_synthesizer = get_response_synthesizer()

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,node_postprocessors=[rerank,postproc]
)

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template":prompt_tmpl}
)

# postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
# rerank = SentenceTransformerRerank(top_n = 2,model = "BAAI/bge-reranker-base")

# query_engine = index.as_query_engine(similarity_top_k = 6, alpha=0.5,node_postprocessors = [postproc, rerank],)

def return_response(question):
    response = query_engine.query(question)
    answer=''
    st.write(response.response)
    st.write('\n\n'+'Source:\n\n')
    source=response.metadata

    for i in source.keys():
        st.write('\nSource:  '+source[i]['document_title']+'\t')    
        file=source[i]['file_name']
        blob_client = blob_service_client.get_blob_client(container_name, file)
        blob_url = blob_client.url
        blob_url=blob_url.replace('https://shreemirrahrag.blob.core.windows.net/docs/docs','https://shreemirrahrag.blob.core.windows.net/docs')
        st.markdown(f"[{file}]({blob_url})"+"\tPage: "+source[i]['page_label']+'\n')




st.set_page_config(page_title='RAG Questions Answered Extractor')
st.header('RAG Questions Answered Extractor')
input=st.text_input('Input: ',key="input")
submit=st.button("Ask")

if submit:
    st.subheader("Implementing...")
    return_response(input)
