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

blob_service_client_1 = BlobServiceClient(account_url='https://shreemirrahrag.blob.core.windows.net/document1', credentials='UlbcKqbBtcs0J+2esmr8AznpwdV1dF5XI+v13kY07sjJ2U8rHmWbtCUqLjb12lD7Bn9k17mWgqjd+AStB4xXcw==')
blob_service_client_2 = BlobServiceClient(account_url='https://shreemirrahrag.blob.core.windows.net/document2', credentials='UlbcKqbBtcs0J+2esmr8AznpwdV1dF5XI+v13kY07sjJ2U8rHmWbtCUqLjb12lD7Bn9k17mWgqjd+AStB4xXcw==')
blob_service_client_3 = BlobServiceClient(account_url='https://shreemirrahrag.blob.core.windows.net/document3', credentials='UlbcKqbBtcs0J+2esmr8AznpwdV1dF5XI+v13kY07sjJ2U8rHmWbtCUqLjb12lD7Bn9k17mWgqjd+AStB4xXcw==')
blob_service_client_4 = BlobServiceClient(account_url='https://shreemirrahrag.blob.core.windows.net/document4', credentials='UlbcKqbBtcs0J+2esmr8AznpwdV1dF5XI+v13kY07sjJ2U8rHmWbtCUqLjb12lD7Bn9k17mWgqjd+AStB4xXcw==')

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

vector_store_1 = PGVectorStore.from_params(
    database='citus',
    host='c-shreemirrah-final.6c2zuyv2zdqcrp.postgres.cosmos.azure.com',
    password='Admin123!',
    port=5432,
    user='citus',
    table_name="rag_document1",
    embed_dim=1536)

vector_store_2 = PGVectorStore.from_params(
    database='citus',
    host='c-shreemirrah-final.6c2zuyv2zdqcrp.postgres.cosmos.azure.com',
    password='Admin123!',
    port=5432,
    user='citus',
    table_name="rag_document2_correct_v1",
    embed_dim=1536)

vector_store_3 = PGVectorStore.from_params(
    database='citus',
    host='c-shreemirrah-final.6c2zuyv2zdqcrp.postgres.cosmos.azure.com',
    password='Admin123!',
    port=5432,
    user='citus',
    table_name="rag_document3",
    embed_dim=1536)

vector_store_4 = PGVectorStore.from_params(
    database='citus',
    host='c-shreemirrah-final.6c2zuyv2zdqcrp.postgres.cosmos.azure.com',
    password='Admin123!',
    port=5432,
    user='citus',
    table_name="rag_document4",
    embed_dim=1536)


context_str="""You're an assistant who fetches answers from the documents. Add all the necessary details."""

template = """
You are a knowledgeable and precise assistant specialized in question-answering tasks, 
particularly from documented sources. 
Your goal is to provide accurate, concise, and contextually relevant answers based on the given information.

Instructions:
Parties Involved: when referring to the parties involved, find the 'specific names' of the parties along with their role and not just their role. For example, don't refer to parties by just 'Licensee' use '<company name> (Licensee)'
Addresses: Addresses are crucial when it comes to documents. Hence, keep a track of addresses and the companies which they correspond to and retreive them accurately. Even if the term 'address' is not explicitly mentioned, addresses are of the format: '<Number> Place Street State-Code(Eg: CA, PA etc), 5-digit Pincode' Eg: 40 Pacifica, Suite 900 Trvine, CA 92618 
Dates: When it comes to retirieving information pertaining to dates in the document, look into the context associated with the dates. For example, you might have to retreive the date when the agreement is signed, In this case you'd have to look for the date that's associated with signature(Look for places such as By: <company-name> Title:<title> Date:date of signature). In other cases, for example, you'd have to find the effective date for the agreement. In such cases you'd have find the dates that correspond to this keyword 'effective'. Use these clues to correctly return the associated dates
Date of signature:(Look for places such as By: <company-name> Title:<title> Date:date of signature). signature dayes are mostly of the fomrat: __/__/____
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


rerank = SentenceTransformerRerank(top_n = 5, model = "BAAI/bge-reranker-base"    )
postproc = MetadataReplacementPostProcessor(target_metadata_key="window")

response_synthesizer = get_response_synthesizer()

def return_response(question):
    st.markdown('Document 1: BPP_Bright final agreement_SPB\n')
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store_1)
    retriever = VectorIndexRetriever(index=index,similarity_top_k=5)

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,node_postprocessors=[rerank,postproc])

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template":prompt_tmpl}
    )

    response = query_engine.query(question)
    st.write(response.response)
    st.write('\n\n'+'Source:\n\n')
    source=response.metadata

    for i in source.keys():
        st.write('\nSource:  '+source[i]['document_title']+'\t')    
        file=source[i]['file_name']
        blob_client = blob_service_client_1.get_blob_client('document1', file)
        blob_url = blob_client.url
        blob_url=blob_url.replace('https://shreemirrahrag.blob.core.windows.net/document1/document1','https://shreemirrahrag.blob.core.windows.net/document1')
        st.markdown(f"[{file}]({blob_url})"+"\tPage: "+source[i]['page_label']+'\n')
    st.markdown('\n\nDocument 2: Bright CoreLogic Infonet Agreement\n')
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store_2)
    retriever = VectorIndexRetriever(index=index,similarity_top_k=5)

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,node_postprocessors=[rerank,postproc])

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template":prompt_tmpl}
    )

    response = query_engine.query(question)
    answer=''
    st.write(response.response)
    st.write('\n\n'+'Source:\n\n')
    source=response.metadata

    for i in source.keys():
        st.write('\nSource:  '+source[i]['document_title']+'\t')    
        file=source[i]['file_name']
        blob_client = blob_service_client_1.get_blob_client('document2', file)
        blob_url = blob_client.url
        blob_url=blob_url.replace('https://shreemirrahrag.blob.core.windows.net/document2/document2','https://shreemirrahrag.blob.core.windows.net/document2')
        st.markdown(f"[{file}]({blob_url})"+"\tPage: "+source[i]['page_label']+'\n')    

    st.markdown('\n\nDocument 3: CC_Bright_Clear Capital Access and License Agreement\n')
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store_3)
    retriever = VectorIndexRetriever(index=index,similarity_top_k=5)

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,node_postprocessors=[rerank,postproc])

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template":prompt_tmpl}
    )

    response = query_engine.query(question)
    st.write(response.response)
    st.write('\n\n'+'Source:\n\n')
    source=response.metadata

    for i in source.keys():
        st.write('\nSource:  '+source[i]['document_title']+'\t')    
        file=source[i]['file_name']
        blob_client = blob_service_client_1.get_blob_client('document3', file)
        blob_url = blob_client.url
        blob_url=blob_url.replace('https://shreemirrahrag.blob.core.windows.net/document3/document3','https://shreemirrahrag.blob.core.windows.net/document3')
        st.markdown(f"[{file}]({blob_url})"+"\tPage: "+source[i]['page_label']+'\n')

    st.markdown('\n\nDocument 4: Signed ShowingTime for the MLS Agreement with Bright MLS\n')
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store_4)
    retriever = VectorIndexRetriever(index=index,similarity_top_k=5)

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,node_postprocessors=[rerank,postproc])

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template":prompt_tmpl}
    )

    response = query_engine.query(question)
    st.write(response.response)
    st.write('\n\n'+'Source:\n\n')
    source=response.metadata

    for i in source.keys():
        st.write('\nSource:  '+source[i]['document_title']+'\t')    
        file=source[i]['file_name']
        blob_client = blob_service_client_1.get_blob_client('document4', file)
        blob_url = blob_client.url
        blob_url=blob_url.replace('https://shreemirrahrag.blob.core.windows.net/document4/document4','https://shreemirrahrag.blob.core.windows.net/document4')
        st.markdown(f"[{file}]({blob_url})"+"\tPage: "+source[i]['page_label']+'\n')

st.set_page_config(page_title='RAG Questions Answered Extractor')
st.header('RAG Questions Answered Extractor')
input=st.text_input('Input: ',key="input")
submit=st.button("Ask")

if submit:
    st.subheader("Implementing...")
    return_response(input)
