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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAI as langchain_azure
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

# Few Shot Examples
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        few_shot_prompt,("user", "{question}"),])
question_gen = prompt | AzureChatOpenAI(temperature=0,deployment_name="DanielChatGPT16k",api_key="c09f91126e51468d88f57cb83a63ee36",api_version="2023-03-15-preview",azure_endpoint="https://chat-gpt-a1.openai.azure.com/") | StrOutputParser()



vector_store_1 = PGVectorStore.from_params(
    database='citus',
    host='c-shreemirrah-final.6c2zuyv2zdqcrp.postgres.cosmos.azure.com',
    password='Admin123!',
    port=5432,
    user='citus',
    table_name="document_1",
    embed_dim=1536)

vector_store_2 = PGVectorStore.from_params(
    database='citus',
    host='c-shreemirrah-final.6c2zuyv2zdqcrp.postgres.cosmos.azure.com',
    password='Admin123!',
    port=5432,
    user='citus',
    table_name="document_2",
    embed_dim=1536)

vector_store_3 = PGVectorStore.from_params(
    database='citus',
    host='c-shreemirrah-final.6c2zuyv2zdqcrp.postgres.cosmos.azure.com',
    password='Admin123!',
    port=5432,
    user='citus',
    table_name="documents_3_v1",
    embed_dim=1536)

vector_store_4 = PGVectorStore.from_params(
    database='citus',
    host='c-shreemirrah-final.6c2zuyv2zdqcrp.postgres.cosmos.azure.com',
    password='Admin123!',
    port=5432,
    user='citus',
    table_name="documents_4_v1",
    embed_dim=1536)

context_str="""You're an assistant who fetches answers from the documents. Add all the necessary details."""

template = """
You are a knowledgeable and precise assistant specialized in question-answering tasks, 
particularly from documented sources. 
Your goal is to provide accurate, concise, and contextually relevant answers based on the given information.

Instructions:

Answer the following question using the provided context. Make sure your answer is concise, accurate, and contextually relevant.
Parties involved in the agreement: Keep in mind that the parties referred to the agreement are actually the 'Companies' involved in the particular agreement. Hence, while referring to 'parties', generate the company names that play a role in the agreement. Never mention solely the role, such as 'Licensee' or 'Licensor' these are not the companies' names. Always mention the "Companies' name"
Dates: When it comes to retrieving information pertaining to dates in the document, look for the event/term associated with each date. For example, effective date is associated with the keyword 'effective date' or any of its synonyms and you must find the date corresponding to that. Similarly, it's important to find dates of signature in the document. They're usually(not always) denoted in the format __/__/____
Address: An address is of the following format: <Number> <Street name> <Area name> <Pin code> for example: 660 American Avenue, King of Prussia, PA, 19406 
Comprehension and Accuracy: Carefully read and comprehend the provided context from the documents to ensure accuracy in your response.
Truthfulness: If the context does not provide enough information to answer the question, clearly state, "I don't know."
Contextual Relevance: Ensure your answer is well-supported by the retrieved context and does not include any information beyond what is provided.

If asked any question like: "Who are the parties to the agreement?": answer the question for "Who are the companies to the agreement?"

Remember if no context is provided please say you don't know the answer
Here is the question and context for you to work with:

\nQuestion: {question} \nContext: {context} \nAnswer:"""


prompt_tmpl = PromptTemplate(template=template,template_var_mappings={"query_str": "question", "context_str": "context"},)


rerank = SentenceTransformerRerank(top_n = 5, model = "BAAI/bge-reranker-base")
postproc = MetadataReplacementPostProcessor(target_metadata_key="window")

response_synthesizer = get_response_synthesizer()

def check(text):
    llm_check=langchain_azure(deployment_name="turbo-instruct",api_version=api_version,api_key=api_key,azure_endpoint="https://chat-gpt-a1.openai.azure.com/")
    prompt="""Return a response with only either of the two words: 'True' or 'False'.
    You are given with the text that is a response from an RAG pipeline. Your job is to tell me whether the answer was successfully retrievd or not.
    Given the text: {text}:
    Your response must be False if the text says that the answer couldn't be retrieved. In such cases, text could contain sentences like 'not provided in the given context', 'I don't know' and the like, in which case your response should be- False.
    And your response must be True, if the text actually contains any valid answers.
    """
    return llm_check.invoke(prompt)
    

def return_response(question):
    st.write("Document: 1 \t BPP_Bright final agreement SPB signature")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store_1)
    retriever = VectorIndexRetriever(index=index,similarity_top_k=5)
    query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,node_postprocessors=[rerank,postproc])

    query_engine.update_prompts(
    {"response_synthesizer:text_qa_template":prompt_tmpl})
    response = query_engine.query(question)
    check_ans=check(response.response)
    if check_ans=='False':
        st.write("\n"+check_ans+"\n")
        new_question=question_gen.invoke({"question": question})
        response = query_engine.query(new_question)
        st.write(response.response)
        st.write('\n\n'+'Source:\n\n')
        source=response.metadata
        page_label=[]
        for i in source.keys():
            #st.write('\nSource:  '+source[i]['document_title']+'\t')    
            file=source[i]['file_name']
            page_label.append(source[i]['page_label'])
        blob_client = blob_service_client_1.get_blob_client('document1', file)
        blob_url = blob_client.url
        blob_url=blob_url.replace('https://shreemirrahrag.blob.core.windows.net/document1/document1','https://shreemirrahrag.blob.core.windows.net/document1')
        file=source[i]['file_name']
        ans=''
        page_label=list(set(page_label))
        for i in range(len(page_label)):
            ans+=str(page_label[i])+','if i!=len(page_label)-1 else str(page_label[i])
        st.markdown(f"[{file}]({blob_url})"+"\tPage(s): "+ans+'\n')
    else:
        st.write("\n"+check_ans+"\n")
        st.write(response.response)
        st.write('\n\n'+'Source:\n\n')
        source=response.metadata
        page_label=[]
        for i in source.keys():
            #st.write('\nSource:  '+source[i]['document_title']+'\t')    
            file=source[i]['file_name']
            page_label.append(source[i]['page_label'])
        blob_client = blob_service_client_1.get_blob_client('document1', file)
        blob_url = blob_client.url
        blob_url=blob_url.replace('https://shreemirrahrag.blob.core.windows.net/document1/document1','https://shreemirrahrag.blob.core.windows.net/document1')
        file=source[i]['file_name']
        ans=''
        page_label=list(set(page_label))
        for i in range(len(page_label)):
            ans+=str(page_label[i])+','if i!=len(page_label)-1 else str(page_label[i])
        st.markdown(f"[{file}]({blob_url})"+"\tPage(s): "+ans+'\n')

    st.write("Document: 2 \t Bright CoreLogic Infonet Agreement (Executed)")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store_2)
    retriever = VectorIndexRetriever(index=index,similarity_top_k=5)
    query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,node_postprocessors=[rerank,postproc])

    query_engine.update_prompts(
    {"response_synthesizer:text_qa_template":prompt_tmpl})
    response = query_engine.query(question)
    check_ans=check(response.response)
    if check_ans=='False':
        st.write("\n"+check_ans+"\n")
        new_question=question_gen.invoke({"question": question})
        response = query_engine.query(new_question)        
        st.write(response.response)
        st.write('\n\n'+'Source:\n\n')
        source=response.metadata
        page_label=[]
        for i in source.keys():
            #st.write('\nSource:  '+source[i]['document_title']+'\t')    
            file=source[i]['file_name']
            page_label.append(source[i]['page_label'])
        blob_client = blob_service_client_2.get_blob_client('document2', file)
        blob_url = blob_client.url
        blob_url=blob_url.replace('https://shreemirrahrag.blob.core.windows.net/document2/document2','https://shreemirrahrag.blob.core.windows.net/document2')
        file=source[i]['file_name']
        ans=''
        page_label=list(set(page_label))
        for i in range(len(page_label)):
            ans+=str(page_label[i])+','if i!=len(page_label)-1 else str(page_label[i])
        st.markdown(f"[{file}]({blob_url})"+"\tPage(s): "+ans+'\n')
    else:
        st.write("\n"+check_ans+"\n")
        st.write(response.response)
        st.write('\n\n'+'Source:\n\n')
        source=response.metadata
        page_label=[]
        for i in source.keys():
            #st.write('\nSource:  '+source[i]['document_title']+'\t')    
            file=source[i]['file_name']
            page_label.append(source[i]['page_label'])
        blob_client = blob_service_client_2.get_blob_client('document2', file)
        blob_url = blob_client.url
        blob_url=blob_url.replace('https://shreemirrahrag.blob.core.windows.net/document2/document2','https://shreemirrahrag.blob.core.windows.net/document2')
        file=source[i]['file_name']
        ans=''
        page_label=list(set(page_label))
        for i in range(len(page_label)):
            ans+=str(page_label[i])+','if i!=len(page_label)-1 else str(page_label[i])
        st.markdown(f"[{file}]({blob_url})"+"\tPage(s): "+ans+'\n')


    st.write("Document: 3 \t CC_Bright_Clear Capital Access and License Agreement")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store_3)
    retriever = VectorIndexRetriever(index=index,similarity_top_k=5)
    query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,node_postprocessors=[rerank,postproc])

    query_engine.update_prompts(
    {"response_synthesizer:text_qa_template":prompt_tmpl})
    response = query_engine.query(question)
    check_ans=check(response.response)
    if check_ans=='False':
        st.write("\n"+check_ans+"\n")
        new_question=question_gen.invoke({"question": question})
        response = query_engine.query(new_question)
        st.write(response.response)
        st.write('\n\n'+'Source:\n\n')
        source=response.metadata
        page_label=[]
        for i in source.keys():
            #st.write('\nSource:  '+source[i]['document_title']+'\t')    
            file=source[i]['file_name']
            page_label.append(source[i]['page_label'])
        blob_client = blob_service_client_3.get_blob_client('document3', file)
        blob_url = blob_client.url
        blob_url=blob_url.replace('https://shreemirrahrag.blob.core.windows.net/document3/document3','https://shreemirrahrag.blob.core.windows.net/document3')
        file=source[i]['file_name']
        ans=''
        page_label=list(set(page_label))
        for i in range(len(page_label)):
            ans+=str(page_label[i])+','if i!=len(page_label)-1 else str(page_label[i])
        st.markdown(f"[{file}]({blob_url})"+"\tPage(s): "+ans+'\n')
    else:
        st.write("\n"+check_ans+"\n")
        st.write(response.response)
        st.write('\n\n'+'Source:\n\n')
        source=response.metadata
        page_label=[]
        for i in source.keys():
            #st.write('\nSource:  '+source[i]['document_title']+'\t')    
            file=source[i]['file_name']
            page_label.append(source[i]['page_label'])
        blob_client = blob_service_client_3.get_blob_client('document3', file)
        blob_url = blob_client.url
        blob_url=blob_url.replace('https://shreemirrahrag.blob.core.windows.net/document3/document3','https://shreemirrahrag.blob.core.windows.net/document3')
        file=source[i]['file_name']
        ans=''
        page_label=list(set(page_label))
        for i in range(len(page_label)):
            ans+=str(page_label[i])+','if i!=len(page_label)-1 else str(page_label[i])
        st.markdown(f"[{file}]({blob_url})"+"\tPage(s): "+ans+'\n')        
    st.write("Document: 4 \t Signed ShowingTime for the MLS Agreement with Bright MLS")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store_4)
    retriever = VectorIndexRetriever(index=index,similarity_top_k=5)
    query_engine = RetrieverQueryEngine(retriever=retriever,response_synthesizer=response_synthesizer,node_postprocessors=[rerank,postproc])
    query_engine.update_prompts(
    {"response_synthesizer:text_qa_template":prompt_tmpl})
    response = query_engine.query(question)
    check_ans=check(response.response)
    if check_ans=='False':
        st.write("\n"+check_ans+"\n")
        new_question=question_gen.invoke({"question": question})
        response = query_engine.query(new_question)        
        st.write(response.response)
        st.write('\n\n'+'Source:\n\n')
        source=response.metadata
        page_label=[]
        for i in source.keys():
            #st.write('\nSource:  '+source[i]['document_title']+'\t')    
            file=source[i]['file_name']
            page_label.append(source[i]['page_label'])
        blob_client = blob_service_client_4.get_blob_client('document4', file)
        blob_url = blob_client.url
        blob_url=blob_url.replace('https://shreemirrahrag.blob.core.windows.net/document4/document4','https://shreemirrahrag.blob.core.windows.net/document4')
        file=source[i]['file_name']
        ans=''
        page_label=list(set(page_label))
        for i in range(len(page_label)):
            ans+=str(page_label[i])+','if i!=len(page_label)-1 else str(page_label[i])
        st.markdown(f"[{file}]({blob_url})"+"\tPage(s): "+ans+'\n')
    else:
        st.write("\n"+check_ans+"\n")
        st.write(response.response)
        st.write('\n\n'+'Source:\n\n')
        source=response.metadata
        page_label=[]
        for i in source.keys():
            #st.write('\nSource:  '+source[i]['document_title']+'\t')    
            file=source[i]['file_name']
            page_label.append(source[i]['page_label'])
        blob_client = blob_service_client_4.get_blob_client('document4', file)
        blob_url = blob_client.url
        blob_url=blob_url.replace('https://shreemirrahrag.blob.core.windows.net/document4/document4','https://shreemirrahrag.blob.core.windows.net/document4')
        file=source[i]['file_name']
        ans=''
        page_label=list(set(page_label))
        for i in range(len(page_label)):
            ans+=str(page_label[i])+','if i!=len(page_label)-1 else str(page_label[i])
        st.markdown(f"[{file}]({blob_url})"+"\tPage(s): "+ans+'\n')        
st.set_page_config(page_title='RAG Questions Answered Extractor')
st.header('RAG Questions Answered Extractor')
input=st.text_input('Input: ',key="input")
submit=st.button("Ask")

if submit:
    st.subheader("Implementing...")
    return_response(input)
