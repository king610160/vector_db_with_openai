from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# set env
from dotenv import load_dotenv
load_dotenv()
OPENAI_KEY = os.getenv('openai_key')
PINECONE_KEY = os.getenv('pinecone_key')
PINECONE_ENV = os.getenv('pinecone_env')

# set enviorn
os.environ['OPENAI_API_KEY'] = OPENAI_KEY
os.environ['PINECONE_INDEX_NAME'] = PINECONE_ENV
os.environ['PINECONE_API_KEY'] = PINECONE_KEY

# load data to document
loader=PyPDFDirectoryLoader('pdf')
data=loader.load()

# split the documents into piece
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,  chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)

# embedding with openai
op_embed=OpenAIEmbeddings()

# use existed pinecone index
vs = PineconeVectorStore(embedding=op_embed)
pinevec = vs.from_documents(
    documents=text_chunks, 
    index_name=PINECONE_ENV,
    embedding=op_embed
)

# add the vector to existed pinecone
# pinevec.add_documents(text_chunks)

# can init the pinecone, to check the index and its condition
# from pinecone import Pinecone

# pc = Pinecone(api_key=PINECONE_KEY)
# # list all indexes
# pc.list_indexes()

# # list assign index
# pc.describe_index(PINECONE_ENV)


# prepare llm and qa with related modules
llm = ChatOpenAI(
    openai_api_key=OPENAI_KEY,
    model_name='gpt-3.5-turbo',  
    temperature=0.0 
)
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff",
    retriever=vs.as_retriever() 
)


import sys
while True:
  user_input = input(f"Input Prompt: ")
  if user_input == 'exit':
    print('Exiting')
    sys.exit()
  if user_input == '':
    continue
  result = qa.invoke({'query': user_input})
  print(f"Answer: {result['result']}")