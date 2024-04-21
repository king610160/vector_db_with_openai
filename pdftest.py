from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader=PyPDFDirectoryLoader('test')
data=loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,  chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)
# cut how many parts...
print(len(text_chunks))

print(text_chunks[0])



