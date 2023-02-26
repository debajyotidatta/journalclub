# -*- coding: utf-8 -*-
"""pdf_parsing_and_embedding.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17pLfFDFDvQ6USeFhwcHkGTkqWTfO2Z54
"""

# Install Requirements
!apt-get -qq install poppler-utils
# Upgrade Pillow to latest version
!pip install -q --user --upgrade pillow
# Install Python Packages
!pip install -q unstructured==0.4.6 layoutparser

!pip install unstructured[local-inference]
!pip install langchain openai pinecone-client
# upgrade to the latest, though has not been tested
# %pip install -q --upgrade unstructured layoutparser
!pip install -q "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2"

from google.colab import drive
drive.mount('/content/drive')

import nltk
import glob
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import PromptTemplate

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
all_files = glob.glob("/content/drive/MyDrive/Hackathon/Glycocholic acid/*.pdf")


os.environ["OPENAI_API_KEY"] = ""
pinecone.init(
    api_key="",  # find at app.pinecone.io
    environment="us-east1-gcp",  # next to api key in console
)

# all_files

embeddings = OpenAIEmbeddings()

import pinecone 
from langchain.vectorstores import Pinecone

index_name = "hackathon"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=40)

for i in range(len(all_files)):
  loader = UnstructuredFileLoader(all_files[i])
  ap = loader.load()
  texts = text_splitter.split_documents(ap)
  docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name)