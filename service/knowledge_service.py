#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
vector service
"""

import os
import nltk
work_dir = '/home/ma-user/work'
nltk.data.path.append(os.path.join(work_dir, 'nltk_data'))

from service.config import LangChainCFG
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter,MarkdownTextSplitter
from langchain.document_loaders import UnstructuredFileLoader,UnstructuredMarkdownLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


class KnowledgeService(object):
    def __init__(self, config):
        self.config = config
        self.knowledge_base = None
        self.docs_path = self.config.docs_path
        self.knowledge_base_path = self.config.knowledge_base_path
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model_path)

    def init_knowledge_base(self):
        """
        初始化本地知识库向量
        """
        print('\n#####init_knowledge_base#####\n')
        docs = []
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        markdown_splitter = MarkdownTextSplitter(chunk_size=200, chunk_overlap=20)
        for doc in os.listdir(self.docs_path):
            if doc.endswith('.txt'):
                print(doc)
                loader = UnstructuredFileLoader(f'{self.docs_path}/{doc}', mode="elements")
                doc = loader.load()
                split_doc = text_splitter.split_documents(doc)
                docs.extend(split_doc)
            elif doc.endswith('.md'):
                print(doc)
                loader = UnstructuredMarkdownLoader(f'{self.docs_path}/{doc}', mode="elements")
                doc = loader.load()
                split_doc = markdown_splitter.split_documents(doc)
                docs.extend(split_doc)
                
        self.knowledge_base = FAISS.from_documents(docs, self.embeddings)

    def add_document(self, document_path):
        split_doc = []
        if document_path.endswith('.txt'):
            print(document_path)
            loader = UnstructuredFileLoader(document_path, mode="elements")
            doc = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            split_doc = text_splitter.split_documents(doc)
        elif doc.endswith('.md'):
            print(document_path)
            loader = UnstructuredMarkdownLoader(document_path, mode="elements")
            doc = loader.load()
            markdown_splitter = MarkdownTextSplitter(chunk_size=200, chunk_overlap=20)
            split_doc = markdown_splitter.split_documents(doc)
        
        if not self.knowledge_base:
            self.knowledge_base = FAISS.from_documents(split_doc, self.embeddings)
        else:
            self.knowledge_base.add_documents(split_doc)

    def load_knowledge_base(self, path):
        if path is None:
            self.knowledge_base = FAISS.load_local(self.knowledge_base_path, self.embeddings)
        else:
            self.knowledge_base = FAISS.load_local(path, self.embeddings)
        return self.knowledge_base
