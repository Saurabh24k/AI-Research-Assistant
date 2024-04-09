# Import essential libraries and classes
from config import *
import os
from dotenv import load_dotenv, find_dotenv
import json
import requests
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.document_loaders.url import UnstructuredURLLoader
from langchain.vectorstores import Weaviate
from langchain.embeddings import HuggingFaceEmbeddings
import weaviate

# Load environment variables
load_dotenv(find_dotenv())

class InfoResearcher:
    """
    A class designed to perform in-depth research using a variety of data sources.
    """
    def __init__(self):
        # Initialize API keys and clients
        self.search_api_key = os.getenv("SERPER_API_KEY")
        self.vector_search_api_key = os.getenv("WEAVIATE_API_KEY")
        auth_details = weaviate.AuthApiKey(api_key=self.vector_search_api_key)
        self.vector_client = weaviate.Client(
            url="https://ai-research-agent-y6v94yb0.weaviate.network",
            auth_client_secret=auth_details
        )
        
        # Setup the template, text splitter, and language model
        self.query_template = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=INPUT_VARIABLES
        )
        self.doc_splitter = RecursiveCharacterTextSplitter(
            separators=SEPARATORS,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.transformer = CTransformers(
            model=MODEL,
            model_type=MODEL_TYPE,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE
        )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDER,
            model_kwargs={'device': 'cpu'}
        )

    def find_articles(self, search_term):
        """
        Search for articles related to the given term.
        """
        endpoint = "https://google.serper.dev/search"
        payload = json.dumps({"q": search_term})

        request_headers = {
            'X-API-KEY': self.search_api_key,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", endpoint, headers=request_headers, data=payload)
        return response.json()
    
    def answer_with_research(self):
        """
        Configure and return a retrieval-based QA chain for research.
        """
        research_chain = RetrievalQA.from_chain_type(
            llm=self.transformer,
            chain_type=CHAIN_TYPE,
            retriever=self.db.as_retriever(search_kwargs=SEARCH_KWARGS),
            return_source_documents=True,
            verbose=True,
            chain_type_kwargs={"prompt": self.query_template}
        )
        return research_chain

    def extract_links(self, search_results):
        """
        Extract URLs from the search results.
        """
        links = []
        try:
            links.append(search_results["answerBox"]["link"])
        except KeyError:
            pass
        for i in range(min(3, len(search_results["organic"]))):
            links.append(search_results["organic"][i]["link"])
        return links
    
    def load_content(self, link_list):
        """
        Load the content from the given list of URLs.
        """
        content_loader = UnstructuredURLLoader(urls=link_list)
        content = content_loader.load()
        return content
    
    def execute_research(self, objective, contents):
        """
        Perform research on a given query using the provided contents.
        """
        document_list = self.doc_splitter.split_documents(contents)
        self.db = Weaviate.from_documents(document_list, self.embedding_model, client=self.vector_client, by_text=False)
        research_bot = self.answer_with_research()
        research_result = research_bot({"query": objective})
        return research_result["result"]

    def conduct_research(self, query):
        """
        Main method to conduct research for a given query.
        """
        articles_found = self.find_articles(query)
        link_list = self.extract_links(articles_found)
        loaded_content = self.load_content(link_list)
        final_answer = self.execute_research(query, loaded_content)
        return final_answer
