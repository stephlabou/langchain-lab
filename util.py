import yaml
import bs4
import re
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

def load_config(path):
    """
    Loads the config yaml from the specified path

    args:
        path - Complete path of the config yaml file to be loaded
    returns:
        yaml - yaml object containing the config file
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)

class RAG():
    def __init__(self, config):
        self.config = config
        self.llm = self.create_llm(config)
        self.embeddings = self.create_embeddings(config)
        self.prompt = None
        self.docs = None
        self.split_docs = None
        self.retriever = None
        self.called = False
        
    def __call__(self, web_path, use_strainer=False, strainer_class=None, split=True, chunk_size=1000, chunk_overlap=200, add_start_index=True, search_type="similarity", topk=2):
        self.load_webpage(web_path, use_strainer, strainer_class)
        self.split_documents(chunk_size, chunk_overlap, add_start_index)
        self.set_retriever(search_type, topk)
        self.create_custom_prompt()
        self.custom_rag_chain = (
                                    {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
                                    | self.prompt
                                    | self.llm
                                    # | StrOutputParser()
                                )
        self.called = True
    
    def clear():
        self.prompt = None
        self.docs = None
        self.split_docs = None
        self.retriever = None
        self.called = False

    def ask(self, question):
        if not self.called:
            assert False, "RAG not initialized with context. Please provide web paths"
        
        # for chunk in self.custom_rag_chain.stream(question):
        #     print(chunk, end="", flush=True)
        answer = self.custom_rag_chain.invoke(question)
        
        # cleaning up the answer 
        if "Helpful Answer" in answer:
            return re.search(r"Helpful Answer:\s*(.*)", answer).group(1)
        return answer
    
    def load_webpage(self, web_path, use_strainer, strainer_class):
        loader = None
        if use_strainer:
            bs4_strainer = bs4.SoupStrainer(class_=(strainer_class))
            loader = WebBaseLoader(
                web_paths=(web_path),
                bs_kwargs={"parse_only": bs4_strainer},
            )
        else: 
            loader = WebBaseLoader(web_paths=(web_path))
        self.docs = loader.load()
    
    def split_documents(self, chunk_size, chunk_overlap, add_start_index):
        # split document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=add_start_index
        )
        self.split_docs = text_splitter.split_documents(self.docs)
    
    def set_retriever(self, search_type, topk):
        vectorstore = FAISS.from_documents(documents=self.split_docs, embedding=self.embeddings)
        retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": topk})
        self.retriever = retriever

    def create_llm(self, config):
        llm = HuggingFaceHub(
            repo_id=config['llm_repo_id'], 
            task=config['llm_task'],
            model_kwargs={"temperature": 0.1, "max_length": config['response_max_length']}
        )

        return llm
    
    def create_embeddings(self, config):
        return HuggingFaceInferenceAPIEmbeddings(api_key=config['HUGGINGFACEHUB_API_TOKEN'], model_name=config['emb_model_name'])
    
    def create_custom_prompt(self):
        template = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know, don't try to make up an answer. \
        Use three sentences maximum and keep the answer as concise as possible. \
        Always say "thanks for asking!" at the end of the answer. \

        {context}

        Question: {question}

        Helpful Answer:"""

        custom_rag_prompt = PromptTemplate.from_template(template)

        self.prompt = custom_rag_prompt
    
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)