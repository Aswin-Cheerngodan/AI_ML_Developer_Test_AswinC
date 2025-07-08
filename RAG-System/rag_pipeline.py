import os
import uuid
import shutil
from pathlib import Path
from typing import Optional
from fastapi import UploadFile
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter  
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

class RagPipeline:
    """Pipeline for Retrieval Augmented Generation."""
    def __init__(self, upload_dir: Path = Path(r"uploads/")):
        """Intialize with upload directory.
        
        Args:
            upload_dir (Path): Directory to save uploaded files. Default "uploads/"
        """
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def upload_file(self, file: UploadFile):
        """Accept and save the file, Updates the file path.
        
        Args:
            file (UploadFile): Uploaded file from FastAPI.

        """
        try:
            # Save the file
            unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = self.upload_dir / unique_filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            # file saved
            self.file_path = file_path
        except Exception as e:
            # Code to run if exception occurs
            return 
        
    def create_vectordb(self):
        """Checks and Loads the document from the file path and creates the vectordb.

        Returns:
            Optional[file]: Returns the vectordb for the RAG. None if fails
        """
        try:
            # Checking the document
            str_file_path = str(self.file_path)
            if str_file_path.endswith(".pdf"):
                loader = PyPDFLoader(self.file_path)
                
            elif str_file_path.endswith(".txt"):
                loader = TextLoader(self.file_path)
                
            # Loading the document
            documents = loader.load()
            # Flushing the saved document
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
                
            # Split document for storing in vector database
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = text_splitter.split_documents(documents)

            # Path to the pre-trained model 
            modelPath = "sentence-transformers/all-MiniLM-l6-v2"

            # Model configuration options, specifying to use the CPU for computations
            model_kwargs = {'device':'cpu'}

            # Encoding options, specifically setting 'normalize_embeddings' to False
            encode_kwargs = {'normalize_embeddings': False}

            # HuggingFaceEmbeddings with the specified parameters
            embeddings = HuggingFaceEmbeddings(
                model_name=modelPath,     # pre-trained model's path
                model_kwargs=model_kwargs, # Model configuration options
                encode_kwargs=encode_kwargs # Encoding options
            )
            
            # Create and store documents in vector database
            vector_db=FAISS.from_documents(docs, embeddings)
            
            return vector_db
        except Exception as e:
            # Code to run if exception occurs
            return None 
        
    def model_pipeline(self):
        """Create model pipeline for generation
        
        Returns:
            pipeline: question answer huggingface pipeline.
        """
        try:
            # Model name 
            model_name = "Intel/dynamic_tinybert"

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)

            # Question-answering pipeline using the model and tokenizer
            question_answerer = pipeline(
                "question-answering", 
                model=model_name, 
                tokenizer=tokenizer,
                return_tensors='pt'
            )

            # Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
            # with additional model-specific arguments (temperature and max_length)
            llm = HuggingFacePipeline(
                pipeline=question_answerer,
                model_kwargs={"temperature": 0.7, "max_length": 512},
            )

            return llm
        except Exception as e:
            # Code to run if Exception occurs
            return 

    def rag_generator(self,query, vector_db, llm):
        """Generates responses based on retrieved information from vector database and query.
        Args:
            query: User's query to the system.
            vector_db: vector database with loaded documents.
        Returns:
            response: The response generated by the rag pipeline.
        """
        try:
            # Retrieving relevant documents from vector database upto 4 documents
            retriever = vector_db.as_retriever(search_kwargs={"k": 4})
            # creating retrieval qa chain with llm, retriever
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False)
            # Run the chain with query
            result = qa.run({"query": query})
            
            return result["result"]
        except Exception as e:
            # Code to run if Exception occurs
            return 



                    
        
    