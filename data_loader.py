import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_txt_data(data_directory_path):
    # 1. Load all .txt files from all subfolders automatically
    loader = DirectoryLoader(
        data_directory_path, 
        glob="**/*.txt", 
        loader_cls=TextLoader, 
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()
    
    # 2. Extract metadata from the file paths
    for doc in documents:
        # The source path looks like: "data\Nvidia\2023_Q4.txt"
        file_path = doc.metadata.get("source", "")
        
        # Normalize the slashes for Windows/Mac compatibility
        normalized_path = file_path.replace("\\", "/")
        path_parts = normalized_path.split("/")
        
        if len(path_parts) >= 3:
            # Extract the folder name as the Company
            doc.metadata["company"] = path_parts[-2] 
            
            # Extract the Year and Quarter from the filename safely
            filename = path_parts[-1].replace(".txt", "")
            file_parts = filename.split("_")
            
            # Ensures index out of bounds does not occur on irregular files
            if len(file_parts) >= 2:
                doc.metadata["year"] = file_parts[0]      # Extracts "2023"
                doc.metadata["quarter"] = file_parts[1]   # Extracts "Q4"
                
    # 3. Split the tagged documents into vectors
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    
    return chunks