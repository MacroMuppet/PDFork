from PDFLCEmbed import create_vector_store, get_embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Test data
test_text = """
This is a test document to verify our Chroma DB setup with Ollama nomic embeddings.
We want to make sure that the embeddings are working correctly and that we can retrieve similar content.
This test will help us verify that everything is configured properly.
"""

def main():
    try:
        # Clean up any existing test store
        test_store_path = Path("./data/test_store")
        if test_store_path.exists():
            logging.info(f"Removing existing test store at {test_store_path}")
            shutil.rmtree(test_store_path)

        # Split text into chunks
        logging.info("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
        )
        chunks = text_splitter.split_text(test_text)
        logging.info(f"Created {len(chunks)} chunks")

        logging.info("Creating vector store with Ollama nomic embeddings...")
        vectorstore = create_vector_store(
            chunks=chunks,
            doc_id="test_store",
            provider="ollama",
            model_name="nomic-embed-text"
        )
        logging.info("Vector store created successfully")

        # Test similarity search
        logging.info("Testing similarity search...")
        query = "How is the embeddings setup configured?"
        results = vectorstore.similarity_search(query)
        
        print(f"\nQuery: {query}")
        print("\nResults:")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.page_content.strip()}")
            print(f"   Metadata: {doc.metadata}\n")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 