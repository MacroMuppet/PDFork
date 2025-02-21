from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM, OllamaEmbeddings
import json
import pickle
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import os
# Add new imports
import fitz  # PyMuPDF for PDF processing and image extraction DOES NOT USUALLY WORK FOR PDFs CREATED FROM PRINT>SAVE AS PDF
import cv2   # OpenCV for image processing and analysis
import numpy as np
from PIL import Image
import io
import sys
import platform

# Added functionality for semantic chunking and mapping of documents 2/20/2025

from semantic_chunker import SemanticChunker
from document_mapper import DocumentMapper

load_dotenv()

def setup_tesseract():
    """Setup Tesseract configuration for the current platform"""
    import pytesseract
    
    if platform.system() == "Windows":
        # Default Tesseract install location on Windows
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if not os.path.exists(tesseract_path):
            # Try conda environment path
            conda_path = os.path.dirname(sys.executable)
            tesseract_path = os.path.join(conda_path, "Library", "bin", "tesseract.exe")
        
        if not os.path.exists(tesseract_path):
            print("Warning: Tesseract not found in standard locations.")
            print("Please ensure Tesseract is installed and the path is correct.")
            print("You can download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
            return False
        
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    return True

def enhance_image_for_ocr(image):
    """Enhance image quality for better OCR results"""
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(binary)
    
    # Dilation to make text more prominent
    kernel = np.ones((1,1), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)
    
    return dilated

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using multiple methods and enhanced error handling"""
    text = ""
    errors = []
    
    # Setup Tesseract
    tesseract_available = setup_tesseract()
    
    try:
        print(f"Attempting to extract text from {pdf_path}")
        
        # Method 1: PyPDF2
        try:
            print("Trying PyPDF2 extraction...")
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if text.strip():
                print("Successfully extracted text using PyPDF2")
                return text
        except Exception as e:
            error_msg = f"PyPDF2 extraction failed: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
        
        # Method 2: PyMuPDF (more robust)
        try:
            print("Trying PyMuPDF extraction...")
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc, 1):
                print(f"Processing page {page_num}/{len(doc)}")
                
                # Try different text extraction methods
                page_text = ""
                
                # Method 2.1: Basic text extraction
                page_text = page.get_text()
                if not page_text.strip():
                    # Method 2.2: Try with different parameters
                    page_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
                
                if not page_text.strip():
                    # Method 2.3: Try extracting text blocks
                    blocks = page.get_text("blocks")
                    for block in blocks:
                        if block[4].strip():  # block[4] contains the text
                            page_text += block[4] + "\n"
                
                # Method 2.4: If still no text and OCR is available, try OCR
                if not page_text.strip() and tesseract_available:
                    print(f"Attempting OCR on page {page_num}")
                    try:
                        # Get page as image
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        img_array = np.array(img)
                        
                        # Enhance image for OCR
                        enhanced_img = enhance_image_for_ocr(img_array)
                        
                        # Save temporary image
                        temp_img_path = f"temp_page_{page_num}.png"
                        cv2.imwrite(temp_img_path, enhanced_img)
                        
                        # Perform OCR with custom configuration
                        import pytesseract
                        custom_config = r'--oem 3 --psm 6'  # Assume uniform text layout
                        page_text = pytesseract.image_to_string(
                            Image.open(temp_img_path),
                            config=custom_config
                        )
                        
                        if page_text.strip():
                            print(f"Successfully extracted text from page {page_num} using OCR")
                    except Exception as ocr_e:
                        print(f"OCR failed for page {page_num}: {str(ocr_e)}")
                    finally:
                        if os.path.exists(temp_img_path):
                            os.remove(temp_img_path)
                
                if page_text.strip():
                    text += page_text + "\n"
            
            doc.close()
            if text.strip():
                print("Successfully extracted text using PyMuPDF/OCR combination")
                return text
        except Exception as e:
            error_msg = f"PyMuPDF extraction failed: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
        
        # If we get here, no method succeeded
        error_msg = "\n".join(errors)
        print(f"All extraction methods failed:\n{error_msg}")
        raise ValueError(f"No text could be extracted from the PDF. Attempted methods failed:\n{error_msg}")
        
    except Exception as e:
        print(f"Fatal error during text extraction: {str(e)}")
        raise Exception(f"Error extracting text from PDF: {str(e)}")


# Load and split the text into chunks SEMANTIC CHUNKING with no character size guardrail
def process_text(text):
    semantic_chunker = SemanticChunker()
    # Let the semantic boundaries be the primary driver of chunking
    chunks = semantic_chunker.create_semantic_chunks(text)
    return chunks

def verify_gpu_availability():
    """Verify if GPU is available and properly configured for Ollama"""
    try:
        import subprocess
        
        # Check if nvidia-smi is available
        try:
            nvidia_smi = subprocess.check_output(['nvidia-smi'])
            print("NVIDIA GPU detected:")
            print(nvidia_smi.decode('utf-8').split('\n')[0])  # Print GPU info
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: nvidia-smi not found. GPU may not be available.")
            return False
        
        # Check Ollama config
        config_path = None
        if os.name == 'nt':  # Windows
            config_path = os.path.join(os.getenv('LOCALAPPDATA'), 'Ollama', 'config.json')
        else:  # Linux/macOS
            config_path = os.path.expanduser('~/.ollama/config.json')
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if not config.get('gpu') or not config.get('cuda'):
                        print(f"Warning: GPU not enabled in Ollama config at {config_path}")
                        print("Add the following to your config file:")
                        print('{\n  "gpu": true,\n  "cuda": true\n}')
                        return False
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in Ollama config at {config_path}")
                return False
        else:
            print(f"Warning: Ollama config not found at {config_path}")
            print("Create the config file with GPU settings enabled.")
            return False
        
        return True
    except Exception as e:
        print(f"Error checking GPU availability: {str(e)}")
        return False

def get_llm(provider="openai", model_name=None):
    """Get LLM based on provider choice"""
    if provider.lower() == "openai":
        return ChatOpenAI(
            temperature=0.7,
            request_timeout=60,
            max_retries=3,
            model_name=model_name or "gpt-3.5-turbo"
        )
    elif provider.lower() == "ollama":
        # Verify GPU availability for Ollama
        gpu_available = verify_gpu_availability()
        if not gpu_available:
            print("Warning: GPU may not be properly configured. Ollama will use CPU only.")
        
        return OllamaLLM(
            model=model_name or "mistral",
            temperature=0.7,
            num_gpu=1 if gpu_available else 0,  # Use GPU only if available
            num_thread=8
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def get_embeddings(provider="openai", model_name=None):
    """Get embeddings based on provider choice"""
    if provider.lower() == "openai":
        return OpenAIEmbeddings(model="text-embedding-ada-002")
    elif provider.lower() == "ollama":
        # Verify GPU availability for Ollama
        gpu_available = verify_gpu_availability()
        if not gpu_available:
            print("Warning: GPU may not be properly configured. Ollama will use CPU only for embeddings.")
        
        return OllamaEmbeddings(
            model=model_name or "nomic-embed-text",
            num_gpu=1 if gpu_available else 0  # Use GPU only if available
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def create_vector_store(chunks, doc_id, provider="openai", model_name=None):
    """Create vector embeddings and store them"""
    if not chunks:
        raise ValueError("No text chunks provided for embedding")
    try:
        # Add document ID to each chunk's metadata
        metadatas = [{"doc_id": doc_id} for _ in chunks]
        
        embeddings = get_embeddings(provider, model_name)
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory=f"./data/{doc_id}"
        )
        return vectorstore
    except Exception as e:
        raise Exception(f"Error creating vector store: {str(e)}")

def create_conversation_chain(vectorstore, provider="openai", model_name=None):
    llm = get_llm(provider, model_name)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return conversation_chain

# Usage example
def query_document(conversation_chain, query, chat_history=[]):
    result = conversation_chain({"question": query, "chat_history": chat_history})
    return result["answer"]

def save_vector_store(vectorstore, file_name):
    """Save vector store to disk"""
    save_path = Path("./vector_stores")
    save_path.mkdir(exist_ok=True)
    
    # Chroma automatically persists when initialized with persist_directory
    vectorstore.persist()
    
    # Save metadata about the vectorstore
    metadata = {
        "date_created": datetime.now().isoformat(),
        "store_type": "chroma",
        "location": str(save_path / file_name)
    }
    
    with open(save_path / f"{file_name}_metadata.json", "w") as f:
        json.dump(metadata, f)

def load_vector_store(file_name):
    """Load vector store from disk"""
    load_path = Path("./vector_stores") / file_name
    
    # Initialize Chroma with the same persist_directory
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=str(load_path),
        embedding_function=embeddings
    )
    
    return vectorstore

def save_chat_history(chat_history, file_name):
    """Save chat history to disk"""
    save_path = Path("./chat_histories")
    save_path.mkdir(exist_ok=True)
    
    with open(save_path / f"{file_name}.pkl", "wb") as f:
        pickle.dump(chat_history, f)

def load_chat_history(file_name):
    """Load chat history from disk"""
    load_path = Path("./chat_histories") / f"{file_name}.pkl"
    
    if load_path.exists():
        with open(load_path, "rb") as f:
            return pickle.load(f)
    return []

def extract_visual_elements(pdf_path):
    """Extract tables, charts, images and other visual elements from PDF"""
    try:
        doc = fitz.open(pdf_path)
        visual_elements = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract images
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to numpy array for analysis
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Basic image classification (you might want to enhance this)
                height, width = image.shape[:2]
                aspect_ratio = width / height
                
                # Attempt to classify the image type
                element_type = "unknown"
                if 0.8 <= aspect_ratio <= 1.2:  # Nearly square
                    if _has_chart_features(image):
                        element_type = "chart"
                    elif _has_table_features(image):
                        element_type = "table"
                    else:
                        element_type = "image"
                
                visual_elements.append({
                    "type": element_type,
                    "page": page_num + 1,
                    "size": (width, height),
                    "content": image_bytes
                })
        
        return visual_elements
    except Exception as e:
        raise Exception(f"Error extracting visual elements from PDF: {str(e)}")

def _has_chart_features(image):
    """Basic detection of chart-like features"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Look for lines (common in charts)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                           minLineLength=100, maxLineGap=10)
    
    return lines is not None and len(lines) > 5

def _has_table_features(image):
    """Basic detection of table-like features"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Look for grid-like patterns
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                           minLineLength=50, maxLineGap=5)
    
    if lines is not None:
        # Check for perpendicular lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180.0 / np.pi)
            angles.append(angle)
        
        # Count horizontal and vertical lines
        horizontal = sum(1 for angle in angles if angle < 10 or angle > 170)
        vertical = sum(1 for angle in angles if 80 < angle < 100)
        
        return horizontal > 2 and vertical > 2
    
    return False

def summarize_sections(vectorstore, pdf_name, provider="openai", model_name=None):
    """Summarize sections of the PDF using specified LLM provider"""
    # Create output directory for summaries
    summary_path = Path("./output/summaries")
    summary_path.mkdir(exist_ok=True)
    
    # Initialize LLM with specified provider
    llm = get_llm(provider, model_name)
    
    # Create a filtered retriever for this document only
    search_kwargs = {
        "k": 5,
        "filter": {"doc_id": pdf_name}
    }
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    try:
        # First, get a general summary of the entire document
        general_docs = retriever.get_relevant_documents(
            "What are the main goals and purpose of this document?"
        )
        
        if not general_docs:
            raise ValueError(f"No content found for document {pdf_name}")
            
        general_summary = llm.predict(
            "Based on the following content, provide a concise summary of the document's main goals and purpose (max 200 words): " + 
            "\n\n".join([doc.page_content for doc in general_docs])
        )

        # Get main conclusions and findings
        conclusions_docs = retriever.get_relevant_documents(
            "What are the main conclusions, findings, and key takeaways of this document?"
        )
        
        conclusions_summary = llm.predict(
            "Based on the following content, provide a clear and structured list of the main conclusions, findings, and key takeaways from this document. " +
            "Focus on concrete results, insights, and important discoveries (max 250 words): " + 
            "\n\n".join([doc.page_content for doc in conclusions_docs])
        )
        
        # Then, identify and summarize main sections
        section_docs = retriever.get_relevant_documents(
            "What are the main sections or topics discussed in this document? List them in order."
        )
        
        if not section_docs:
            raise ValueError(f"No sections found for document {pdf_name}")
            
        section_names_text = llm.predict(
            "Based on the following content, list up to 5 main sections or topics discussed in this document in order. " +
            "Format as a simple list with one section name per line, no numbers or bullets: " +
            "\n\n".join([doc.page_content for doc in section_docs])
        )
        
        # Process section names
        section_names = [name.strip() for name in section_names_text.split('\n') if name.strip()][:5]  # Limit to 5 sections
        
        # Get summaries for each identified section
        section_summaries = []
        for section_name in section_names:
            try:
                # Get content relevant to this section
                section_docs = retriever.get_relevant_documents(
                    f"What are the key points discussed in the section about {section_name}?"
                )
                
                if section_docs:
                    # Generate summary for this section
                    section_summary = llm.predict(
                        f"Summarize the key points and findings from the section about '{section_name}' based on the following content (max 150 words): " +
                        "\n\n".join([doc.page_content for doc in section_docs])
                    )
                    
                    section_summaries.append((section_name, section_summary))
            except Exception as e:
                print(f"Warning: Failed to summarize section '{section_name}': {str(e)}")
                continue
        
        # Save summaries to a text file
        summary_file = summary_path / f"{pdf_name}_Summary.txt"
        with open(summary_file, "w", encoding='utf-8') as f:
            # Write general summary first
            f.write("General Summary\n")
            f.write("=" * 50 + "\n")
            f.write(general_summary + "\n\n")
            
            # Write main conclusions and findings
            f.write("Main Conclusions and Findings\n")
            f.write("=" * 50 + "\n")
            f.write(conclusions_summary + "\n\n")
            
            # Write section summaries
            if section_summaries:
                f.write("Detailed Section Summaries\n")
                f.write("=" * 50 + "\n\n")
                for section_name, summary in section_summaries:
                    f.write(f"{section_name}\n")
                    f.write("-" * len(section_name) + "\n")
                    f.write(summary + "\n\n")
        
        print(f"Summary saved to: {summary_file}")
        return general_summary, conclusions_summary, section_summaries
    
    except Exception as e:
        print(f"Error generating summary for {pdf_name}: {str(e)}")
        raise

# Modify the process_and_save_pdf function to include summarization
def process_and_save_pdf(pdf_path, store_name, provider="openai", model_name=None):
    """Process PDF and save all necessary data using specified provider"""
    # Create base output directory
    base_dir = Path("./output")
    base_dir.mkdir(exist_ok=True)
    print(f"Output directory created at: {base_dir.absolute()}")
    print(f"Using {provider} provider with model: {model_name or 'default'}")
    
    # Process text
    text = extract_text_from_pdf(pdf_path)
    chunks = process_text(text)
    
    # Save raw text as JSON
    text_path = base_dir / "raw_text"
    text_path.mkdir(exist_ok=True)
    text_file = text_path / f"{store_name}_text.json"
    with open(text_file, "w", encoding='utf-8') as f:
        json.dump({
            "text": text,
            "date_processed": datetime.now().isoformat(),
            "source_file": pdf_path,
            "provider": provider,
            "model": model_name
        }, f, ensure_ascii=False, indent=2)
    print(f"Raw text saved to: {text_file}")
    
    # Create and save vector store
    vectorstore = create_vector_store(chunks, store_name, provider, model_name)
    save_vector_store(vectorstore, store_name)
    
    # Summarize sections (updated to handle new return value)
    general_summary, conclusions_summary, section_summaries = summarize_sections(
        vectorstore, 
        store_name, 
        provider, 
        model_name
    )
    
    # Save visual elements in document-specific folder
    visuals_base_path = base_dir / "visual_elements"
    visuals_base_path.mkdir(exist_ok=True)
    
    # Create document-specific folder
    doc_visuals_path = visuals_base_path / store_name
    doc_visuals_path.mkdir(exist_ok=True)
    print(f"Visual elements will be saved to: {doc_visuals_path}")
    
    visual_elements = extract_visual_elements(pdf_path)
    
    # Save metadata
    metadata = {
        "date_created": datetime.now().isoformat(),
        "num_elements": len(visual_elements),
        "elements": [{
            "type": elem["type"],
            "page": elem["page"],
            "size": elem["size"]
        } for elem in visual_elements]
    }
    
    metadata_file = doc_visuals_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)
    print(f"Metadata saved to: {metadata_file}")
    
    # Save images in document folder
    for idx, elem in enumerate(visual_elements):
        image_file = doc_visuals_path / f"visual_{idx}.png"
        with open(image_file, "wb") as f:
            f.write(elem["content"])
        print(f"Saved visual element {idx} to: {image_file}")
    
    return vectorstore, visual_elements

def load_and_query(store_name, query, chat_history_name=None):
    """Load saved data and query the document"""
    # Load vector store
    vectorstore = load_vector_store(store_name)
    
    # Load or initialize chat history
    chat_history = load_chat_history(chat_history_name) if chat_history_name else []
    
    # Create conversation chain and query
    conversation_chain = create_conversation_chain(vectorstore)
    response = query_document(conversation_chain, query, chat_history)
    
    # Save updated chat history
    if chat_history_name:
        chat_history.append((query, response))
        save_chat_history(chat_history, chat_history_name)
    
    return response

def process_pdf_and_move(pdf_path):
    """Process a single PDF and move it to finished folder after processing"""
    try:
        # Get the PDF filename
        pdf_name = Path(pdf_path).stem
        print(f"\n{'='*80}\nProcessing: {pdf_name}\n{'='*80}")
        
        # Process the PDF
        vectorstore, visual_elements = process_and_save_pdf(pdf_path, pdf_name)
        
        # Generate summary
        try:
            summarize_sections(vectorstore, pdf_name)
        except Exception as e:
            print(f"Warning: Could not generate summary: {str(e)}")
        
        # Move the PDF to finished folder
        dest_path = Path("finished") / Path(pdf_path).name
        Path(pdf_path).rename(dest_path)
        
        print(f"Successfully processed and moved {pdf_name} to finished folder")
        return True
    except Exception as e:
        print(f"Error processing {pdf_name}: {str(e)}")
        return False

def batch_process_pdfs(provider="openai", model_name=None):
    """Process all PDFs in the preprocess folder using specified provider"""
    preprocess_dir = Path("preprocess")
    if not preprocess_dir.exists():
        print("Error: preprocess directory not found")
        return
    
    # Create finished directory if it doesn't exist
    finished_dir = Path("finished")
    finished_dir.mkdir(exist_ok=True)
    
    # Get all PDF files
    pdf_files = list(preprocess_dir.glob("*.pdf"))
    total_pdfs = len(pdf_files)
    
    if total_pdfs == 0:
        print("No PDF files found in preprocess directory")
        return
    
    print(f"Found {total_pdfs} PDF files to process")
    print(f"Using {provider} provider with model: {model_name or 'default'}")
    
    # Initialize document mapper with embeddings model
    embeddings = get_embeddings(provider, model_name)
    document_mapper = DocumentMapper(embeddings)
    
    # Process each PDF
    successful = 0
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\nProcessing file {i} of {total_pdfs}")
        try:
            # Get the PDF filename as doc_id
            doc_id = pdf_path.stem
            
            # Process the PDF and get vectorstore
            vectorstore, visual_elements = process_and_save_pdf(str(pdf_path), doc_id, provider, model_name)
            
            # Extract text for document mapping
            text = extract_text_from_pdf(str(pdf_path))
            semantic_chunker = SemanticChunker()
            chunks = semantic_chunker.create_semantic_chunks(text)
            
            # Add to document mapper
            metadata = {
                "path": str(pdf_path),
                "processed_date": datetime.now().isoformat(),
                "num_chunks": len(chunks),
                "num_visuals": len(visual_elements) if visual_elements else 0
            }
            document_mapper.add_document(doc_id, doc_id, chunks, metadata)
            
            # Move the PDF to finished folder
            dest_path = finished_dir / pdf_path.name
            pdf_path.rename(dest_path)
            
            successful += 1
            print(f"Successfully processed {doc_id}")
            
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {str(e)}")
            continue
    
    # Save the document relationship visualization
    try:
        document_mapper.visualize_relationships('document_relationships.html')
        print("\nDocument relationship visualization saved to document_relationships.html")
    except Exception as e:
        print(f"Error generating relationship visualization: {str(e)}")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful}/{total_pdfs} files")
    print(f"Failed: {total_pdfs - successful} files")


# OLD CODE FOR BATCH PROCESSING NO MAPPING TOPIC RELATIONSHIPS
# def batch_process_pdfs(provider="openai", model_name=None):
#     """Process all PDFs in the preprocess folder using specified provider"""
#     preprocess_dir = Path("preprocess")
#     if not preprocess_dir.exists():
#         print("Error: preprocess directory not found")
#         return
    
#     # Create finished directory if it doesn't exist
#     finished_dir = Path("finished")
#     finished_dir.mkdir(exist_ok=True)
    
#     # Get all PDF files
#     pdf_files = list(preprocess_dir.glob("*.pdf"))
#     total_pdfs = len(pdf_files)
    
#     if total_pdfs == 0:
#         print("No PDF files found in preprocess directory")
#         return
    
#     print(f"Found {total_pdfs} PDF files to process")
#     print(f"Using {provider} provider with model: {model_name or 'default'}")
    
#     # Process each PDF
#     successful = 0
#     for i, pdf_path in enumerate(pdf_files, 1):
#         print(f"\nProcessing file {i} of {total_pdfs}")
#         if process_pdf_and_move(str(pdf_path)):
#             successful += 1
    
#     print(f"\nProcessing complete!")
#     print(f"Successfully processed: {successful}/{total_pdfs} files")
#     print(f"Failed: {total_pdfs - successful} files")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDFs with AI-powered analysis")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai",
                      help="Choose the AI provider (default: openai)")
    parser.add_argument("--model", type=str, default=None,
                      help="Specify the model name (default: provider's default model)")
    parser.add_argument("--force-cpu", action="store_true",
                      help="Force CPU usage even if GPU is available")
    
    args = parser.parse_args()
    
    if args.provider.lower() == "ollama" and not args.force_cpu:
        # Verify GPU at startup
        if verify_gpu_availability():
            print("GPU support verified for Ollama")
        else:
            print("Warning: GPU support not verified. Use --force-cpu to suppress this warning.")
            if input("Continue anyway? (y/n): ").lower() != 'y':
                sys.exit(1)
    
    batch_process_pdfs(provider=args.provider, model_name=args.model)
