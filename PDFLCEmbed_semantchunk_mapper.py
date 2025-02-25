from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
import json
import pickle
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import os
# Add new imports
import fitz  # PyMuPDF for PDF processing and image extraction
import cv2   # OpenCV for image processing and analysis
import numpy as np
from PIL import Image
import io
import sys
import platform
import math  # For mathematical operations in image analysis
import statistics  # For statistical analysis in chart detection

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
        
        # Method 1: PyMuPDF (primary method)
        try:
            print("Using PyMuPDF extraction...")
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc, 1):
                print(f"Processing page {page_num}/{len(doc)}")
                
                # Try different text extraction methods
                page_text = ""
                
                # Method 1.1: Basic text extraction
                page_text = page.get_text()
                if not page_text.strip():
                    # Method 1.2: Try with different parameters
                    page_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
                
                if not page_text.strip():
                    # Method 1.3: Try extracting text blocks
                    blocks = page.get_text("blocks")
                    for block in blocks:
                        if block[4].strip():  # block[4] contains the text
                            page_text += block[4] + "\n"
                
                # Method 1.4: If still no text and OCR is available, try OCR
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
        
        # Method 2: PyPDF2 (fallback method)
        if not text.strip():
            try:
                print("Falling back to PyPDF2 extraction...")
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
        
        # Using mistral as default LLM because it's optimized for text generation and chat
        return ChatOllama(
            model=model_name or "mistral",
            temperature=0.7,
            num_ctx=2048,
            num_gpu=1 if gpu_available else 0,
            num_thread=8,
            stop=["</s>"],
            repeat_penalty=1.1,
            top_k=10,
            top_p=0.8,
            mirostat_mode=0,
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
            print("Warning: GPU may not be properly configured. Ollama will use CPU only.")
        
        try:
            # Using nomic-embed-text as primary choice because it's specifically optimized for creating text embeddings
            print("Initializing Ollama embeddings with nomic-embed-text model...")
            embeddings = OllamaEmbeddings(
                model="nomic-embed-text",  # Specialized embedding model
                temperature=0.0,  # Keep it deterministic for embeddings
                num_ctx=2048,  # Context window size
                num_gpu=1 if gpu_available else 0,  # Use GPU if available
                num_thread=8  # Number of CPU threads
            )
            # Test the embeddings with both single and batch methods
            try:
                print("Testing embeddings with a sample text...")
                # Test single text embedding
                test_result = embeddings.embed_documents(["test"])
                print("Embeddings test successful!")
                return embeddings
            except Exception as test_e:
                print(f"Embeddings test failed: {str(test_e)}")
                raise
        except Exception as e:
            print(f"\nWarning: Failed to initialize nomic-embed-text: {str(e)}")
            print("Falling back to all-minilm model...")
            try:
                # all-minilm is another embedding-specific model, used as fallback
                print("Initializing Ollama embeddings with all-minilm model...")
                embeddings = OllamaEmbeddings(
                    model="all-minilm",  # Alternative embedding model
                    temperature=0.0,  # Keep it deterministic for embeddings
                    num_ctx=2048,  # Context window size
                    num_gpu=1 if gpu_available else 0,  # Use GPU if available
                    num_thread=8  # Number of CPU threads
                )
                # Test the embeddings
                try:
                    print("Testing embeddings with a sample text...")
                    test_result = embeddings.embed_documents(["test"])
                    print("Embeddings test successful!")
                    return embeddings
                except Exception as test_e:
                    print(f"Embeddings test failed: {str(test_e)}")
                    raise
            except Exception as e2:
                print(f"\nError initializing embeddings: {str(e2)}")
                print("\nPlease ensure you have downloaded the required Ollama models:")
                print("Run: ollama pull nomic-embed-text")
                print("  or: ollama pull all-minilm")
                print("\nIf the error persists, try running:")
                print("ollama rm nomic-embed-text")
                print("ollama pull nomic-embed-text")
                print("  or")
                print("ollama rm all-minilm")
                print("ollama pull all-minilm")
                print("\nAlso try upgrading langchain-ollama:")
                print("pip install --upgrade langchain-ollama")
                raise
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def sanitize_collection_name(name):
    """Sanitize the collection name to meet Chroma's requirements"""
    import re
    # Replace spaces and invalid characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9-]', '_', name)
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Ensure it starts and ends with alphanumeric character
    sanitized = re.sub(r'^[^a-zA-Z0-9]+', '', sanitized)
    sanitized = re.sub(r'[^a-zA-Z0-9]+$', '', sanitized)
    # If name is too short, pad it
    if len(sanitized) < 3:
        sanitized = sanitized + "_doc"
    # If name is too long, truncate it
    if len(sanitized) > 63:
        sanitized = sanitized[:63]
        # Ensure it ends with alphanumeric
        sanitized = re.sub(r'[^a-zA-Z0-9]+$', '', sanitized)
    return sanitized

def create_vector_store(chunks, doc_id, provider="openai", model_name=None):
    """Create vector embeddings and store them"""
    if not chunks:
        raise ValueError("No text chunks provided for embedding")
    try:
        print(f"\nCreating vector store for {doc_id}")
        print(f"Number of chunks to process: {len(chunks)}")
        print(f"Using provider: {provider} with model: {model_name or 'default'}")
        
        # Add document ID to each chunk's metadata
        metadatas = [{"doc_id": doc_id} for _ in chunks]
        
        print("Initializing embedding model...")
        try:
            embeddings = get_embeddings(provider, model_name)
            
            print("Creating and storing embeddings in Chroma DB...")
            print(f"This may take several minutes for large documents...")
            
            # Create Chroma client with persist directory
            import chromadb
            chroma_client = chromadb.PersistentClient(path=f"./data/{doc_id}")
            
            # Sanitize collection name
            collection_name = sanitize_collection_name(doc_id)
            print(f"Using collection name: {collection_name}")
            
            # Create collection
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            # Get embeddings for chunks
            print("Computing embeddings for chunks...")
            embedded_vectors = embeddings.embed_documents(chunks)
            
            # Add documents to collection
            print("Adding documents to Chroma collection...")
            collection.add(
                embeddings=embedded_vectors,
                documents=chunks,
                metadatas=metadatas,
                ids=[f"{collection_name}_{i}" for i in range(len(chunks))]
            )
            
            # Create LangChain Chroma wrapper
            vectorstore = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=f"./data/{doc_id}"
            )
            
            print(f"Vector store successfully created at: ./data/{doc_id}")
            return vectorstore
        except Exception as e:
            print(f"Error during embedding creation: {str(e)}")
            if "encode" in str(e):
                print("\nThis error often means the embedding model is not properly installed.")
                print("Please try running:")
                print("  ollama pull nomic-embed-text")
                print("  or")
                print("  ollama pull all-minilm")
            raise
    except Exception as e:
        raise Exception(f"Error creating vector store: {str(e)}")

def create_conversation_chain(vectorstore, provider="openai", model_name=None):
    """Create a conversation chain with the specified LLM and vector store"""
    llm = get_llm(provider, model_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Create the conversation template
    template = """Answer the following question based on the provided context:
    
    Context: {context}
    
    Question: {question}
    
    Answer: """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the conversation chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def query_document(conversation_chain, query, chat_history=[]):
    """Query the document using the conversation chain"""
    try:
        response = conversation_chain.invoke(query)
        return response
    except Exception as e:
        print(f"Error querying document: {e}")
        return None

def save_vector_store(vectorstore, file_name):
    """Save vector store to disk"""
    try:
        print(f"\nSaving vector store for {file_name}...")
        save_path = Path("./vector_stores")
        save_path.mkdir(exist_ok=True)
        
        # Chroma is already persistent when initialized with persist_directory
        # Just save metadata about the vectorstore
        metadata = {
            "date_created": datetime.now().isoformat(),
            "store_type": "chroma",
            "location": str(save_path / file_name)
        }
        
        metadata_file = save_path / f"{file_name}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)
        
        print(f"Vector store metadata saved to: {metadata_file}")
        
    except Exception as e:
        print(f"Warning: Error saving vector store metadata: {str(e)}")
        # Don't raise the exception since the vector store is already persistent

def load_vector_store(file_name, provider="ollama", model_name="nomic-embed-text"):
    """Load vector store from disk"""
    try:
        # Get the correct embeddings based on provider
        print(f"Initializing embeddings with {provider} ({model_name})")
        embeddings = get_embeddings(provider, model_name)
        
        # Load the vector store with the correct embeddings
        load_path = Path("./data") / file_name
        if not load_path.exists():
            raise ValueError(f"No vector store found at {load_path}")
            
        # Use sanitized collection name
        collection_name = sanitize_collection_name(file_name)
        print(f"Loading collection: {collection_name}")
        
        # Initialize Chroma with the same persist_directory and embeddings
        vectorstore = Chroma(
            persist_directory=str(load_path),
            collection_name=collection_name,
            embedding_function=embeddings
        )
        
        return vectorstore
    except Exception as e:
        raise Exception(f"Error loading vector store: {str(e)}")

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
                
                if image is None or image.size == 0:
                    print(f"Warning: Could not decode image on page {page_num+1}")
                    continue
                
                # Get image dimensions
                height, width = image.shape[:2]
                
                # Skip very small images (likely icons or decorations)
                if width < 100 or height < 100:
                    continue
                
                # Attempt to classify the image type with enhanced detection
                element_type = classify_visual_element(image)
                
                visual_elements.append({
                    "type": element_type,
                    "page": page_num + 1,
                    "size": (width, height),
                    "content": image_bytes
                })
        
        return visual_elements
    except Exception as e:
        raise Exception(f"Error extracting visual elements from PDF: {str(e)}")

def classify_visual_element(image):
    """Classify an image as chart, table, or regular image using multiple detection methods"""
    # Calculate confidence scores for each type
    chart_score = calculate_chart_confidence(image)
    table_score = calculate_table_confidence(image)
    
    # Debug information
    print(f"Classification scores - Chart: {chart_score:.2f}, Table: {table_score:.2f}")
    
    # Determine the type based on confidence scores
    if chart_score > 0.6 and chart_score > table_score:
        return "chart"
    elif table_score > 0.6 and table_score > chart_score:
        return "table"
    else:
        return "image"

def calculate_chart_confidence(image):
    """Calculate confidence score that an image is a chart using multiple features"""
    # Initialize confidence score
    confidence = 0.0
    
    # 1. Check for lines (common in charts)
    line_score = _detect_chart_lines(image)
    confidence += line_score * 0.3  # 30% weight
    
    # 2. Check for color distribution (charts often have distinct color groups)
    color_score = _analyze_color_distribution(image)
    confidence += color_score * 0.2  # 20% weight
    
    # 3. Check for text density and distribution (charts often have axis labels, legends)
    text_score = _analyze_text_regions(image)
    confidence += text_score * 0.2  # 20% weight
    
    # 4. Check for histogram-like features (bar/column charts)
    histogram_score = _detect_histogram_features(image)
    confidence += histogram_score * 0.3  # 30% weight
    
    return confidence

def calculate_table_confidence(image):
    """Calculate confidence score that an image is a table using multiple features"""
    # Initialize confidence score
    confidence = 0.0
    
    # 1. Check for grid structure (primary table feature)
    grid_score = _detect_table_grid(image)
    confidence += grid_score * 0.4  # 40% weight
    
    # 2. Check for text alignment (tables have aligned text)
    alignment_score = _detect_text_alignment(image)
    confidence += alignment_score * 0.3  # 30% weight
    
    # 3. Check for consistent row heights (tables often have uniform rows)
    row_score = _detect_uniform_rows(image)
    confidence += row_score * 0.3  # 30% weight
    
    return confidence

def _detect_chart_lines(image):
    """Detect lines that are characteristic of charts (axes, grid lines)"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                           minLineLength=50, maxLineGap=10)
    
    if lines is None:
        return 0.0
    
    # Analyze line properties
    total_lines = len(lines)
    if total_lines < 3:
        return 0.0
    
    # Calculate line angles
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180.0 / np.pi)
        angles.append(angle)
    
    # Count horizontal and vertical lines (common in charts for axes)
    horizontal = sum(1 for angle in angles if angle < 10 or angle > 170)
    vertical = sum(1 for angle in angles if 80 < angle < 100)
    
    # Charts typically have at least one horizontal and one vertical axis
    if horizontal > 0 and vertical > 0:
        # Calculate score based on proportion of horizontal and vertical lines
        axis_score = min(1.0, (horizontal + vertical) / max(10, total_lines))
        return min(1.0, 0.3 + axis_score * 0.7)  # Base 0.3 + up to 0.7 for axes
    
    return 0.2  # Some lines but not clearly axes

def _analyze_color_distribution(image):
    """Analyze color distribution to identify chart-like patterns"""
    # Resize image for faster processing
    resized = cv2.resize(image, (200, 200))
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    
    # Calculate color histogram
    hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [30], [0, 256])
    
    # Normalize histograms
    cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
    
    # Calculate peaks in hue histogram (distinct colors often indicate chart elements)
    peaks = 0
    for i in range(1, len(hist_h) - 1):
        if hist_h[i] > hist_h[i-1] and hist_h[i] > hist_h[i+1] and hist_h[i] > 0.1:
            peaks += 1
    
    # Charts often have distinct color groups (3-7 is common)
    if 2 <= peaks <= 8:
        return 0.6 + min(0.4, peaks * 0.05)  # Higher score for optimal number of color groups
    elif peaks > 8:
        return 0.4  # Too many colors might be a photo
    else:
        return 0.2  # Too few distinct colors
        
def _analyze_text_regions(image):
    """Analyze text regions to identify chart-like text patterns"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find potential text regions
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours of potential text regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 3:
        return 0.1  # Too few text-like regions
    
    # Filter contours by size (text is usually small)
    height, width = image.shape[:2]
    min_area = (width * height) * 0.0001  # Minimum area for text
    max_area = (width * height) * 0.05    # Maximum area for text
    
    text_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    
    if len(text_contours) < 3:
        return 0.2  # Few potential text regions
    
    # Check for alignment patterns (charts often have aligned text on axes)
    x_coords = [cv2.boundingRect(c)[0] for c in text_contours]
    y_coords = [cv2.boundingRect(c)[1] for c in text_contours]
    
    # Count elements with similar x or y coordinates (aligned text)
    x_aligned = 0
    y_aligned = 0
    
    for i in range(len(x_coords)):
        for j in range(i+1, len(x_coords)):
            if abs(x_coords[i] - x_coords[j]) < width * 0.03:
                x_aligned += 1
            if abs(y_coords[i] - y_coords[j]) < height * 0.03:
                y_aligned += 1
    
    # Calculate alignment score
    total_possible = (len(text_contours) * (len(text_contours) - 1)) / 2
    if total_possible > 0:
        alignment_ratio = (x_aligned + y_aligned) / total_possible
        return min(1.0, 0.3 + alignment_ratio * 0.7)
    
    return 0.3  # Some text-like regions but not clearly aligned

def _detect_histogram_features(image):
    """Detect features characteristic of bar/column charts"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to separate foreground from background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 3:
        return 0.0  # Too few contours for a bar chart
    
    # Filter contours by size and shape
    height, width = image.shape[:2]
    min_area = (width * height) * 0.005  # Minimum area for a bar
    
    # Look for rectangle-like contours (bars/columns)
    rectangularity_scores = []
    aspect_ratios = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # Check how rectangular the contour is
        x, y, w, h = cv2.boundingRect(contour)
        bounding_rect_area = w * h
        rectangularity = area / bounding_rect_area if bounding_rect_area > 0 else 0
        
        # Calculate aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        
        if rectangularity > 0.7:  # Fairly rectangular
            rectangularity_scores.append(rectangularity)
            aspect_ratios.append(aspect_ratio)
    
    # Check if we found enough rectangular contours
    if len(rectangularity_scores) < 2:
        return 0.1  # Not enough rectangular shapes
    
    # Check for similar aspect ratios (bars in charts often have similar shapes)
    if len(aspect_ratios) >= 2:
        # Calculate standard deviation of aspect ratios
        try:
            std_dev = statistics.stdev(aspect_ratios)
            mean_ratio = statistics.mean(aspect_ratios)
            
            # Low standard deviation relative to mean indicates uniform bars
            if mean_ratio > 0 and std_dev / mean_ratio < 0.5:
                return min(1.0, 0.5 + len(rectangularity_scores) * 0.1)
        except statistics.StatisticsError:
            pass
    
    # Some rectangular shapes but not clearly a histogram
    return 0.3 + min(0.3, len(rectangularity_scores) * 0.05)

def _detect_table_grid(image):
    """Detect grid-like structures characteristic of tables"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive threshold to handle different lighting conditions
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Dilate to connect nearby lines
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # Find lines
    lines = cv2.HoughLinesP(dilated, 1, np.pi/180, threshold=50, 
                           minLineLength=30, maxLineGap=20)
    
    if lines is None or len(lines) < 4:  # Need minimum lines for a table
        return 0.0
    
    # Separate horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > abs(y2 - y1):  # Horizontal line
            horizontal_lines.append((x1, y1, x2, y2))
        else:  # Vertical line
            vertical_lines.append((x1, y1, x2, y2))
    
    # Check if we have both horizontal and vertical lines
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return 0.1  # Not enough lines in both directions
    
    # Check for intersections (grid cells)
    intersections = 0
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            # Check if lines intersect
            h_x1, h_y1, h_x2, h_y2 = h_line
            v_x1, v_y1, v_x2, v_y2 = v_line
            
            # Simple line intersection check
            if (min(h_x1, h_x2) <= max(v_x1, v_x2) and max(h_x1, h_x2) >= min(v_x1, v_x2) and
                min(h_y1, h_y2) <= max(v_y1, v_y2) and max(h_y1, h_y2) >= min(v_y1, v_y2)):
                intersections += 1
    
    # Tables typically have many intersections forming a grid
    if intersections > 4:
        # Calculate score based on number of intersections
        return min(1.0, 0.4 + (intersections / (len(horizontal_lines) * len(vertical_lines))) * 0.6)
    
    return 0.2  # Some lines but not clearly a grid

def _detect_text_alignment(image):
    """Detect aligned text characteristic of tables"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find text
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours of potential text regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 5:  # Need minimum text regions for a table
        return 0.0
    
    # Filter contours by size (text is usually small)
    height, width = image.shape[:2]
    min_area = (width * height) * 0.0001  # Minimum area for text
    max_area = (width * height) * 0.05    # Maximum area for text
    
    text_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    
    if len(text_contours) < 5:
        return 0.1  # Not enough potential text regions
    
    # Get bounding rectangles for text regions
    bounding_rects = [cv2.boundingRect(c) for c in text_contours]
    
    # Extract x and y coordinates
    x_coords = [rect[0] for rect in bounding_rects]
    y_coords = [rect[1] for rect in bounding_rects]
    
    # Group similar y-coordinates (rows in table)
    y_groups = []
    for y in y_coords:
        added = False
        for group in y_groups:
            if any(abs(y - existing_y) < height * 0.03 for existing_y in group):
                group.append(y)
                added = True
                break
        if not added:
            y_groups.append([y])
    
    # Group similar x-coordinates (columns in table)
    x_groups = []
    for x in x_coords:
        added = False
        for group in x_groups:
            if any(abs(x - existing_x) < width * 0.03 for existing_x in group):
                group.append(x)
                added = True
                break
        if not added:
            x_groups.append([x])
    
    # Tables typically have multiple rows and columns
    if len(y_groups) >= 3 and len(x_groups) >= 2:
        # Calculate score based on number of rows and columns
        row_col_score = min(1.0, (len(y_groups) + len(x_groups)) / 15)
        
        # Check if text is well-aligned in rows and columns
        alignment_score = 0.0
        if len(y_groups) > 0:
            # Calculate average number of elements per row
            avg_row_elements = sum(len(group) for group in y_groups) / len(y_groups)
            if avg_row_elements >= 2:
                alignment_score = min(1.0, avg_row_elements / 5)
        
        return min(1.0, 0.3 + (row_col_score * 0.4) + (alignment_score * 0.3))
    
    return 0.1  # Some text but not clearly aligned in table format

def _detect_uniform_rows(image):
    """Detect uniform row heights characteristic of tables"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply horizontal Sobel filter to detect horizontal edges
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    
    # Threshold to get horizontal lines
    _, binary = cv2.threshold(sobel_y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Dilate horizontally to connect line segments
    kernel = np.ones((1, 15), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # Find horizontal lines
    lines = cv2.HoughLinesP(dilated, 1, np.pi/180, threshold=50, 
                           minLineLength=image.shape[1] * 0.3, maxLineGap=20)
    
    if lines is None or len(lines) < 3:  # Need minimum horizontal lines for table rows
        return 0.0
    
    # Extract y-coordinates of horizontal lines
    y_coords = [line[0][1] for line in lines]
    y_coords.sort()
    
    # Calculate distances between consecutive lines
    distances = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
    
    if len(distances) < 2:
        return 0.1  # Not enough rows
    
    # Check for uniform row heights
    try:
        mean_distance = statistics.mean(distances)
        std_dev = statistics.stdev(distances)
        
        # Low standard deviation relative to mean indicates uniform rows
        if mean_distance > 0 and std_dev / mean_distance < 0.3:
            return min(1.0, 0.5 + len(distances) * 0.1)
    except statistics.StatisticsError:
        pass
    
    return 0.2  # Some horizontal lines but not clearly uniform

def _has_chart_features(image):
    """Legacy function for backward compatibility"""
    return calculate_chart_confidence(image) > 0.6

def _has_table_features(image):
    """Legacy function for backward compatibility"""
    return calculate_table_confidence(image) > 0.6

def summarize_sections(vectorstore, pdf_name, provider="openai", model_name=None):
    """Summarize sections of the PDF using specified LLM provider"""
    from time import time
    start_time = time()
    
    print(f"\nStarting document summarization for: {pdf_name}")
    print(f"Using {provider} provider with model: {model_name or 'default'}")
    
    # Create output directory for summaries
    summary_path = Path("./output/summaries")
    summary_path.mkdir(exist_ok=True)
    
    # Initialize LLM with specified provider
    print("Initializing LLM for summarization...")
    llm = get_llm(provider, model_name)
    
    # Create a filtered retriever for this document only
    search_kwargs = {
        "k": 3,  # Reduced from 5 to 3 for faster retrieval
        "filter": {"doc_id": pdf_name}
    }
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    try:
        # Batch retrieve documents for all summaries at once using invoke
        print("\nRetrieving document content for summaries...")
        general_docs = retriever.invoke(
            "What are the main goals, conclusions, and key sections of this document?"
        )
        
        if not general_docs:
            raise ValueError(f"No content found for document {pdf_name}")
        
        # For Ollama, process in smaller chunks
        if provider.lower() == "ollama":
            print("Using chunked processing for Ollama...")
            
            # Process general summary
            print("\nGenerating general summary...")
            summary_start = time()
            general_content = "\n\n".join([doc.page_content for doc in general_docs[:2]])
            general_summary = llm.invoke(
                "Provide a concise summary of the document's main goals and purpose (max 200 words): " + general_content
            ).content.strip()
            print(f"General summary generation took {time() - summary_start:.1f} seconds")
            
            # Process conclusions
            print("\nGenerating conclusions...")
            conclusions_start = time()
            conclusions_content = "\n\n".join([doc.page_content for doc in general_docs[1:3]])
            conclusions = llm.invoke(
                "List the key takeaways, findings, and important discoveries (max 250 words). Format as numbered points with descriptive headers: " + conclusions_content
            ).content.strip()
            print(f"Conclusions generation took {time() - conclusions_start:.1f} seconds")
            
            # Process sections
            print("\nGenerating section summaries...")
            sections_start = time()
            sections_content = "\n\n".join([doc.page_content for doc in general_docs[2:]])
            sections = llm.invoke(
                "List up to 5 main sections or topics discussed, with a brief summary of each (max 100 words per section). Format as numbered points with clear section titles: " + sections_content
            ).content.strip()
            print(f"Sections generation took {time() - sections_start:.1f} seconds")
            
            # Format the complete summary with proper visual structure
            complete_summary = f"""===========================================
           DOCUMENT SUMMARY
===========================================

GENERAL SUMMARY
-------------------------------------------
{general_summary}

MAIN CONCLUSIONS AND FINDINGS
-------------------------------------------
{conclusions}

MAIN SECTIONS
-------------------------------------------
{sections}

==========================================="""
            
        else:
            # For OpenAI, use the original single-call approach with formatted prompt
            doc_content = "\n\n".join([doc.page_content for doc in general_docs])
            comprehensive_prompt = """Based on the provided content, generate a comprehensive summary with the following sections:

===========================================
           DOCUMENT SUMMARY
===========================================

GENERAL SUMMARY
-------------------------------------------
Provide a concise summary of the document's main goals and purpose (max 200 words).

MAIN CONCLUSIONS AND FINDINGS
-------------------------------------------
List the key takeaways, findings, and important discoveries (max 250 words).
Format as numbered points with descriptive headers.

MAIN SECTIONS
-------------------------------------------
List up to 5 main sections or topics discussed, with a brief summary of each (max 100 words per section).
Format as numbered points with clear section titles.

===========================================

Content for analysis:
{content}""".format(content=doc_content)

            print("\nGenerating comprehensive summary...")
            summary_start = time()
            complete_summary = llm.invoke(comprehensive_prompt).content.strip()
            print(f"Summary generation took {time() - summary_start:.1f} seconds")
        
        # Save summary to file
        summary_file = summary_path / f"{pdf_name}_Summary.txt"
        print(f"\nSaving summary to: {summary_file}")
        with open(summary_file, "w", encoding='utf-8') as f:
            f.write(complete_summary)
        
        print(f"Summary successfully saved to: {summary_file}")
        print(f"Total summarization time: {time() - start_time:.1f} seconds")
        
        return complete_summary
    
    except Exception as e:
        print(f"Error generating summary for {pdf_name}: {str(e)}")
        raise

def process_and_save_pdf(pdf_path, store_name, provider="openai", model_name=None, generate_summaries=True):
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
    
    # Generate summaries if requested
    complete_summary = None
    if generate_summaries:
        try:
            complete_summary = summarize_sections(
                vectorstore, 
                store_name, 
                provider, 
                model_name
            )
        except Exception as e:
            print(f"Warning: Could not generate summary: {str(e)}")
    
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
    
    return vectorstore, visual_elements, complete_summary

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
        vectorstore, visual_elements, complete_summary = process_and_save_pdf(pdf_path, pdf_name)
        
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

def batch_process_pdfs(provider="openai", model_name=None, generate_summaries=True):
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
            vectorstore, visual_elements, complete_summary = process_and_save_pdf(
                str(pdf_path), 
                doc_id, 
                provider, 
                model_name,
                generate_summaries
            )
            
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

def query_with_ollama(doc_id, query, model_name="mistral", temperature=0.7):
    """Query a document using Ollama LLM with vector store context"""
    try:
        # Load vector store with Ollama embeddings
        print(f"\nLoading vector store for document: {doc_id}")
        vectorstore = load_vector_store(
            doc_id,
            provider="ollama",
            model_name="nomic-embed-text"  # Use the same embedding model that was likely used to create the store
        )
        
        # Initialize Ollama LLM
        print(f"Initializing Ollama LLM with model: {model_name}")
        llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_ctx=4096,  # Larger context window for more comprehensive responses
            num_gpu=1 if verify_gpu_availability() else 0,
            num_thread=8,
            stop=["</s>"],
            repeat_penalty=1.1,
            top_k=10,
            top_p=0.8
        )
        
        # Create retriever with higher k for more context
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Enhanced prompt template for better context utilization
        template = """You are a knowledgeable assistant with access to a document's content through a vector database.
        Use the following context from the document to answer the question accurately and comprehensively.
        If the context doesn't contain enough information to answer fully, say so.
        
        Context from document:
        {context}
        
        Question: {question}
        
        Detailed Answer: """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create and execute the chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("Querying document...")
        response = chain.invoke(query)
        return response
        
    except Exception as e:
        print(f"Error querying document: {str(e)}")
        return None

def query_all_documents(query, model_name="mistral", temperature=0.7):
    """Query across all documents in the data directory using Ollama LLM"""
    try:
        # Initialize Ollama LLM
        print(f"Initializing Ollama LLM with model: {model_name}")
        llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_ctx=4096,
            num_gpu=1 if verify_gpu_availability() else 0,
            num_thread=8,
            stop=["</s>"],
            repeat_penalty=1.1,
            top_k=10,
            top_p=0.8
        )
        
        # First, check if the query is asking for specific data or numbers
        initial_prompt = """You are a helpful AI assistant. Analyze the following question to determine if it requires specific data, numbers, statistics, or factual information from documents.

        If the question asks for ANY of the following, respond with 'NEED_SEARCH':
        1. Specific numbers, statistics, or measurements
        2. Historical data from a particular time period
        3. Information about specific countries, organizations, or entities
        4. Comparisons between different items or time periods
        5. Details from reports, studies, or documents
        
        Only answer from your own knowledge if the question is about general concepts, definitions, or widely known facts.
        
        Question: {question}
        
        Decision: """
        
        print("Analyzing if question requires document search...")
        initial_response = llm.invoke(initial_prompt.format(question=query))
        initial_answer = initial_response.content.strip()
        
        if not initial_answer.startswith("NEED_SEARCH"):
            print("Question can be answered with model's knowledge.")
            return initial_answer
        
        print("Question requires document search.")
        
        # Find all document directories in the data folder
        data_dir = Path("./data")
        if not data_dir.exists():
            raise ValueError("No data directory found. Please process some documents first.")
        
        doc_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if not doc_dirs:
            raise ValueError("No processed documents found in the data directory.")
        
        print(f"\nFound {len(doc_dirs)} document(s) to search")
        
        # Initialize embeddings
        print("Initializing embeddings with nomic-embed-text")
        embeddings = get_embeddings("ollama", "nomic-embed-text")
        
        # Search each document and collect results
        all_results = []
        for doc_dir in doc_dirs:
            try:
                collection_name = sanitize_collection_name(doc_dir.name)
                vectorstore = Chroma(
                    persist_directory=str(doc_dir),
                    collection_name=collection_name,
                    embedding_function=embeddings
                )
                # Get results from this document
                results = vectorstore.similarity_search_with_score(
                    query,
                    k=2  # Get top 2 results from each document
                )
                for doc, score in results:
                    all_results.append((doc, score, doc_dir.name))
            except Exception as e:
                print(f"Warning: Could not search document {doc_dir.name}: {str(e)}")
                continue
        
        if not all_results:
            return "No relevant information found in any document."
        
        # Sort results by similarity score (lower score is better)
        all_results.sort(key=lambda x: x[1])
        
        # Take top 5 results across all documents
        top_results = all_results[:5]
        
        # Create context from top results
        context_parts = []
        for doc, score, doc_name in top_results:
            context_parts.append(f"From document '{doc_name}':\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt template for data-focused search
        template = """You are a helpful assistant with access to multiple documents. Answer the following question using the provided context.

        Important Instructions:
        1. ALWAYS check the context for specific data, numbers, or statistics first
        2. If you find relevant data in the context, use it instead of your general knowledge
        3. Be precise with numbers and statistics from the documents
        4. If you find a table or structured data, extract and present the exact values
        5. Cite the specific numbers or data points you find
        6. If you can't find the exact data in the context, say so clearly
        
        Context from documents:
        {context}
        
        Question: {question}
        
        Focused Answer: """
        
        # Create prompt with context and query
        prompt = template.format(context=context, question=query)
        
        # Get response from LLM
        print("Generating response from search results...")
        response = llm.invoke(prompt)
        
        return response.content
        
    except Exception as e:
        print(f"Error searching documents: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDFs with AI-powered analysis")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai",
                      help="Choose the AI provider (default: openai)")
    parser.add_argument("--model", type=str, default=None,
                      help="Specify the model name (default: provider's default model)")
    parser.add_argument("--force-cpu", action="store_true",
                      help="Force CPU usage even if GPU is available")
    parser.add_argument("--query", action="store_true",
                      help="Enter query mode after processing")
    parser.add_argument("--doc-id", type=str,
                      help="Document ID to query (optional in query mode, if not provided will search all documents)")
    
    args = parser.parse_args()
    
    if args.provider.lower() == "ollama" and not args.force_cpu:
        # Verify GPU at startup
        if verify_gpu_availability():
            print("GPU support verified for Ollama")
        else:
            print("Warning: GPU support not verified. Use --force-cpu to suppress this warning.")
            if input("Continue anyway? (y/n): ").lower() != 'y':
                sys.exit(1)
    
    if not args.query:
        # Normal processing mode
        generate_summaries = input("\nDo you want the program to generate detailed summaries of the processed PDFs? (Y/N): ").lower() == 'y'
        if generate_summaries:
            print("Summary generation enabled. This may increase processing time significantly.")
        else:
            print("Summary generation disabled. Processing will be faster.")
        
        batch_process_pdfs(provider=args.provider, model_name=args.model, generate_summaries=generate_summaries)
    else:
        # Query mode
        print("\nEntering query mode")
        if args.doc_id:
            print(f"Searching in document: {args.doc_id}")
        else:
            print("Searching across all documents")
        print("Type 'exit' to quit")
        
        while True:
            query = input("\nEnter your question: ").strip()
            if query.lower() == 'exit':
                break
            
            if args.doc_id:
                # Search single document
                response = query_with_ollama(
                    args.doc_id,
                    query,
                    model_name=args.model or "mistral"
                )
            else:
                # Search all documents
                response = query_all_documents(
                    query,
                    model_name=args.model or "mistral"
                )
            
            if response:
                print("\nAnswer:")
                print(response)
            else:
                print("No response received. Please try again.")
