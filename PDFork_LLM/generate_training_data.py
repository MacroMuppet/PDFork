from pathlib import Path
import json
from datetime import datetime
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os
import nltk
nltk.download('punkt')  # Download the punkt tokenizer data

load_dotenv()

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
        return Ollama(
            model=model_name or "mistral",
            temperature=0.7
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
            model=model_name or "nomic-embed-text"
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def load_vector_store(doc_id, provider="openai", model_name=None):
    """Load vector store for a specific document"""
    load_path = Path("./data") / doc_id
    
    if not load_path.exists():
        raise ValueError(f"No vector store found for document: {doc_id}")
    
    embeddings = get_embeddings(provider, model_name)
    vectorstore = Chroma(
        persist_directory=str(load_path),
        embedding_function=embeddings
    )
    
    return vectorstore

def analyze_content_complexity(content):
    """Analyze content complexity to determine optimal number of questions"""
    # Count words and unique terms
    words = content.split()
    word_count = len(words)
    unique_words = len(set(words))
    
    # Calculate lexical density (ratio of unique words to total words)
    lexical_density = unique_words / word_count if word_count > 0 else 0
    
    # Count potential key indicators of complex content
    technical_indicators = sum(1 for word in words if len(word) > 8)  # Long words
    numerical_content = sum(1 for word in words if any(c.isdigit() for c in word))  # Numbers
    
    # Base number of questions on content length and complexity
    base_questions = max(3, min(10, word_count // 200))  # 1 question per ~200 words, min 3, max 10
    
    # Adjust based on complexity factors
    complexity_multiplier = 1.0
    if lexical_density > 0.6:  # High unique word ratio
        complexity_multiplier += 0.2
    if technical_indicators / word_count > 0.1:  # Many technical terms
        complexity_multiplier += 0.2
    if numerical_content / word_count > 0.05:  # Significant numerical content
        complexity_multiplier += 0.2
    
    return round(base_questions * complexity_multiplier)

def analyze_chunk_importance(llm, content):
    """Analyze the importance of a chunk to determine if it needs more questions"""
    prompt = """Analyze the following content and rate its importance from 1-5 based on these criteria:
    1. Contains key findings or conclusions (2 points)
    2. Introduces new important concepts (1 point)
    3. Contains numerical data or statistics (1 point)
    4. Presents critical arguments or evidence (1 point)
    
    Provide only the numerical score (1-5).
    
    Content: {content}
    
    Score:"""
    
    try:
        score = int(llm.predict(prompt.format(content=content)).strip())
        return min(5, max(1, score))  # Ensure score is between 1 and 5
    except:
        return 3  # Default to medium importance if scoring fails

def determine_optimal_questions(llm, content):
    """Determine the optimal number of questions for a content chunk"""
    # Get base number from content complexity
    base_questions = analyze_content_complexity(content)
    
    # Adjust based on content importance
    importance_score = analyze_chunk_importance(llm, content)
    
    # Calculate final number of questions
    # Importance score of 1-5 will modify base questions by -20% to +20%
    importance_multiplier = 0.8 + (importance_score * 0.1)  # Maps 1-5 to 0.9-1.3
    
    final_questions = round(base_questions * importance_multiplier)
    
    # Ensure reasonable bounds
    return max(2, min(15, final_questions))

def generate_questions(llm, content, num_questions=5):
    """Generate diverse questions based on the content"""
    # Determine optimal number of questions if not specified
    if num_questions == 5:  # Default value
        num_questions = determine_optimal_questions(llm, content)
        print(f"Dynamically determined to generate {num_questions} questions for this chunk")
    
    prompt = f"""Based on the following content, generate {num_questions} diverse and meaningful questions that could be asked about this content. 
    Focus on questions that test understanding of key concepts, findings, and relationships.
    Ensure questions are varied in complexity and cover different aspects:
    - Include both factual and analytical questions
    - Mix specific details and broader concepts
    - Include questions about relationships between ideas
    - If present, include questions about numerical data or statistics
    
    Format each question on a new line.
    Content: {content}
    
    Questions:"""
    
    questions = llm.predict(prompt).strip().split('\n')
    return [q.strip() for q in questions if q.strip()]

def generate_answer(llm, question, content):
    """Generate a detailed answer for a question based on the content"""
    prompt = f"""Based on the following content, provide a clear, accurate, and detailed answer to the question.
    Include specific details and examples from the content when relevant.
    
    Content: {content}
    
    Question: {question}
    
    Answer:"""
    
    return llm.predict(prompt).strip()

def split_into_sentence_chunks(text, sentences_per_chunk=3):
    """Split text into chunks of specified number of sentences."""
    try:
        # Simple preprocessing to handle common issues
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Use the standard punkt tokenizer (not punkt_tab)
        sentences = nltk.sent_tokenize(text)
        
        # Filter out empty or very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip().split()) > 3]
        
        # Group sentences into chunks
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ' '.join(sentences[i:i + sentences_per_chunk])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
        
        if not chunks:  # If no chunks were created, fall back to simple splitting
            return _fallback_split(text, sentences_per_chunk)
            
        return chunks
    except Exception as e:
        print(f"Warning: Standard sentence tokenization failed ({str(e)}), using fallback method")
        return _fallback_split(text, sentences_per_chunk)

def _fallback_split(text, sentences_per_chunk=3):
    """Fallback method for splitting text when NLTK tokenization fails."""
    # Clean the text first
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())  # Normalize whitespace
    
    # Split on common sentence endings
    sentences = []
    current = ""
    
    # Split on sentence endings while preserving the punctuation
    for part in text.replace('!', '.').replace('?', '.').split('.'):
        part = part.strip()
        if part:
            current += part + '.'
            if len(part) > 20:  # Only consider it a sentence if it's long enough
                sentences.append(current)
                current = ""
    
    if current:  # Add any remaining text
        sentences.append(current)
    
    # Group into chunks
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i + sentences_per_chunk])
        if len(chunk.split()) >= 10:  # Only add if chunk has at least 10 words
            chunks.append(chunk)
    
    return chunks

def create_training_dataset(doc_id, provider="openai", model_name=None, sentences_per_chunk=3):
    """Create a QnA training dataset from a document's vector store"""
    print(f"\nProcessing document: {doc_id}")
    
    # Load vector store
    try:
        vectorstore = load_vector_store(doc_id, provider, model_name)
    except Exception as e:
        print(f"Error loading vector store for {doc_id}: {str(e)}")
        return None
    
    # Initialize LLM
    llm = get_llm(provider, model_name)
    
    # Get all document chunks - using collection.get() instead of retriever
    try:
        # Get all chunks from the collection
        all_docs = vectorstore.get()
        if not all_docs or not all_docs['documents']:
            print(f"No content found in vector store for {doc_id}")
            return None
        
        chunks = all_docs['documents']
        metadatas = all_docs['metadatas']
        total_chunks = len(chunks)
        
        print(f"Found {total_chunks} content chunks")
        
        training_data = []
        
        for i, (chunk, metadata) in enumerate(zip(chunks, metadatas), 1):
            print(f"\nProcessing chunk {i}/{total_chunks}")
            
            if not chunk.strip():
                print(f"Skipping empty chunk {i}")
                continue
                
            try:
                # Split the chunk into sub-chunks
                sub_chunks = split_into_sentence_chunks(chunk, sentences_per_chunk)
                print(f"Split into {len(sub_chunks)} sub-chunks of {sentences_per_chunk} sentences each")
                
                for j, sub_chunk in enumerate(sub_chunks, 1):
                    if len(sub_chunk.split()) < 10:  # Skip very short chunks
                        print(f"Skipping short sub-chunk {j} (less than 10 words)")
                        continue
                        
                    # Determine number of questions based on content
                    num_questions = determine_optimal_questions(llm, sub_chunk)
                    print(f"Generating {num_questions} questions for sub-chunk {j}")
                    
                    # Generate questions for this sub-chunk
                    questions = generate_questions(llm, sub_chunk, num_questions)
                    
                    for question in questions:
                        # Generate detailed answer
                        answer = generate_answer(llm, question, sub_chunk)
                        
                        # Add to training data
                        training_data.append({
                            "question": question,
                            "answer": answer,
                            "metadata": {
                                "source_doc": doc_id,
                                "chunk_index": i,
                                "sub_chunk_index": j,
                                "chunk_content": sub_chunk,
                                "num_sentences": sentences_per_chunk,
                                "generated_at": datetime.now().isoformat(),
                                "original_metadata": metadata
                            }
                        })
                        print(f"Generated Q&A pair: {question[:50]}...")
            
            except Exception as e:
                print(f"Error processing chunk {i}: {str(e)}")
                continue
        
        # Save training data with enhanced metadata
        output_dir = Path("./training_data")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"{doc_id}_training_data.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump({
                "document_id": doc_id,
                "total_pairs": len(training_data),
                "total_chunks": total_chunks,
                "sentences_per_chunk": sentences_per_chunk,
                "generated_at": datetime.now().isoformat(),
                "provider": provider,
                "model": model_name,
                "training_pairs": training_data
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\nSaved {len(training_data)} training pairs to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error processing document {doc_id}: {str(e)}")
        return None

def process_all_documents(provider="openai", model_name=None, sentences_per_chunk=3, doc_ids=None):
    """Process documents in the data directory"""
    data_dir = Path("./data")
    if not data_dir.exists():
        print("No data directory found")
        return
    
    # Get all document directories
    all_doc_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if not all_doc_dirs:
        print("No processed documents found")
        return
    
    # Filter to specific documents if provided
    if doc_ids:
        doc_dirs = [d for d in all_doc_dirs if d.name in doc_ids]
        if not doc_dirs:
            print(f"None of the specified documents {doc_ids} were found")
            print("Available documents:")
            for d in all_doc_dirs:
                print(f"- {d.name}")
            return
    else:
        doc_dirs = all_doc_dirs
    
    print(f"Found {len(doc_dirs)} documents to process")
    print(f"Using {provider} provider with model: {model_name or 'default'}")
    print(f"Generating Q&A pairs every {sentences_per_chunk} sentences")
    
    results = []
    for doc_dir in doc_dirs:
        try:
            doc_id = doc_dir.name
            output_file = create_training_dataset(
                doc_id=doc_id,
                provider=provider,
                model_name=model_name,
                sentences_per_chunk=sentences_per_chunk
            )
            if output_file:
                results.append((doc_id, output_file))
        except Exception as e:
            print(f"Error processing document {doc_id}: {str(e)}")
            continue
    
    print("\nProcessing complete!")
    print(f"Successfully processed {len(results)}/{len(doc_dirs)} documents")
    for doc_id, output_file in results:
        print(f"- {doc_id}: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training data from processed documents")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai",
                      help="Choose the AI provider (default: openai)")
    parser.add_argument("--model", type=str, default=None,
                      help="Specify the model name (default: provider's default model)")
    parser.add_argument("--sentences", type=int, default=3,
                      help="Number of sentences per Q&A chunk (default: 3)")
    parser.add_argument("--docs", nargs="+", help="Specific document IDs to process (space-separated)")
    
    args = parser.parse_args()
    
    process_all_documents(
        provider=args.provider,
        model_name=args.model,
        sentences_per_chunk=args.sentences,
        doc_ids=args.docs
    ) 