import spacy
from typing import List, Dict, Tuple
import re
from dataclasses import dataclass
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

@dataclass
class DocumentSection:
    """Represents a semantic section of a document"""
    title: str
    content: str
    level: int  # Heading level
    start_idx: int
    end_idx: int

class SemanticChunker:
    def __init__(self):
        # Load spaCy model for NLP tasks
        self.nlp = spacy.load("en_core_web_sm")
        
        # Common document section markers
        self.section_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headers
            r'^(\d+\.?\d*\.?\s+.+)$',  # Numbered sections
            r'^([A-Z][^.!?]*(?:[.!?]|$))',  # Capitalized phrases at start of line
        ]
    
    def identify_sections(self, text: str) -> List[DocumentSection]:
        """Identify document sections based on headers and structural markers"""
        sections = []
        lines = text.split('\n')
        current_section = None
        
        for idx, line in enumerate(lines):
            # Check for section headers
            for pattern in self.section_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    # If we had a previous section, close it
                    if current_section:
                        current_section.end_idx = idx
                        sections.append(current_section)
                    
                    # Start new section
                    level = self._determine_heading_level(line)
                    current_section = DocumentSection(
                        title=match.group(1),
                        content="",
                        level=level,
                        start_idx=idx,
                        end_idx=-1
                    )
                    break
            
            # Add content to current section
            if current_section:
                current_section.content += line + "\n"
        
        # Close final section
        if current_section:
            current_section.end_idx = len(lines)
            sections.append(current_section)
        
        return sections
    
    def create_semantic_chunks(self, text: str, min_chunk_size: int = 100, max_chunk_size: int = 1000) -> List[str]:
        """Create chunks based on semantic sections while respecting size constraints"""
        sections = self.identify_sections(text)
        chunks = []
        
        for section in sections:
            # Process each section into coherent chunks
            section_chunks = self._process_section(section, min_chunk_size, max_chunk_size)
            chunks.extend(section_chunks)
        
        return chunks
    
    def _process_section(self, section: DocumentSection, min_size: int, max_size: int) -> List[str]:
        """Process a section into coherent chunks"""
        chunks = []
        current_chunk = section.title + "\n"
        sentences = sent_tokenize(section.content)
        
        for sentence in sentences:
            # Check if adding sentence would exceed max size
            if len(current_chunk) + len(sentence) > max_size and len(current_chunk) >= min_size:
                chunks.append(current_chunk)
                current_chunk = section.title + "\n" + sentence  # Start new chunk with section context
            else:
                current_chunk += " " + sentence
        
        # Add final chunk if it meets minimum size
        if len(current_chunk) >= min_size:
            chunks.append(current_chunk)
        
        return chunks
    
    def _determine_heading_level(self, line: str) -> int:
        """Determine the heading level based on formatting"""
        # Count markdown style headers
        if line.startswith('#'):
            return len(re.match(r'^#+', line).group(0))
        
        # Count numbered section depth
        if re.match(r'^\d+', line):
            return len(line.split('.'))
        
        return 1  # Default level for other types of headers
    
    def analyze_semantic_coherence(self, chunk: str) -> float:
        """Analyze the semantic coherence of a chunk"""
        doc = self.nlp(chunk)
        
        # Calculate semantic coherence based on:
        # 1. Sentence similarity
        # 2. Topic consistency
        # 3. Reference resolution
        
        coherence_score = 0.0
        sentences = list(doc.sents)
        
        if len(sentences) < 2:
            return 1.0
        
        # Calculate similarity between consecutive sentences
        similarities = []
        for i in range(len(sentences) - 1):
            similarity = sentences[i].similarity(sentences[i + 1])
            similarities.append(similarity)
        
        coherence_score = sum(similarities) / len(similarities)
        return coherence_score

def enhance_chunk_metadata(chunk: str, nlp) -> Dict:
    """Enhance chunk with additional metadata"""
    doc = nlp(chunk)
    
    return {
        'text': chunk,
        'entities': [(ent.text, ent.label_) for ent in doc.ents],
        'noun_phrases': [np.text for np in doc.noun_chunks],
        'key_terms': [token.text for token in doc if token.is_stop == False and token.is_punct == False],
        'word_count': len([token for token in doc if not token.is_punct])
    }