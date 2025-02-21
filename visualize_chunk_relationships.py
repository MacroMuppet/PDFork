from pathlib import Path
import json
from datetime import datetime
import numpy as np
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import pyvis.network as net
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from PDFLCEmbed_semantchunk_mapper import sanitize_collection_name, verify_gpu_availability
import argparse
from collections import Counter
import re

class ChunkVisualizer:
    def __init__(self, provider="ollama", model_name="nomic-embed-text"):
        """Initialize the chunk visualizer with specified provider and model"""
        self.provider = provider
        self.model_name = model_name
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize Mistral LLM for topic analysis using the new OllamaLLM class
        self.llm = OllamaLLM(
            model="mistral",
            temperature=0.0  # Keep it deterministic for topic generation
        )
        
        # Common acronyms and their proper capitalization
        self.acronyms = {
            'us': 'US',
            'usa': 'USA',
            'un': 'UN',
            'eu': 'EU',
            'uk': 'UK',
            'nato': 'NATO',
            'doe': 'DOE',
            'dod': 'DOD',
            'nasa': 'NASA',
            'fbi': 'FBI',
            'cia': 'CIA',
            'nsa': 'NSA',
            'epa': 'EPA',
            'nrc': 'NRC',
            'iaea': 'IAEA',
            'leu': 'LEU',
            'heu': 'HEU',
            'nnsa': 'NNSA',
        }
        
        # Get embeddings model
        if provider.lower() == "openai":
            self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        else:
            gpu_available = verify_gpu_availability()
            self.embeddings = OllamaEmbeddings(
                model=model_name,
                num_gpu=1 if gpu_available else 0
            )
    
    def format_topic(self, topic):
        """Format a topic with proper capitalization and acronym handling"""
        if not topic:
            return "Unknown"
            
        # Split into words
        words = str(topic).split()
        formatted_words = []
        
        for word in words:
            word_lower = word.lower()
            # Check if it's an acronym we know
            if word_lower in self.acronyms:
                formatted_words.append(self.acronyms[word_lower])
            # Check if it's an unknown acronym (all caps)
            elif word.isupper() and len(word) >= 2:
                formatted_words.append(word)
            # Check if it might be an unknown acronym (based on pattern)
            elif re.match(r'^[A-Za-z]{2,}$', word) and word.upper() == word:
                formatted_words.append(word.upper())
            # Title case for regular words, except for common lowercase words
            else:
                formatted_words.append(word.title())
        
        return " ".join(formatted_words) if formatted_words else "Unknown"
    
    def detect_acronyms(self, text):
        """Detect potential acronyms in text and add to known acronyms"""
        # Find potential acronyms using regex
        # Look for: (ACRONYM) or ACRONYM - full form
        acronym_patterns = [
            r'\(([A-Z]{2,})\)',  # (DOE)
            r'([A-Z]{2,})\s*[-–—]\s*([A-Za-z\s]+)',  # DOE - Department of Energy
            r'([A-Za-z\s]+)\s*[(]([A-Z]{2,})[)]'  # Department of Energy (DOE)
        ]
        
        for pattern in acronym_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) == 1:
                    acronym = match.group(1).lower()
                    if acronym not in self.acronyms:
                        self.acronyms[acronym] = match.group(1)
                elif len(match.groups()) == 2:
                    acronym = (match.group(1) if match.group(1).isupper() else match.group(2)).lower()
                    if acronym not in self.acronyms:
                        self.acronyms[acronym] = acronym.upper()
    
    def extract_topics(self, text):
        """Extract main topics from text using enhanced NLP analysis"""
        # First, detect any new acronyms in the text
        self.detect_acronyms(text)
        
        doc = self.nlp(text)
        topics = []
        
        # Get noun phrases with context
        for chunk in doc.noun_chunks:
            # Skip very long phrases
            if len(chunk.text.split()) <= 4:
                # Check if the phrase contains any of our known acronyms
                contains_acronym = any(acr.lower() in chunk.text.lower() for acr in self.acronyms.values())
                if contains_acronym:
                    # Format with acronym handling
                    topics.append(self.format_topic(chunk.text))
                else:
                    # Add regular noun phrases
                    topics.append(chunk.text.lower())
        
        # Get named entities with their types
        for ent in doc.ents:
            # Format entity based on its type
            if ent.label_ in ['ORG', 'GPE', 'PERSON']:
                # Organizations, locations, and persons get special formatting
                topics.append(self.format_topic(ent.text))
            elif ent.label_ in ['DATE', 'TIME', 'MONEY', 'PERCENT']:
                # Skip temporal and numerical entities unless they're part of a larger phrase
                continue
            else:
                # Other entities are added as is
                topics.append(ent.text.lower())
        
        # Add standalone acronyms found in text
        words = text.split()
        for word in words:
            if word.upper() == word and len(word) >= 2 and word.isalpha():
                topics.append(word)
        
        # Remove duplicates and sort
        unique_topics = []
        seen = set()
        for topic in topics:
            topic_lower = topic.lower()
            if topic_lower not in seen:
                seen.add(topic_lower)
                unique_topics.append(topic)
        
        return sorted(unique_topics)
    
    def extract_context(self, text, topic):
        """Extract relevant context for a topic from the text"""
        if not text or not topic:
            return None
            
        try:
            doc = self.nlp(str(text))
            topic_lower = str(topic).lower()
            
            # Find the sentence containing the topic
            context_sentence = None
            for sent in doc.sents:
                if topic_lower in sent.text.lower():
                    context_sentence = sent
                    break
            
            if not context_sentence:
                return None
            
            # Extract key information around the topic
            context_info = []
            
            try:
                # Look for specific patterns
                # 1. "of [something]" pattern
                of_match = re.search(f"{re.escape(topic_lower)} of ([^.,:;]+)", context_sentence.text.lower())
                if of_match:
                    context_info.append(f"of {self.format_topic(of_match.group(1))}")
                
                # 2. "by [someone/org]" pattern
                by_match = re.search(f"{re.escape(topic_lower)} by ([^.,:;]+)", context_sentence.text.lower())
                if by_match:
                    context_info.append(f"by {self.format_topic(by_match.group(1))}")
                
                # 3. "for [purpose]" pattern
                for_match = re.search(f"{re.escape(topic_lower)} for ([^.,:;]+)", context_sentence.text.lower())
                if for_match:
                    context_info.append(f"for {self.format_topic(for_match.group(1))}")
                
                # 4. Look for related entities in the same sentence
                related_ents = []
                for ent in context_sentence.ents:
                    if ent.text and ent.text.lower() != topic_lower:
                        if ent.label_ in ['ORG', 'GPE']:
                            related_ents.append(self.format_topic(ent.text))
                        elif ent.label_ == 'PERSON':
                            related_ents.append(f"by {self.format_topic(ent.text)}")
                
                if related_ents:
                    context_info.extend(related_ents[:2])  # Limit to 2 related entities
                
                # 5. Look for key verbs associated with the topic
                main_verb = None
                topic_token = None
                for token in context_sentence:
                    if topic_lower in token.text.lower():
                        topic_token = token
                        break
                
                if topic_token:
                    # Find the main verb associated with the topic
                    for token in context_sentence:
                        if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                            main_verb = token.text
                            break
                    
                    if main_verb:
                        context_info.insert(0, main_verb.lower())
                
                return context_info if context_info else None
                
            except Exception as e:
                print(f"Warning: Error extracting context patterns: {str(e)}")
                return None
                
        except Exception as e:
            print(f"Warning: Error in extract_context: {str(e)}")
            return None

    def determine_main_topic(self, text):
        """Determine the main topic of a chunk of text with improved categorization and context"""
        try:
            # Extract all topics
            topics = self.extract_topics(text)
            
            if not topics:
                return "Miscellaneous"
            
            # Count topic occurrences and get context
            topic_info = {}
            doc = self.nlp(str(text).lower())
            
            for topic in topics:
                if not topic:
                    continue
                    
                try:
                    # Count occurrences considering variations
                    topic_lower = str(topic).lower()
                    count = len([t for t in doc if topic_lower in t.text.lower()])
                    
                    # Give extra weight to topics with acronyms
                    if any(acr.lower() in topic_lower for acr in self.acronyms.values()):
                        count *= 1.5
                    
                    # Give extra weight to named entities
                    doc_topic = self.nlp(str(topic))
                    if len(doc_topic.ents) > 0:
                        count *= 1.3
                    
                    # Get context for the topic
                    context = self.extract_context(text, topic)
                    
                    topic_info[topic] = {
                        'count': count,
                        'context': context
                    }
                except Exception as e:
                    print(f"Warning: Error processing topic '{topic}': {str(e)}")
                    continue
            
            if not topic_info:
                return "Miscellaneous"
            
            # Sort topics by count
            sorted_topics = sorted(topic_info.items(), key=lambda x: x[1]['count'], reverse=True)
            
            # If we have multiple significant topics
            if len(sorted_topics) > 1:
                topic1, info1 = sorted_topics[0]
                topic2, info2 = sorted_topics[1]
                
                # Format topics with context
                topic1_formatted = self.format_topic_with_context(topic1, info1['context'])
                topic2_formatted = self.format_topic_with_context(topic2, info2['context'])
                
                # If topics are similarly important (within 20% of each other)
                if info2['count'] >= info1['count'] * 0.8:
                    # Check if one topic is an acronym and the other is related
                    if any(acr in str(topic1).upper() for acr in self.acronyms.values()) and \
                       any(acr in str(topic2).upper() for acr in self.acronyms.values()):
                        return f"{topic1_formatted} & {topic2_formatted}"
                    # If one has an acronym, put it first
                    elif any(acr in str(topic1).upper() for acr in self.acronyms.values()):
                        return f"{topic1_formatted}: {topic2_formatted}"
                    elif any(acr in str(topic2).upper() for acr in self.acronyms.values()):
                        return f"{topic2_formatted}: {topic1_formatted}"
                    # Otherwise, combine with a separator
                    else:
                        return f"{topic1_formatted} - {topic2_formatted}"
            
            # If we only have one significant topic
            topic, info = sorted_topics[0]
            return self.format_topic_with_context(topic, info['context'])
            
        except Exception as e:
            print(f"Warning: Error in determine_main_topic: {str(e)}")
            return "Miscellaneous"

    def format_topic_with_context(self, topic, context):
        """Format a topic with its context"""
        if not topic:
            return "Unknown"
            
        # Format the main topic
        formatted_topic = self.format_topic(str(topic))
        
        if not context:
            return formatted_topic
        
        try:
            # Add context in brackets
            context_str = "; ".join(str(c) for c in context if c)
            if len(context_str) > 50:  # Truncate if too long
                context_str = context_str[:47] + "..."
            
            return f"{formatted_topic} [{context_str}]"
        except Exception as e:
            print(f"Warning: Error formatting topic context: {str(e)}")
            return formatted_topic

    def find_chunk_relationships(self, chunks, threshold=0.5):
        """Find relationships between chunks based on embeddings similarity"""
        # Get embeddings for all chunks
        chunk_embeddings = self.embeddings.embed_documents(chunks)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(chunk_embeddings)
        
        # Find relationships above threshold
        relationships = []
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                similarity = similarity_matrix[i][j]
                if similarity > threshold:
                    # Extract shared topics
                    topics_i = set(self.extract_topics(chunks[i]))
                    topics_j = set(self.extract_topics(chunks[j]))
                    shared_topics = topics_i.intersection(topics_j)
                    
                    relationships.append({
                        'source': i,
                        'target': j,
                        'similarity': similarity,
                        'shared_topics': list(shared_topics)
                    })
        
        return relationships
    
    def create_preview_label(self, text, max_length=50):
        """Create a concise preview label from text"""
        # Clean the text
        text = text.replace('\n', ' ').strip()
        
        # Get the first sentence
        doc = self.nlp(text)
        first_sentence = next(doc.sents).text.strip()
        
        # Truncate if too long
        if len(first_sentence) > max_length:
            # Try to cut at a word boundary
            truncated = first_sentence[:max_length].rsplit(' ', 1)[0]
            return f"{truncated}..."
        
        return first_sentence

    def analyze_chunk_with_llm(self, chunk):
        """Use Mistral LLM to analyze a chunk and return a human-readable topic"""
        prompt = """Analyze the following text and provide a clear, human-readable topic that captures its main subject matter.
        The topic should be:
        1. Concise (2-6 words)
        2. Use proper capitalization for names and acronyms
        3. Include key relationships or context if relevant
        4. Be understandable to a general audience
        
        For example:
        - Instead of "DOE LEU" use "DOE Uranium Enrichment Program"
        - Instead of "Policy Implementation" use "Nuclear Policy Changes by DOE"
        
        Text to analyze:
        {text}
        
        Topic:"""
        
        try:
            # Use invoke instead of predict
            topic = self.llm.invoke(prompt.format(text=chunk)).strip()
            
            # Clean up the response
            # Remove any common prefixes that LLMs might add
            topic = re.sub(r'^(Topic:|Main Topic:|The topic is:|This text discusses)', '', topic, flags=re.IGNORECASE)
            # Remove any quotes around the topic
            topic = re.sub(r'^["\']+|["\']+$', '', topic.strip())
            
            # Ensure proper capitalization of known acronyms
            for acronym, proper_form in self.acronyms.items():
                # Use word boundaries to avoid partial matches
                topic = re.sub(r'\b' + re.escape(acronym) + r'\b', proper_form, topic, flags=re.IGNORECASE)
            
            return topic if topic else "Miscellaneous Content"
        except Exception as e:
            print(f"Warning: Error getting LLM topic: {str(e)}")
            # Fall back to the original topic determination method
            return self.determine_main_topic(chunk)

    def format_node_label(self, label, max_chars_per_line=20):
        """Format a node label with newlines for better readability"""
        if not label:
            return "Unknown"
            
        # Split into words and join with spaces
        words = label.split()
        current_line = []
        lines = []
        line_length = 0
        
        for word in words:
            if line_length + len(word) > max_chars_per_line and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                line_length = len(word)
            else:
                current_line.append(word)
                line_length += len(word) + (1 if current_line else 0)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Join with actual newlines instead of HTML breaks
        return '\n'.join(lines)

    def create_visualization(self, doc_id, output_file=None):
        """Create visualization for chunks in a document"""
        # Load vector store
        load_path = Path("./data") / doc_id
        if not load_path.exists():
            raise ValueError(f"No vector store found for document: {doc_id}")
        
        # Use sanitized collection name
        collection_name = sanitize_collection_name(doc_id)
        print(f"Using collection name: {collection_name}")
        
        # Load vector store
        vectorstore = Chroma(
            persist_directory=str(load_path),
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        
        # Get chunks from collection
        collection = vectorstore._collection
        if not collection:
            raise ValueError(f"No collection found for {doc_id}")
        
        results = collection.get()
        if not results or not results['documents']:
            raise ValueError(f"No content found in vector store for {doc_id}")
        
        chunks = results['documents']
        metadatas = results['metadatas']
        
        print(f"Found {len(chunks)} chunks")
        
        # Find relationships between chunks
        relationships = self.find_chunk_relationships(chunks)
        
        # Create network visualization with dark blue theme
        network = net.Network(
            height='900px',
            width='100%',
            bgcolor='#1A1F2E',
            font_color='#FFB3B3',
            directed=False
        )
        
        # Set initial options for HTML support
        network.options = {
            "nodes": {
                "font": {
                    "multi": "html"
                }
            }
        }
        
        # Update node styling
        for i, (chunk, metadata) in enumerate(zip(chunks, metadatas)):
            # Get main topic using LLM analysis
            main_topic = self.analyze_chunk_with_llm(chunk)
            all_topics = self.extract_topics(chunk)
            
            # Format topics summary with context
            topics_with_context = []
            for topic in all_topics[:5]:  # Show top 5 topics
                context = self.extract_context(chunk, topic)
                if context:
                    topics_with_context.append(f"{topic} [{'; '.join(context)}]")
                else:
                    topics_with_context.append(topic)
            
            topics_summary = "\n".join(topics_with_context)
            
            # Create preview label from chunk content
            preview_label = self.analyze_chunk_with_llm(chunk)
            # Format the label with line breaks
            formatted_label = self.format_node_label(preview_label, max_chars_per_line=20)
            
            # Add node with updated styling for dark blue theme
            network.add_node(
                i,
                label=formatted_label,
                title=f"Chunk {i+1}\nMain Topic: {main_topic}\nTopics:\n{topics_summary}\n\nFull Preview: {chunk[:500]}...",
                size=30,
                shape='star',
                color={
                    'background': '#800000',
                    'border': '#FFFFFF',
                    'highlight': {
                        'background': '#A52A2A',
                        'border': '#F8F8FF'
                    }
                },
                font={
                    'size': 14,
                    'color': '#FFB3B3',
                    'face': 'arial',
                    'background': '#1D2230',
                    'strokeWidth': 0,
                    'multi': 'html',
                    'bold': False,
                    'maxWdt': 200,
                    'vadjust': -8,
                    'mod': 'box'
                }
            )
        
        # Update edge styling
        for rel in relationships:
            # Create edge label with shared topics
            topics_label = "\n".join(rel['shared_topics'][:3])
            if len(rel['shared_topics']) > 3:
                topics_label += "\n..."
            
            # Calculate thickness based on similarity score
            # Exponentially increase thickness for higher similarities
            base_thickness = 2  # minimum thickness
            similarity_factor = rel['similarity'] ** 2  # square the similarity to exaggerate differences
            thickness = base_thickness + (similarity_factor * 12)  # max thickness around 14 for similarity=1.0
            
            network.add_edge(
                rel['source'],
                rel['target'],
                value=rel['similarity'] * 5,
                width=thickness,  # Set the calculated thickness
                title=f"Similarity: {rel['similarity']:.2f}\n\nShared Topics:\n{topics_label}",
                color={
                    'color': '#454545',  # Muted grey for inactive edges
                    'highlight': '#800000',  # Deep maroon for highlighted edges
                    'opacity': 0.6,  # Reduced opacity for better contrast
                    'inherit': False
                }
            )
        
        # Update physics and styling options
        network.options.update({
            "physics": {
                "enabled": True,
                "forceAtlas2Based": {
                    "gravitationalConstant": -1500,  # Balanced between -2000 and -1000
                    "centralGravity": 0.05,         # Reduced from 0.1 to allow more spread
                    "springLength": 300,            # Increased from 200 to space out more
                    "springConstant": 0.06,         # Balanced between 0.05 and 0.08
                    "damping": 0.4,
                    "avoidOverlap": 0.8             # Increased from 0.5 for better spacing
                },
                "maxVelocity": 28,                  # Balanced between 25 and 30
                "minVelocity": 0.1,
                "solver": "forceAtlas2Based",
                "stabilization": {
                    "enabled": True,
                    "iterations": 1000,
                    "updateInterval": 25
                },
                "timestep": 0.3
            },
            "nodes": {
                "shape": "star",
                "size": 35,
                "font": {
                    "size": 14,
                    "face": "arial",
                    "color": "#FFB3B3",
                    "background": "#1D2230",
                    "strokeWidth": 0,
                    "multi": "html",
                    "bold": False,
                    "maxWdt": 200,
                    "vadjust": -8,
                    "mod": "box"
                },
                "color": {
                    "border": "#FFFFFF",
                    "background": "#800000",
                    "highlight": {
                        "border": "#F8F8FF",
                        "background": "#A52A2A"
                    }
                },
                "scaling": {
                    "min": 30,
                    "max": 40,
                    "label": {
                        "enabled": True,
                        "min": 14,
                        "max": 14
                    }
                },
                "margin": {
                    "top": 15,
                    "bottom": 15,
                    "left": 15,
                    "right": 15
                }
            },
            "edges": {
                "color": {
                    "color": "#454545",
                    "highlight": "#800000",
                    "hover": "#800000",
                    "opacity": 0.6,
                    "inherit": False
                },
                "smooth": {
                    "type": "continuous",
                    "roundness": 0.5
                },
                "length": 300,  # Increased from 200 to match springLength
                "width": 2,
                "selectionWidth": 8,
                "hoverWidth": 8,
                "scaling": {
                    "min": 2,
                    "max": 14
                }
            },
            "interaction": {
                "hover": True,
                "hoverConnectedEdges": True,
                "selectConnectedEdges": True,
                "navigationButtons": True,
                "keyboard": True,
                "hideEdgesOnDrag": False
            },
            "manipulation": {
                "enabled": False
            },
            "configure": {
                "enabled": False
            }
        })

        # Update node selection JavaScript
        network.html += """
        <script type="text/javascript">
            network.on("selectNode", function(params) {
                var selectedNode = params.nodes[0];
                var edges = network.body.edges;
                Object.values(edges).forEach(function(edge) {
                    if (edge.connected) {
                        var isConnected = edge.from.id === selectedNode || edge.to.id === selectedNode;
                        var currentWidth = edge.options.width || 2;  // Get current width or use default
                        edge.color = {
                            color: isConnected ? "#800000" : "#2D2D2D",
                            opacity: isConnected ? 1.0 : 0.2,  // More pronounced opacity difference
                            inherit: false
                        };
                        // If connected, make even thicker based on original width
                        edge.width = isConnected ? (currentWidth * 1.5) : (currentWidth * 0.5);
                    }
                });
                network.redraw();
            });

            network.on("deselectNode", function(params) {
                var edges = network.body.edges;
                Object.values(edges).forEach(function(edge) {
                    // Restore original width from the edge options
                    var originalWidth = edge.options.width || 2;
                    edge.color = {
                        color: "#454545",
                        opacity: 0.6,
                        inherit: false
                    };
                    edge.width = originalWidth;  // Restore the original similarity-based width
                });
                network.redraw();
            });
        </script>
        """
        
        # Save visualization
        if output_file is None:
            output_file = f"{doc_id}_chunk_relationships.html"
        
        network.save_graph(output_file)
        print(f"\nVisualization saved to: {output_file}")
        return output_file

def main():
    parser = argparse.ArgumentParser(description="Create visualization of chunk relationships within a document")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="ollama",
                      help="Choose the AI provider (default: ollama)")
    parser.add_argument("--model", type=str, default="nomic-embed-text",
                      help="Specify the embeddings model name (default: nomic-embed-text)")
    parser.add_argument("--doc", type=str, required=True,
                      help="Document ID to visualize")
    parser.add_argument("--output", type=str, default=None,
                      help="Output file name (default: {doc_id}_chunk_relationships.html)")
    parser.add_argument("--threshold", type=float, default=0.5,
                      help="Similarity threshold for showing relationships (default: 0.5)")
    
    args = parser.parse_args()
    
    try:
        visualizer = ChunkVisualizer(args.provider, args.model)
        output_file = visualizer.create_visualization(args.doc, args.output)
        print("Visualization created successfully!")
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 