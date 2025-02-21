import networkx as nx
from typing import List, Dict, Set, Tuple
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DocumentNode:
    """Represents a document in the relationship graph"""
    id: str
    title: str
    chunks: List[str]
    metadata: Dict
    embedding: np.ndarray
    creation_date: datetime

@dataclass
class DocumentRelation:
    """Represents a relationship between documents"""
    source_id: str
    target_id: str
    relation_type: str
    strength: float
    shared_entities: Set[str]
    shared_topics: Set[str]

class DocumentMapper:
    def __init__(self, embedding_model):
        self.nlp = spacy.load("en_core_web_sm")
        self.embedding_model = embedding_model
        self.document_graph = nx.MultiDiGraph()
        
    def add_document(self, doc_id: str, title: str, chunks: List[str], metadata: Dict) -> None:
        """Add a document to the relationship graph"""
        # Create document embedding
        doc_embedding = self._create_document_embedding(chunks)
        
        # Create document node
        doc_node = DocumentNode(
            id=doc_id,
            title=title,
            chunks=chunks,
            metadata=metadata,
            embedding=doc_embedding,
            creation_date=datetime.now()
        )
        
        # Add to graph
        self.document_graph.add_node(doc_id, document=doc_node)
        
        # Update relationships with existing documents
        self._update_relationships(doc_id)
    
    def _create_document_embedding(self, chunks: List[str]) -> np.ndarray:
        """Create a document-level embedding from chunks"""
        chunk_embeddings = self.embedding_model.embed_documents(chunks)
        return np.mean(chunk_embeddings, axis=0)
    
    def _update_relationships(self, doc_id: str) -> None:
        """Update relationships between the new document and existing documents"""
        new_doc = self.document_graph.nodes[doc_id]['document']
        
        for other_id in self.document_graph.nodes():
            if other_id != doc_id:
                other_doc = self.document_graph.nodes[other_id]['document']
                
                # Find relationships
                relations = self._identify_relationships(new_doc, other_doc)
                
                # Add relationships to graph
                for relation in relations:
                    self.document_graph.add_edge(
                        relation.source_id,
                        relation.target_id,
                        relation_type=relation.relation_type,
                        strength=relation.strength,
                        shared_entities=relation.shared_entities,
                        shared_topics=relation.shared_topics
                    )
    
    def _identify_relationships(self, doc1: DocumentNode, doc2: DocumentNode) -> List[DocumentRelation]:
        """Identify relationships between two documents"""
        relations = []
        
        # Calculate similarity score
        similarity = cosine_similarity(
            doc1.embedding.reshape(1, -1),
            doc2.embedding.reshape(1, -1)
        )[0][0]
        
        # Find shared entities and topics
        doc1_entities = self._extract_entities(doc1.chunks)
        doc2_entities = self._extract_entities(doc2.chunks)
        shared_entities = doc1_entities.intersection(doc2_entities)
        
        doc1_topics = self._extract_topics(doc1.chunks)
        doc2_topics = self._extract_topics(doc2.chunks)
        shared_topics = doc1_topics.intersection(doc2_topics)
        
        # Create relationships based on different criteria
        if similarity > 0.5:
            relations.append(DocumentRelation(
                source_id=doc1.id,
                target_id=doc2.id,
                relation_type="similar_content",
                strength=similarity,
                shared_entities=shared_entities,
                shared_topics=shared_topics
            ))
        
        if len(shared_entities) > 0:
            relations.append(DocumentRelation(
                source_id=doc1.id,
                target_id=doc2.id,
                relation_type="shared_entities",
                strength=len(shared_entities) / max(len(doc1_entities), len(doc2_entities)),
                shared_entities=shared_entities,
                shared_topics=shared_topics
            ))
        
        # Check temporal relationships
        if abs((doc1.creation_date - doc2.creation_date).days) < 30:
            relations.append(DocumentRelation(
                source_id=doc1.id,
                target_id=doc2.id,
                relation_type="temporal_proximity",
                strength=1.0,
                shared_entities=shared_entities,
                shared_topics=shared_topics
            ))
        
        return relations
    
    def _extract_entities(self, chunks: List[str]) -> Set[str]:
        """Extract named entities from chunks"""
        entities = set()
        for chunk in chunks:
            doc = self.nlp(chunk)
            for ent in doc.ents:
                entities.add((ent.text, ent.label_))
        return entities
    
    def _extract_topics(self, chunks: List[str]) -> Set[str]:
        """Extract main topics from chunks"""
        topics = set()
        for chunk in chunks:
            doc = self.nlp(chunk)
            # Extract noun phrases and key terms as topics
            topics.update([np.text for np in doc.noun_chunks])
        return topics
    
    def get_related_documents(self, doc_id: str, relation_type: str = None, min_strength: float = 0.0) -> List[Tuple[str, float]]:
        """Get related documents with optional filtering"""
        related_docs = []
        
        for edge in self.document_graph.edges(doc_id, data=True):
            _, target_id, data = edge
            
            if (relation_type is None or data['relation_type'] == relation_type) and \
               data['strength'] >= min_strength:
                related_docs.append((target_id, data['strength']))
        
        return sorted(related_docs, key=lambda x: x[1], reverse=True)
    
    def visualize_relationships(self, output_file: str = 'document_relationships.html') -> None:
        """Create a visualization of document relationships"""
        import pyvis.network as net
        
        # Check if we have enough documents to visualize relationships
        if len(self.document_graph.nodes()) < 1:
            print("No documents to visualize.")
            return
        
        if len(self.document_graph.nodes()) == 1:
            print("Only one document processed. No relationships to visualize.")
            
            # Create a simple visualization with just one node
            network = net.Network(height='750px', width='100%', bgcolor='#ffffff', font_color='black')
            
            # Add the single node
            node_id = list(self.document_graph.nodes())[0]
            doc = self.document_graph.nodes[node_id]['document']
            network.add_node(node_id, label=doc.title, title=f"Document: {doc.title}")
            
            # Save visualization
            try:
                network.save_graph(output_file)
                print(f"Created single-node visualization at: {output_file}")
            except Exception as e:
                print(f"Error saving visualization: {str(e)}")
            return
        
        # Create network visualization for multiple documents
        try:
            network = net.Network(height='750px', width='100%', bgcolor='#ffffff', font_color='black')
            
            # Add nodes
            for node_id in self.document_graph.nodes():
                doc = self.document_graph.nodes[node_id]['document']
                network.add_node(node_id, label=doc.title, title=f"Document: {doc.title}")
            
            # Add edges
            for edge in self.document_graph.edges(data=True):
                source, target, data = edge
                network.add_edge(
                    source, 
                    target,
                    title=f"Type: {data['relation_type']}\nStrength: {data['strength']:.2f}",
                    value=data['strength']
                )
            
            # Save visualization
            network.save_graph(output_file)
            print(f"Relationship visualization saved to: {output_file}")
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")
            print("This might be due to no valid relationships between documents.")
        