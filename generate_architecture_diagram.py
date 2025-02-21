from diagrams import Diagram, Cluster
from diagrams.programming.language import Python
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.compute import Server
from diagrams.programming.framework import React
from diagrams.generic.storage import Storage
from diagrams.generic.compute import Rack

with Diagram("PDFork Architecture", show=False, direction="TB"):
    with Cluster("Input"):
        pdf = Storage("PDF Files")
        config = Storage("Configuration")

    with Cluster("Core Processing"):
        pe = Python("PDF Extraction")
        tc = Python("Text Chunking")
        ve = Python("Visual Extractor")
        sc = Python("Semantic Chunker")

    with Cluster("Vector Storage"):
        db = PostgreSQL("ChromaDB")
        vs = Storage("Vector Store")
        em = Python("Embedding Models")

    with Cluster("LLM Integration"):
        ol = Server("Ollama Interface")
        qa = Python("Query Analyzer")
        rc = Python("Response Generator")

    with Cluster("Visualization"):
        vd = React("Dashboard")
        cg = Python("Chunk Graph")
        rd = React("React Components")

    with Cluster("Training Data"):
        tg = Python("Training Generator")
        ca = Python("Complexity Analyzer")
        qg = Python("Question Generator")

    # Input connections
    pdf >> pe
    config >> ol
    config >> em

    # Core Processing flow
    pe >> tc
    pe >> ve
    tc >> sc
    sc >> vs

    # Vector Storage connections
    vs >> db
    em >> vs
    db >> qa

    # LLM Integration flow
    qa >> rc
    ol >> rc
    db >> rc

    # Visualization connections
    vs >> cg
    cg >> vd
    rd >> vd

    # Training Data generation
    sc >> ca
    ca >> qg
    db >> tg
    qg >> tg 