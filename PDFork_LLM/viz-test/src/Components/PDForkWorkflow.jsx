import React, { useState } from 'react';
import { ArrowRightCircle, FileText, Database, Cpu, BrainCircuit, BarChart2, Network } from 'lucide-react';

const WorkflowStep = ({ icon: Icon, title, description, isActive, onClick }) => (
  <div 
    className={`relative p-4 border rounded-lg shadow-sm cursor-pointer transition-all duration-300 ${
      isActive ? 'bg-blue-50 border-blue-500' : 'bg-white border-gray-200 hover:border-blue-300'
    }`}
    onClick={onClick}
  >
    <div className="flex items-center mb-2">
      <Icon className={`w-6 h-6 ${isActive ? 'text-blue-600' : 'text-gray-600'}`} />
      <h3 className={`ml-2 font-semibold ${isActive ? 'text-blue-600' : 'text-gray-800'}`}>
        {title}
      </h3>
    </div>
    <p className={`text-sm ${isActive ? 'text-blue-700' : 'text-gray-600'}`}>
      {description}
    </p>
  </div>
);

const PDForkWorkflow = () => {
  const [activeStep, setActiveStep] = useState(0);
  
  const steps = [
    {
      icon: FileText,
      title: "PDF Input & Text Extraction",
      description: "Upload PDFs to preprocess/ directory. Multiple extraction methods including OCR ensure robust text capture.",
    },
    {
      icon: BrainCircuit,
      title: "Semantic Chunking",
      description: "Text is intelligently divided into semantic chunks based on document structure and content meaning.",
    },
    {
      icon: Database,
      title: "Vector Storage",
      description: "Chunks are embedded and stored in Chroma database for efficient semantic search and retrieval.",
    },
    {
      icon: Cpu,
      title: "LLM Processing",
      description: "OpenAI or Ollama models analyze content to generate summaries and extract key information.",
    },
    {
      icon: BarChart2,
      title: "Visual Element Analysis",
      description: "Charts, tables, and images are extracted and classified using computer vision techniques.",
    },
    {
      icon: Network,
      title: "Relationship Mapping",
      description: "Documents are interconnected based on content similarity, shared entities, and temporal proximity.",
    }
  ];

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white">
      <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">
        PDFork Processing Pipeline
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        {steps.map((step, index) => (
          <React.Fragment key={index}>
            <WorkflowStep
              {...step}
              isActive={activeStep === index}
              onClick={() => setActiveStep(index)}
            />
            {index % 2 === 0 && index !== steps.length - 1 && (
              <div className="hidden md:flex items-center justify-center">
                <ArrowRightCircle className="w-6 h-6 text-blue-400" />
              </div>
            )}
          </React.Fragment>
        ))}
      </div>

      <div className="mt-8 p-4 bg-gray-50 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">Output Organization</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="p-3 bg-white rounded shadow-sm">
            <p className="font-medium text-gray-700">📁 output/raw_text/</p>
            <p className="text-gray-600">Extracted text in JSON format</p>
          </div>
          <div className="p-3 bg-white rounded shadow-sm">
            <p className="font-medium text-gray-700">📁 output/summaries/</p>
            <p className="text-gray-600">Generated document summaries</p>
          </div>
          <div className="p-3 bg-white rounded shadow-sm">
            <p className="font-medium text-gray-700">📁 output/visual_elements/</p>
            <p className="text-gray-600">Extracted visuals with metadata</p>
          </div>
          <div className="p-3 bg-white rounded shadow-sm">
            <p className="font-medium text-gray-700">📁 data/</p>
            <p className="text-gray-600">Chroma vector stores</p>
          </div>
          <div className="p-3 bg-white rounded shadow-sm">
            <p className="font-medium text-gray-700">📁 training_data/</p>
            <p className="text-gray-600">Generated Q&A datasets</p>
          </div>
          <div className="p-3 bg-white rounded shadow-sm">
            <p className="font-medium text-gray-700">📁 finished/</p>
            <p className="text-gray-600">Processed PDF storage</p>
          </div>
        </div>
      </div>
      
      <div className="mt-8 text-sm text-gray-500 text-center">
        Click on any step to learn more about its functionality
      </div>
    </div>
  );
};

export default PDForkWorkflow;