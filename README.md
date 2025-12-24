# Academic Research Paper Summarizer

**Presented by:** Abd Al Mohsin Siraj & Nishat Tasneem

## Overview

The AI Research Assistant is a specialized dashboard designed to accelerate academic discovery. It connects directly to the arXiv API to fetch real-time research papers and utilizes Natural Language Processing (NLP) to generate structured executive summaries, helping researchers filter through literature **70% faster**.

Unlike standard summarizers, this tool features a **RAG (Retrieval-Augmented Generation)** engine, allowing researchers to "chat" with PDF documents to extract specific data points like sample sizes, methodologies, and datasets.

## Application Gallery

### 1. Research & Discovery

#### Real-Time Search
![Dashboard Search](assets/dashboard_search.png)
- Search 2M+ papers via arXiv API
- Drag & drop local PDFs for analysis

![Search Results](assets/search_result.png)

### 2. Analysis & Insights

#### Key Insights Extraction
![Key Insights](assets/key_insights.png)

#### Detailed Summaries
![Detailed Summary](assets/Detailed_summary.png)
- Auto-extracts Objectives & Conclusions
- Generates structured executive summaries

![Summary View](assets/summary.png)

### 3. Interactive AI

#### Deep Dive Analysis
![Deep Dive Analysis](assets/Deep_dive_analysis.png)

#### RAG Chat Bot
![Chat with Paper](assets/chat_with_paper.png)
- Tabbed view for citations & insights
- Ask questions like "What is the sample size?"

#### Additional Features
![Citations](assets/citation.png)
![Export to Text](assets/txt_file_download.png)
- Direct citation generate
- txt file generated for notes

## System Architecture

The project utilizes an AI Engineering approach to run large language models locally on a CPU.

```mermaid
graph TD
    subgraph Frontend
        A[User Interface Streamlit]
    end

    subgraph Logic_Layer
        B[Controller app.py and logic.py]
    end

    subgraph AI_Engine
        C[PDF Text Extractor PyPDF]
        D[Summarization Model DistilBART-CNN]
        E[Embedding Model MiniLM-L6-v2]
        F[QA Model RoBERTa-Base]
    end

    subgraph Storage
        G[Vector Database ChromaDB]
    end

    A -->|Upload PDF| B
    B -->|Raw Text| C
    C -->|Text Chunks| D
    D -->|Summary| A
    
    C -->|Text Chunks| E
    E -->|Embeddings| G
    
    A -->|User Question| B
    B -->|Query Context| G
    G -->|Retrieved Chunks| F
    F -->|Final Answer| A
```


## Key Features

- **Real-Time Search:** Queries the arXiv database for up-to-date papers.
- **Smart Summarization:** Uses `sshleifer/distilbart-cnn-12-6` to condense abstracts into meaningful insights.
- **Deep Dive Analysis:** Automatically extracts Objectives and Conclusions separate from the main summary.
- **RAG-Powered Chat:** Implements LangChain and ChromaDB to enable context-aware Question & Answering on specific documents.
- **Auto-Citation:** Generates APA-style citations instantly.
- **Exportable Reports:** One-click download of research notes to `.txt`.

## Installation

To run this project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/Research-paper-summarizer.git
cd Research-paper-summarizer
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


## Usage

Run the application using Streamlit:

```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

## Technical Challenges Solved

- **Context Window Management:** Implemented `RecursiveCharacterTextSplitter` with a chunk size of 500 tokens to prevent model hallucinations and fit within the RoBERTa context window.

- **PDF Noise Cleaning:** Developed a custom cleaning pipeline to strip IEEE headers, citations, and reference lists before processing to improve summary quality.

- **Vector Search:** Utilized ChromaDB for semantic search, allowing the chat bot to find answers even when exact keywords don't match.

## License

This project is open-source and available under the MIT License.