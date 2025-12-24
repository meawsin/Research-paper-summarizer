"""
Logic module for AI Research Assistant (Pro Version).
Handles arXiv API, PDF Parsing, Local Summarization, and RAG (Q&A).
"""
import re
import arxiv
from pypdf import PdfReader
from transformers import pipeline

# --- ROBUST IMPORTS FOR RAG ---
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. SUMMARIZATION MODEL (Uses a smaller, faster model for stability)
SUMMARIZER = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")

# 2. Q&A MODEL (Extractive)
QA_PIPELINE = pipeline("question-answering", model="deepset/roberta-base-squad2")

# 3. EMBEDDING MODEL
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


class ResearchAssistantRAG:
    """Handles the 'Chat with PDF' functionality."""
    def __init__(self):
        self.vector_db = None
        # Reduced chunk size to fit within RoBERTa's context window
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

    def ingest_text(self, text):
        if not text: return False
        
        clean_text = clean_garbage(text)
        texts = self.text_splitter.split_text(clean_text)
        
        self.vector_db = Chroma.from_texts(
            texts=texts,
            embedding=EMBEDDING_MODEL,
            collection_name="current_paper_analysis"
        )
        return True

    def ask_question(self, question):
        if not self.vector_db: return "Please process a PDF first.", ""
        
        # Guardrail for conversational inputs
        if question.lower().strip() in ['hi', 'hello', 'hey', 'thanks']:
            return "Hello! I am ready to answer questions about this research paper.", ""

        # Retrieve top chunks
        docs = self.vector_db.similarity_search(question, k=3)
        context = " ".join([d.page_content for d in docs])
        
        try:
            result = QA_PIPELINE(question=question, context=context)
            answer = result['answer']
            
            # IMPROVEMENT: If answer is very short, try to expand it to the full sentence
            # This makes the chat feel more "intelligent" and less robotic.
            if len(answer) < 20:
                sentences = context.split('.')
                for s in sentences:
                    if answer in s:
                        answer = s.strip() + "."
                        break
            
            return answer, context
        except Exception as e:
            return f"I couldn't find a specific answer in the text. (Error: {str(e)})", context

    def clear_db(self):
        if self.vector_db:
            try:
                self.vector_db.delete_collection()
            except Exception:
                pass
            self.vector_db = None

rag_engine = ResearchAssistantRAG()


# --- CLEANING FUNCTIONS ---

def clean_garbage(text):
    """Aggressively removes academic noise."""
    if not text: return ""
    text = re.sub(r'(?i)^.*?IEEE TRANSACTIONS.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?i)Index Terms.*', '', text)
    text = re.sub(r'(?i)Keywords:.*', '', text)
    text = re.sub(r'(?i)©.*', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = text.replace("-\n", "").replace("\n", " ")
    return text

def remove_references(text):
    """
    CRITICAL FIX: Cuts off the text before the References section.
    This prevents the 'Conclusion' analysis from reading the bibliography.
    """
    patterns = [
        r'\n\s*References\s*\n',
        r'\n\s*Bibliography\s*\n',
        r'\n\s*REFERENCES\s*\n',
        r'\n\s*[0-9]+\.?\s*References\s*\n'
    ]
    
    min_len = len(text) * 0.6 # Ensure we don't cut early if a paper mentions "references" in the intro

    for p in patterns:
        matches = list(re.finditer(p, text))
        if matches:
            # Take the LAST match that is after the 60% mark
            last_match = matches[-1]
            if last_match.start() > min_len:
                return text[:last_match.start()]
                
    return text

def clean_tone(text):
    """Converts 'We/Our' to 'The authors/Their'."""
    if not text: return ""
    text = re.sub(r'(?i)\bwe\b', 'the authors', text)
    text = re.sub(r'(?i)\bour\b', 'their', text)
    sentences = text.split('. ')
    text = ". ".join([s[0].upper() + s[1:] if len(s) > 1 else s for s in sentences])
    return text

def search_papers(topic):
    search = arxiv.Search(query=topic, max_results=8, sort_by=arxiv.SortCriterion.Relevance)
    results = []
    for result in search.results():
        authors = [a.name for a in result.authors]
        author_str = f"{authors[0]} et al." if len(authors) > 1 else authors[0]
        year = result.published.year
        citation = f"{author_str} ({year}). {result.title}. arXiv preprint."
        results.append({
            "title": result.title,
            "abstract": result.summary.replace("\n", " "),
            "url": result.pdf_url,
            "published": result.published,
            "authors": author_str,
            "citation": citation,
            "source": "arxiv"
        })
    return results

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            text += content + "\n"
        
        # APPLY FIX: Remove references strictly
        text = remove_references(text)
        return clean_garbage(text)
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def chunk_text(text, chunk_size=3500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def _run_model(text, max_len=400, min_len=50):
    if not text or len(text.strip()) < 50:
        return ""
    
    # Truncate to avoid model crash
    text = text[:3000] 

    try:
        summary_list = SUMMARIZER(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True
        )
        return summary_list[0]['summary_text']
    except Exception:
        return text[:300] + "..."

def format_as_student_bullets(text):
    sentences = text.replace("._", ".").split(". ")
    formatted = []
    for s in sentences:
        s = s.strip()
        if len(s) > 20:
            if not s.endswith("."): s += "."
            s = s[0].upper() + s[1:]
            formatted.append(f"• {s}")
    return "\n\n".join(formatted)

def generate_summary(text, summary_type="detailed"):
    rag_engine.clear_db()
    rag_engine.ingest_text(text)

    if summary_type == "abstract":
        try:
            summary = _run_model(text, max_len=200, min_len=60)
            return format_as_student_bullets(clean_tone(summary))
        except Exception as e:
            return f"Error: {str(e)}"

    chunks = chunk_text(text, chunk_size=4000)
    if len(chunks) == 1:
        return clean_tone(_run_model(chunks[0], max_len=600, min_len=200))

    chunk_summaries = []
    # Limit to first 3 chunks to save time/memory
    for chunk in chunks[:3]:
        try:
            summary = _run_model(chunk, max_len=300, min_len=100)
            chunk_summaries.append(summary)
        except Exception:
            continue
    return clean_tone("\n\n".join(chunk_summaries))

def extract_insights(text):
    text = clean_garbage(text)
    
    # 1. Improved Objective Search
    sentences = text[:5000].split(". ")
    strong_starts = ["the objective", "this paper", "we propose", "the goal", "this study", "we aim", "the purpose"]
    
    found_objs = []
    for s in sentences:
        s = clean_tone(s.strip())
        if any(start in s.lower() for start in strong_starts) and 30 < len(s) < 400:
            found_objs.append(f"• {s}")
            if len(found_objs) >= 2: break
    
    if not found_objs:
        # Fallback: Summarize the introduction
        try:
            intro_summary = _run_model(text[:2000], max_len=120, min_len=40)
            found_objs = [clean_tone(intro_summary)]
        except:
            found_objs = ["Objectives could not be automatically extracted."]

    # 2. Extract Conclusion (Now safer due to remove_references)
    # Search for a header first
    conc_match = re.search(r'(?i)\n\s*(Conclusion|Concluding Remarks|Summary)\s*\n', text)
    if conc_match:
        conclusion_text = text[conc_match.end():][:3000] # Take text AFTER the header
    else:
        conclusion_text = text[-2500:] # Fallback to end of text

    try:
        conc_summary = _run_model(conclusion_text, max_len=150, min_len=40)
        conc_summary = clean_tone(conc_summary)
    except:
        conc_summary = "Conclusion not available."

    return {
        "Objectives": "\n".join(found_objs),
        "Key Conclusion": conc_summary
    }