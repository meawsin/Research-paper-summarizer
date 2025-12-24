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

# 1. SUMMARIZATION MODEL
SUMMARIZER = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

# 2. Q&A MODEL
QA_PIPELINE = pipeline("question-answering", model="deepset/roberta-base-squad2")

# 3. EMBEDDING MODEL
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


class ResearchAssistantRAG:
    """Handles the 'Chat with PDF' functionality."""
    def __init__(self):
        self.vector_db = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def ingest_text(self, text):
        if not text: return False
        
        # CLEAN TEXT BEFORE INDEXING (Fixes the "PorContext" garbage)
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
        
        # Retrieve top 3 chunks for better context
        docs = self.vector_db.similarity_search(question, k=3)
        context = " ".join([d.page_content for d in docs])
        
        try:
            result = QA_PIPELINE(question=question, context=context)
            return result['answer'], context
        except Exception as e:
            return f"Error: {str(e)}", context

    def clear_db(self):
        if self.vector_db:
            try:
                self.vector_db.delete_collection()
            except Exception:
                pass
            self.vector_db = None

rag_engine = ResearchAssistantRAG()


# --- CLEANING FUNCTIONS ---

def clean_latex(text):
    if not text: return ""
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    text = re.sub(r'\{\\em (.*?)\}', r'\1', text)
    text = re.sub(r'\\textbf\{(.*?)\}', r'\1', text)
    text = re.sub(r'\\cite\{.*?\}', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_garbage(text):
    """
    Aggressively removes academic noise and FIXES SPACING issues.
    """
    if not text: return ""
    
    # 1. Fix common PDF parsing issue: "WordOneWordTwo" -> "WordOne WordTwo"
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    
    # 2. Remove IEEE/Journal Headers (Case Insensitive, Multiline)
    text = re.sub(r'(?i)^.*?IEEE TRANSACTIONS.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?i)^.*?Journal of.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?i)^.*?Vol\.\s?\d+.*?$', '', text, flags=re.MULTILINE)
    
    # 3. Remove "Index Terms" and Keywords blocks
    text = re.sub(r'(?i)Index Terms.*', '', text)
    text = re.sub(r'(?i)Keywords:.*', '', text)
    
    # 4. Clean "1 Abstract —" artifacts
    # Removes "1 Abstract —" so the sentence starts cleanly with "The authors..."
    text = re.sub(r'(?i)^\d*\s*Abstract\s*[—–-]\s*', '', text, flags=re.MULTILINE)
    
    # 5. Remove Copyright and URLs
    text = re.sub(r'(?i)©.*', '', text)
    text = re.sub(r'https?://\S+', '', text)
    
    # 6. Fix newlines breaking sentences
    text = text.replace("-\n", "").replace("\n", " ")
    
    return text

def clean_tone(text):
    """Converts 'We/Our' to 'The authors/Their'."""
    if not text: return ""
    text = re.sub(r'(?i)\bwe\b', 'the authors', text)
    text = re.sub(r'(?i)\bour\b', 'their', text)
    text = re.sub(r'(?i)\bmy\b', 'the author\'s', text)
    text = re.sub(r'(?i)\bi\b', 'the author', text)
    
    text = text.replace("the authors are", "the authors are") \
               .replace("the authors study", "the authors study") \
               .replace("the authors has", "the authors have") \
               .replace("the authors find", "the authors find")
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
            "abstract": clean_latex(result.summary.replace("\n", " ")),
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
        total_pages = len(reader.pages)
        for i, page in enumerate(reader.pages):
            content = page.extract_text()
            # Stop if we hit References to save time/tokens
            if i > total_pages * 0.7:
                header_check = content[:100].lower()
                if "references" in header_check or "bibliography" in header_check:
                    break
            text += content + "\n"
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
    # CRASH FIX: Return empty string if input is empty
    if not text or len(text.strip()) < 50:
        return text if text else ""
        
    if len(text) > 3500:
        text = text[:3500]
        
    estimated_tokens = len(text) // 4
    safe_min_len = min(min_len, int(estimated_tokens * 0.4))
    safe_min_len = max(safe_min_len, 20)
    
    safe_max_len = min(max_len, int(estimated_tokens * 0.9)) 
    safe_max_len = max(safe_max_len, safe_min_len + 10)

    try:
        summary_list = SUMMARIZER(
            text,
            max_length=safe_max_len,
            min_length=safe_min_len,
            do_sample=False,
            truncation=True
        )
        return summary_list[0]['summary_text']
    except Exception:
        # Fallback if AI fails (prevents index out of range)
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
            summary = clean_tone(summary)
            return format_as_student_bullets(summary)
        except Exception as e:
            return f"Error generating student summary: {str(e)}"

    chunks = chunk_text(text, chunk_size=4000)
    if len(chunks) == 1:
        summary = _run_model(chunks[0], max_len=600, min_len=200)
        return clean_tone(summary)

    chunk_summaries = []
    print(f"Processing {len(chunks)} chunks for detailed summary...")
    for i, chunk in enumerate(chunks):
        try:
            summary = _run_model(chunk, max_len=300, min_len=100)
            chunk_summaries.append(summary)
        except Exception:
            continue
    detailed_summary = "\n\n".join(chunk_summaries)
    return clean_tone(detailed_summary)

def extract_insights(text):
    text = clean_garbage(text)
    
    # 1. Try to find explicit objectives using keyword search
    sentences = text[:5000].split(". ")
    strong_starts = ["the objective", "this paper", "we propose", "the goal", "this study"]
    
    found_objs = []
    for s in sentences:
        s = clean_tone(s.strip())
        if any(start in s.lower() for start in strong_starts) and len(s) > 30 and len(s) < 300:
            found_objs.append(f"• {s}")
            if len(found_objs) >= 3: break
            
    # --- THIS IS THE PART THAT CHANGED ---
    # 2. IF FAILED: Generate a 3-4 line Introduction Summary instead of an error
    if not found_objs:
        try:
            # Summarize the first 2000 chars (Abstract + Intro) to get the main idea
            intro_summary = _run_model(text[:2000], max_len=150, min_len=60)
            intro_summary = clean_tone(intro_summary)
            # Return it as a single paragraph without bullets
            found_objs = [intro_summary] 
        except Exception:
            found_objs = ["The specific objectives could not be automatically isolated."]
    # -------------------------------------

    # 3. Extract Conclusion
    conclusion_text = text[-3000:]
    try:
        conc_summary = _run_model(conclusion_text, max_len=150, min_len=40)
        conc_summary = clean_tone(conc_summary)
    except:
        conc_summary = "Conclusion not available."

    return {
        "Objectives": "\n".join(found_objs),
        "Key Conclusion": conc_summary
    }
    text = clean_garbage(text)
    
    # 1. Try to find explicit objectives using keyword search
    sentences = text[:5000].split(". ")
    strong_starts = ["the objective", "this paper", "we propose", "the goal", "this study"]
    
    found_objs = []
    for s in sentences:
        s = clean_tone(s.strip())
        if any(start in s.lower() for start in strong_starts) and len(s) > 30 and len(s) < 300:
            found_objs.append(f"• {s}")
            if len(found_objs) >= 3: break
            
    # --- THIS IS THE PART THAT CHANGED ---
    # 2. IF FAILED: Generate a 3-4 line Introduction Summary instead of an error
    if not found_objs:
        try:
            # Summarize the first 2000 chars (Abstract + Intro) to get the main idea
            intro_summary = _run_model(text[:2000], max_len=150, min_len=60)
            intro_summary = clean_tone(intro_summary)
            # Return it as a single paragraph without bullets
            found_objs = [intro_summary] 
        except Exception:
            found_objs = ["The specific objectives could not be automatically isolated."]
    # -------------------------------------

    # 3. Extract Conclusion
    conclusion_text = text[-3000:]
    try:
        conc_summary = _run_model(conclusion_text, max_len=150, min_len=40)
        conc_summary = clean_tone(conc_summary)
    except:
        conc_summary = "Conclusion not available."

    return {
        "Objectives": "\n".join(found_objs),
        "Key Conclusion": conc_summary
    }
    """
    Smart extraction that filters out IEEE headers and garbage.
    """
    # 1. Clean garbage again just to be safe
    text = clean_garbage(text)

    # 2. LOCATE INTRODUCTION
    intro_match = re.search(r'(?i)\bI\.\s*Introduction\b|\b1\.?\s*Introduction\b|\bIntroduction\b', text)
    if intro_match:
        start_idx = intro_match.end()
        intro_text = text[start_idx : start_idx + 4000]
    else:
        intro_text = text[: int(len(text) * 0.3)]

    # 3. LOCATE CONCLUSION
    conc_match = re.search(r'(?i)\bConclusion\b|\bConcluding Remarks\b|\bSummary\b', text)
    if conc_match:
        start_idx = conc_match.end()
        conclusion_raw_text = text[start_idx:]
    else:
        conclusion_raw_text = text[-2000:]

    # --- OBJECTIVE EXTRACTION ---
    protected_intro = intro_text.replace("U.S.", "US").replace("U.K.", "UK").replace("Fig.", "Fig").replace("al.", "al")
    sentences = protected_intro.split(". ")
    
    strong_starts = ["this paper", "this study", "the goal", "the objective", "we propose", "we present", "the aim"]
    keywords = ["aim", "objective", "purpose", "propose", "goal", "present", "paper describes"]
    
    tier1, tier2 = [], []
    seen = set()

    for s in sentences:
        s_clean = s.strip()
        # FILTER: Skip junk sentences (IEEE headers, short lines, ALL CAPS titles)
        if len(s_clean) < 30: continue
        if "ieee" in s_clean.lower(): continue 
        if s_clean.isupper(): continue 
        
        # Check Tier 1
        if any(s_clean.lower().startswith(start) for start in strong_starts):
            s_final = clean_tone(s_clean)
            if s_final not in seen:
                tier1.append(s_final)
                seen.add(s_final)
        # Check Tier 2
        elif any(k in s_clean.lower() for k in keywords):
            s_final = clean_tone(s_clean)
            if s_final not in seen:
                tier2.append(s_final)
                seen.add(s_final)

    final_objectives = tier1[:3] if tier1 else tier2[:3]

    if final_objectives:
        formatted_objs = "\n".join([f"• {obj}." for obj in final_objectives])
    else:
        formatted_objs = "• Objectives not explicitly identified in the introduction text."

    # --- CONCLUSION SUMMARIZATION ---
    try:
        simplified_conclusion = _run_model(conclusion_raw_text, max_len=150, min_len=40)
        simplified_conclusion = clean_tone(simplified_conclusion)
    except Exception:
        simplified_conclusion = "Conclusion could not be automatically summarized."

    # Restore abbreviations
    formatted_objs = formatted_objs.replace(" US ", " U.S. ").replace(" UK ", " U.K. ")
    simplified_conclusion = simplified_conclusion.replace(" US ", " U.S. ").replace(" UK ", " U.K. ")

    return {"Objectives": formatted_objs, "Key Conclusion": simplified_conclusion}