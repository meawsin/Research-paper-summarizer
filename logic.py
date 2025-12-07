"""
Logic module for AI Research Assistant (Pro Version).
Handles arXiv API, PDF Parsing, and Local Summarization.
"""
import arxiv
from pypdf import PdfReader
from transformers import pipeline

# Load the summarization model (Local Inference)
# framework="pt" forces PyTorch
SUMMARIZER = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

def search_papers(topic):
    """
    Fetches top 8 relevant papers (Upgraded from 3).
    """
    search = arxiv.Search(
        query=topic,
        max_results=8,  # INCREASED TO 8
        sort_by=arxiv.SortCriterion.Relevance
    )

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
    """
    Extracts text from a PDF file object.
    Engineered to grab the first 2 pages (Intro) and last 2 pages (Conclusion)
    to fit within Local AI token limits.
    """
    try:
        reader = PdfReader(pdf_file)
        text = ""
        total_pages = len(reader.pages)

        # Strategy: Read first 2 pages (Introduction) + Last Page (Conclusion)
        pages_to_read = [0, 1]
        if total_pages > 2:
            pages_to_read.append(total_pages - 1)

        for p in pages_to_read:
            if p < total_pages:
                # pylint: disable=no-member
                text += reader.pages[p].extract_text() + "\n"

        return text
    except Exception as e:  # pylint: disable=broad-exception-caught
        return f"Error reading PDF: {str(e)}"

def generate_summary(text):
    """
    Summarizes text. Handles long PDF text by truncating intelligently.
    """
    # BART model limit is ~1024 tokens. We truncate input to ~3000 chars safely.
    if len(text) > 3000:
        text = text[:3000]

    input_len = len(text.split())

    # Dynamic limits based on input size
    new_max = max(50, int(input_len * 0.4))
    new_min = max(30, int(input_len * 0.15))

    # Cap max length to prevent model errors
    new_max = min(new_max, 400)

    if input_len < 50:
        return text

    try:
        summary_list = SUMMARIZER(
            text,
            max_length=new_max,
            min_length=new_min,
            do_sample=False,
            truncation=True
        )
        return summary_list[0]['summary_text']
    except Exception:  # pylint: disable=broad-exception-caught
        return text # Fallback

def extract_insights(text):
    """
    Smart Extraction of Objectives and Conclusions.
    """
    sentences = text.split('. ')

    # Keywords to hunt for
    obj_keywords = ["aim", "objective", "purpose", "propose", "goal", "introduct"]
    conc_keywords = ["conclude", "result", "show", "find", "demonstrate"]

    objectives = [
        s for s in sentences if any(x in s.lower() for x in obj_keywords)
    ]
    conclusions = [
        s for s in sentences if any(x in s.lower() for x in conc_keywords)
    ]

    # Fallback Strategy
    final_obj = objectives[0] if objectives else sentences[0]
    final_conc = conclusions[0] if conclusions else sentences[-1]

    return {
        "Objectives": final_obj + ".",
        "Key Conclusion": final_conc + "."
    }