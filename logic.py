"""
Logic module for AI Research Summarizer.
Handles arXiv API searching, Hugging Face summarization, and Citations.
"""
import arxiv
from transformers import pipeline

# Load the summarization model
# framework="pt" forces PyTorch to avoid CPU conversion crashes
SUMMARIZER = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

def search_papers(topic):
    """
    Fetches top 3 relevant papers from arXiv based on the topic.
    """
    search = arxiv.Search(
        query=topic,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = []
    for result in search.results():
        # Create a basic APA-style citation
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
            "citation": citation
        })
    return results

def generate_summary(text):
    """
    Summarizes the text with dynamic length limits to prevent warnings.
    """
    input_len = len(text.split())

    # Dynamic limits: Summary is roughly 1/2 the size of the original
    new_max = max(40, int(input_len * 0.5))
    new_min = max(20, int(input_len * 0.2))

    if input_len < 50:
        return text

    try:
        summary_list = SUMMARIZER(
            text,
            max_length=new_max,
            min_length=new_min,
            do_sample=False
        )
        return summary_list[0]['summary_text']
    except Exception:  # pylint: disable=broad-exception-caught
        # Fallback to original text if AI fails
        return text

def extract_insights(text):
    """
    Smart Extraction: Tries to find keywords, but falls back to
    First/Last sentences so the UI is never empty.
    """
    sentences = text.split('. ')

    # 1. Try to find specific keywords
    objectives = [
        s for s in sentences
        if any(x in s.lower() for x in ["aim", "objective", "purpose", "propose", "goal"])
    ]
    conclusions = [
        s for s in sentences
        if any(x in s.lower() for x in ["conclude", "result", "show", "find"])
    ]

    # 2. Fallback Strategy
    final_obj = objectives[0] if objectives else sentences[0]
    final_conc = conclusions[0] if conclusions else sentences[-1]

    return {
        "Objectives": final_obj + ".",
        "Key Conclusion": final_conc + "."
    }