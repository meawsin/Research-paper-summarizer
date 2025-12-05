# Academic Research Paper Summarizer

### Automated Literature Review Dashboard
**Presented by:** Abd Al Mohsin Siraj & Nishat Tasneem

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![AI](https://img.shields.io/badge/AI-Hugging_Face_Transformers-yellow)

## 📌 Overview
The **AI Research Assistant** is a specialized dashboard designed to accelerate academic discovery. It connects directly to the **arXiv API** to fetch real-time research papers and utilizes **Natural Language Processing (NLP)** to generate structured executive summaries, helping researchers filter through literature 70% faster.

##  Key Features
* **Real-Time Search:** Queries the arXiv database for up-to-date papers.
* **Smart Summarization:** Uses `facebook/bart-large-cnn` to condense abstracts into meaningful insights.
* **Deep Dive Analysis:** Automatically extracts **Objectives** and **Conclusions** separate from the main summary.
* **Auto-Citation:** Generates APA-style citations instantly.
* **Exportable Reports:** One-click download of research notes to `.txt`.
* **Adaptive UI:** Professional Light/Dark mode toggle.

## 🛠️ Architecture
This project utilizes an **AI Engineering** approach to run large language models locally on a CPU:
1.  **Data Layer:** `arxiv` API wrapper.
2.  **Inference Engine:** PyTorch + Hugging Face Transformers.
3.  **Frontend:** Streamlit with Custom CSS injection.

## 📦 Installation
To run this project locally:

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/YourUsername/ai-research-assistant.git](https://github.com/YourUsername/ai-research-assistant.git)
   cd ai-research-assistant

2. **Create a Virtual Environment (Optional but recommended):**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

## ⚡ Usage
Run the application using Streamlit:

   ```bash
   pip install -r requirements.txt
```

The application will launch in your default web browser at http://localhost:8501.

## 📄 License
This project is open-source and available under the MIT License.