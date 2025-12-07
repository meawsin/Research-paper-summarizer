"""
Main application file for the Academic Research Summarizer.
Runs the Streamlit interface in Permanent Dark Mode.
"""
import streamlit as st
import logic

# 1. Page Configuration
st.set_page_config(
    page_title="Academic Research Summarizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. SESSION STATE MANAGEMENT
if 'history' not in st.session_state:
    st.session_state.history = []
if 'pdf_analysis' not in st.session_state:
    st.session_state.pdf_analysis = None  # Stores the PDF result so it doesn't vanish

# 3. PERMANENT DARK THEME VARIABLES
BG_COLOR = "#0e1117"
TEXT_COLOR = "#fafafa"
CARD_BG = "#1e1e1e"
BORDER_COLOR = "#333333"
BADGE_BG = "#333333"
BADGE_TEXT = "#e0e0e0"
INPUT_BG = "#262730"

# 4. INJECT PROFESSIONAL CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR};
    }}

    .stApp {{
        background-color: {BG_COLOR};
    }}

    /* INPUT FIELDS */
    .stTextInput input {{
        background-color: {INPUT_BG} !important;
        color: {TEXT_COLOR} !important;
        border: 1px solid {BORDER_COLOR};
    }}

    /* CONTAINERS (Cards) */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {{
        background-color: {CARD_BG};
        border-radius: 10px;
        border: 1px solid {BORDER_COLOR};
    }}

    /* BADGES */
    .badge {{
        background-color: {BADGE_BG};
        color: {BADGE_TEXT};
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85em;
        font-weight: 600;
        margin-right: 8px;
        border: 1px solid {BORDER_COLOR};
    }}

    /* BUTTONS */
    .stButton > button {{
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        height: 3em;
        border: 1px solid {BORDER_COLOR};
        background-color: {INPUT_BG};
        color: {TEXT_COLOR};
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        border-color: #3b82f6;
        color: #3b82f6;
    }}

    /* LINKS */
    a {{ color: #3b82f6 !important; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}

    /* HIGHLIGHT BOX */
    .insight-box {{
        background-color: {INPUT_BG};
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 4px;
        margin-top: 10px;
    }}
</style>
""", unsafe_allow_html=True)

# 5. SIDEBAR NAVIGATION
with st.sidebar:
    st.markdown("### **Research Paper Summarizer**")
    st.markdown("---")

    # Tool Selection
    nav_mode = st.radio("Select Tool:", ["🔎 Search arXiv", "📂 Upload PDF"])

    st.markdown("---")
    st.markdown("### 🕒 Recent History")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            st.caption(f"📄 {item}")
    else:
        st.caption("No papers yet.")

    st.markdown("---")
    st.markdown("**User Profile:**")
    st.caption("👤 **Abd Al Mohsin Siraj**")
    st.markdown(
        "<div style='font-size:0.8em; color:#888;'>© 2254901009 & 2254901085</div>",
        unsafe_allow_html=True
    )

# 6. HEADER
st.title("Academic Research Summarizer")

# ==========================================
# MODE 1: SEARCH ARXIV
# ==========================================
if nav_mode == "🔎 Search arXiv":
    st.markdown("#### Search 2M+ papers. AI reads the abstract for you.")

    # Search Bar
    col1, col2 = st.columns([4, 1])
    with col1:
        topic = st.text_input(
            "Research Topic",
            placeholder="e.g. Swarm Robotics...",
            label_visibility="collapsed"
        )
    with col2:
        search_btn = st.button("🔎 Search", use_container_width=True)

    # Perform Search
    if topic:
        if 'papers' not in st.session_state or search_btn:
            with st.spinner('Accessing arXiv Database...'):
                st.session_state.papers = logic.search_papers(topic)

        papers = st.session_state.papers

        if papers:
            st.markdown(f"Found **{len(papers)}** results for: *'{topic}'*")
            st.markdown("---")

            for i, paper in enumerate(papers):
                # USE NATIVE STREAMLIT CONTAINER FOR THE CARD
                with st.container(border=True):
                    # Header
                    st.subheader(f"{i+1}. {paper['title']}")

                    # Badges Row
                    st.markdown(f"""
                        <span class="badge">📅 {paper['published'].year}</span>
                        <span class="badge">✍️ {paper['authors']}</span>
                        <a href="{paper['url']}" target="_blank">🔗 Open PDF</a>
                    """, unsafe_allow_html=True)

                    st.markdown("")  # Spacer

                    # Abstract Preview
                    with st.expander("▼ Read Abstract Preview"):
                        st.write(paper['abstract'])

                    # Analyze Button
                    if st.button(f"⚡ Deep Dive Analysis", key=f"btn_{i}", use_container_width=True):
                        with st.spinner('Running AI Analysis...'):
                            summary = logic.generate_summary(paper['abstract'])
                            insights = logic.extract_insights(paper['abstract'])

                            # Update History
                            if paper['title'] not in st.session_state.history:
                                st.session_state.history.append(paper['title'])

                            # Display Tabs
                            t1, t2, t3 = st.tabs(["💡 Insights", "📝 Summary", "🔖 Citation"])

                            with t1:
                                st.markdown(f"""
                                <div class="insight-box">
                                    <b>🎯 Objective:</b><br>{insights['Objectives']}
                                    <hr style="margin:10px 0; opacity:0.2;">
                                    <b>📊 Conclusion:</b><br>{insights['Key Conclusion']}
                                </div>
                                """, unsafe_allow_html=True)

                            with t2:
                                st.write(summary)

                            with t3:
                                st.code(paper['citation'], language="markdown")

                            # Download
                            report_text = (
                                f"TITLE: {paper['title']}\n\n"
                                f"SUMMARY:\n{summary}\n\n"
                                f"OBJECTIVE:\n{insights['Objectives']}\n\n"
                                f"CONCLUSION:\n{insights['Key Conclusion']}"
                            )
                            st.download_button(
                                "📥 Save Report",
                                report_text,
                                file_name=f"paper_{i+1}.txt",
                                use_container_width=True
                            )
        else:
            st.info("No papers found. Try a broader topic.")

# ==========================================
# MODE 2: UPLOAD PDF
# ==========================================
elif nav_mode == "📂 Upload PDF":
    st.markdown("#### Analyze Full PDF Files")
    st.caption("Upload a paper and the AI will read the Introduction and Conclusion.")

    uploaded_file = st.file_uploader("Drop PDF Here", type="pdf")

    if uploaded_file:
        with st.container(border=True):
            col_a, col_b = st.columns([1, 4])
            with col_a:
                st.markdown("## 📄")
            with col_b:
                st.markdown(f"**{uploaded_file.name}**")
                st.caption(f"{round(uploaded_file.size/1024, 1)} KB • Local File")

            if st.button("⚡ Analyze PDF", use_container_width=True):
                with st.spinner("Extracting text and analyzing..."):
                    raw_text = logic.extract_text_from_pdf(uploaded_file)

                    # Store result in session state so it doesn't vanish
                    st.session_state.pdf_analysis = {
                        "filename": uploaded_file.name,
                        "summary": logic.generate_summary(raw_text),
                        "insights": logic.extract_insights(raw_text)
                    }

                    if uploaded_file.name not in st.session_state.history:
                        st.session_state.history.append(uploaded_file.name)

        # Show Results if available
        if (st.session_state.pdf_analysis and
                st.session_state.pdf_analysis['filename'] == uploaded_file.name):
            data = st.session_state.pdf_analysis
            st.markdown("### 📝 Analysis Results")

            t1, t2 = st.tabs(["💡 Key Insights", "📝 Full Summary"])

            with t1:
                st.markdown(f"""
                <div class="insight-box">
                    <b>🎯 Objective:</b><br>{data['insights']['Objectives']}
                    <hr style="margin:10px 0; opacity:0.2;">
                    <b>📊 Conclusion:</b><br>{data['insights']['Key Conclusion']}
                </div>
                """, unsafe_allow_html=True)

            with t2:
                st.write(data['summary'])

            report_text = (
                f"FILE: {data['filename']}\n\n"
                f"SUMMARY:\n{data['summary']}\n\n"
                f"INSIGHTS:\n{data['insights']}"
            )
            st.download_button(
                "📥 Download Analysis",
                report_text,
                file_name="analysis.txt",
                use_container_width=True
            )