import streamlit as st
import logic

# 1. Page Configuration
st.set_page_config(
    page_title="Research Paper Summarizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. PROFESSIONAL CSS STYLING
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: black;
    }
    
    /* CARD STYLE: White box with shadow for each paper */
    .paper-card {
        background-color: black;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* HEADERS */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #0f172a;
    }
    
    /* METADATA BADGES (Year, Author) */
    .badge {
        background-color: #e2e8f0;
        color: #475569;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.85em;
        font-weight: 600;
        margin-right: 10px;
    }
    
    /* HIGHLIGHT BOX for Insights */
    .insight-box {
        background-color: black;
        border-left: 5px solid #3b82f6;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }
    
    /* Button Styling to look like an App */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        transition: all 0.2s;
    }
</style>
""", unsafe_allow_html=True)

# 3. SIDEBAR BRANDING
with st.sidebar:
    st.markdown("### **Research Paper Summarizer**")
    st.caption("AI-Powered Paper searching and summarization tool.")
    st.markdown("---")
    
    st.markdown("**User Profile:**")
    st.markdown("👤 **Abd Al Mohsin Siraj**")
    
    st.markdown("---")
    with st.expander("ℹ️ How it works"):
        st.write("1. Enter a topic.")
        st.write("2. Browse abstracts.")
        st.write("3. Click 'Deep Dive' for AI analysis.")

    st.markdown("<div style='font-size:0.85em; color:#6b7280;'>© 2254901009 & 2254901085</div>", unsafe_allow_html=True)
        

# 4. MAIN INTERFACE
st.title("Academic Research Paper Summarizer")
st.markdown("Search arXiv's database, analyze papers, and export citations instantly.")

# Search Bar Area
col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_input("Search Term", placeholder="e.g. Swarm Robotics, Dark Matter...", label_visibility="collapsed")
with col2:
    search_btn = st.button("🔎 Search Papers")

# 5. RESULTS LOGIC
if topic:
    # Logic to fetch papers
    if 'papers' not in st.session_state or search_btn:
        with st.spinner('Querying arXiv Database...'):
            st.session_state.papers = logic.search_papers(topic)

    papers = st.session_state.papers
    
    if papers:
        st.markdown(f"### Results for: *'{topic}'*")
        st.markdown("---")

        # LOOP THROUGH PAPERS
        for i, paper in enumerate(papers):
            # --- START OF PAPER CARD ---
            with st.container():
                st.markdown(f"""
                <div class="paper-card">
                    <h3 style="margin-bottom:0px;">{i+1}. {paper['title']}</h3>
                    <div style="margin-top: 10px; margin-bottom: 15px;">
                        <span class="badge">📅 {paper['published'].year}</span>
                        <span class="badge">✍️ {paper['authors']}</span>
                        <a href="{paper['url']}" target="_blank" style="text-decoration: none; color: #3b82f6; font-weight: bold; margin-left: 10px;">🔗 Open PDF</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Abstract Preview (Visible by default in expander)
                with st.expander("▼ Read Abstract Preview", expanded=False):
                    st.write(paper['abstract'])

                # Analysis Button
                if st.button(f"⚡ Deep Dive Analysis", key=f"btn_{i}"):
                    with st.spinner('Reading paper...'):
                        summary = logic.generate_summary(paper['abstract'])
                        insights = logic.extract_insights(paper['abstract'])
                        
                        # --- RESULTS TABS ---
                        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "📝 AI Summary", "🔖 Citation"])
                        
                        with tab1:
                            st.markdown(f"""
                            <div class="insight-box">
                                <b>🎯 Objective:</b><br>{insights['Objectives']}
                                <br><br>
                                <b>📊 Conclusion:</b><br>{insights['Key Conclusion']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with tab2:
                            st.write(summary)
                            
                        with tab3:
                            st.code(paper['citation'], language="markdown")
                            
                        # Download Button Logic
                        report_text = f"Title: {paper['title']}\nSummary: {summary}\nObjective: {insights['Objectives']}"
                        st.download_button("📥 Save Report", data=report_text, file_name=f"paper_{i+1}.txt")
                
                st.markdown("<br>", unsafe_allow_html=True) # Spacer
            # --- END OF PAPER CARD ---

    else:
        st.info("No papers found. Try a broader topic.")