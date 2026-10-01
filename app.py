import streamlit as st
import tempfile
import os
from extractor import extract_and_parse_paper
from simplifier import simplify_paper


# Page configuration
st.set_page_config(
    page_title="Academic Paper Simplifier",
    page_icon="📄",
    layout="wide"
)

# Title and description
st.title("📄 Academic Paper Simplifier")
st.markdown("""
Convert complex academic papers into plain-English summaries.
Upload a PDF and get an easy-to-understand breakdown with readability metrics.
""")

# Sidebar for mode and model selection
st.sidebar.header("Settings")

# Mode selection
mode = st.sidebar.selectbox(
    "Choose Mode",
    [
        "Core Simplifier",
        "ELI5 (Explain Like I'm 5)",
        "Comparative Insight",
        "Bullet Points"
    ],
    index=0,
    help="Select how you want to analyze papers"
)

# Map display names to internal mode names
mode_map = {
    "Core Simplifier": "core",
    "ELI5 (Explain Like I'm 5)": "eli5",
    "Comparative Insight": "comparative",
    "Bullet Points": "bullets"
}
selected_mode = mode_map[mode]

model_choice = st.sidebar.selectbox(
    "Choose AI Model",
    [
        "facebook/bart-large-cnn",
        "google/pegasus-xsum",
        "google/flan-t5-base"
    ],
    index=0
)

st.sidebar.markdown(f"""
### Mode: {mode}

""")

if selected_mode == "core":
    st.sidebar.markdown("""
**Core Simplifier:**
- Summarize a single paper
- Plain-English explanation
- Key contributions & results
- Readability metrics
    """)
elif selected_mode == "eli5":
    st.sidebar.markdown("""
**ELI5 Mode:**
- 3 levels of explanation:
  - 👨‍🔬 **Expert**: Technical summary
  - 🎓 **Student**: Moderate detail
  - 👶 **Layman**: Simple terms
- Compare readability across levels
    """)
elif selected_mode == "comparative":
    st.sidebar.markdown("""
**Comparative Insight:**
- Upload 2-5 papers
- Find common themes
- Identify conflicts
- Compare approaches
    """)
else:
    st.sidebar.markdown("""
**Bullet Points:**
- Create one-line bullets for all key details
- Section-wise bullet lists
- Easy to scan and share
    """)

st.sidebar.markdown("""
---
**Models:**
- **BART**: Balanced, detailed
- **Pegasus**: Concise
- **Flan-T5**: Flexible, instruction-tuned
""")

# Main content area - varies by mode
if selected_mode == "comparative":
    # Multiple file upload for comparative mode
    uploaded_files = st.file_uploader(
        "Upload 2-5 academic papers (PDF)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Select multiple PDF files to compare"
    )
    
    if uploaded_files and len(uploaded_files) >= 2:
        st.success(f"✅ {len(uploaded_files)} files uploaded")
        for f in uploaded_files:
            st.write(f"- {f.name}")
    elif uploaded_files:
        st.warning("⚠️ Please upload at least 2 papers for comparison")
else:
    # Single file upload for core and eli5 modes
    uploaded_file = st.file_uploader(
        "Upload your academic paper (PDF)",
        type=['pdf'],
        help="Select a PDF file of an academic paper to simplify"
    )
    uploaded_files = None

if (selected_mode != "comparative" and uploaded_file is not None) or (selected_mode == "comparative" and uploaded_files and len(uploaded_files) >= 2):
    # Display file info
    if selected_mode != "comparative":
        st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    # Simplify button
    button_text = "🚀 Simplify Paper" if selected_mode != "comparative" else "🔍 Compare Papers"
    if st.button(button_text, type="primary"):
        spinner_text = "Processing..." if selected_mode != "comparative" else "Comparing papers... This may take several minutes."
        with st.spinner(spinner_text):
            try:
                if selected_mode == "comparative":
                    # Process multiple papers
                    papers_data = []
                    temp_paths = []
                    
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            temp_paths.append(tmp_file.name)
                        
                        with st.status(f"Extracting text from {uploaded_file.name}..."):
                            sections = extract_and_parse_paper(temp_paths[-1])
                            papers_data.append((uploaded_file.name, sections))
                            st.write(f"✓ Found {len(sections)} sections")
                    
                    # Compare papers
                    with st.status("Generating comparative analysis..."):
                        result = simplify_paper(
                            model_name=model_choice,
                            mode=selected_mode,
                            papers_sections=papers_data
                        )
                        st.write("✓ Analysis complete")
                    
                    # Clean up temp files
                    for tmp_path in temp_paths:
                        os.unlink(tmp_path)
                        
                else:
                    # Process single paper (core or eli5)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    with st.status("Extracting text from PDF..."):
                        sections = extract_and_parse_paper(tmp_path)
                        st.write(f"✓ Found {len(sections)} sections")
                    
                    with st.status("Simplifying content with AI..."):
                        result = simplify_paper(
                            sections=sections,
                            model_name=model_choice,
                            mode=selected_mode
                        )
                        st.write("✓ Simplification complete")
                    
                    os.unlink(tmp_path)
                
                # Display results based on mode
                st.markdown("---")
                
                if selected_mode == "core":
                    # Core mode output
                    st.header("📝 Simplified Summary")
                    st.markdown(result['key_takeaways'])
                    st.markdown("---")
                    st.markdown("### Section-by-Section Breakdown")
                    st.markdown(result['formatted_output'])
                    
                    st.markdown("---")
                    st.header("📊 Readability Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        flesch = result['readability']['flesch_reading_ease']
                        st.metric("Flesch Reading Ease", f"{flesch:.1f}")
                        if flesch >= 90:
                            st.success("Very Easy")
                        elif flesch >= 60:
                            st.info("Standard")
                        elif flesch >= 30:
                            st.warning("Difficult")
                        else:
                            st.error("Very Difficult")
                    with col2:
                        st.metric("Grade Level", f"{result['readability']['flesch_kincaid_grade']:.1f}")
                    with col3:
                        st.metric("Gunning Fog", f"{result['readability']['gunning_fog']:.1f}")
                    
                    st.markdown("---")
                    st.download_button(
                        label="📥 Download Summary",
                        data=f"{result['key_takeaways']}\n\n{result['formatted_output']}",
                        file_name=f"simplified_{uploaded_file.name.replace('.pdf', '.txt')}",
                        mime="text/plain"
                    )
                
                elif selected_mode == "eli5":
                    # ELI5 mode output with tabs
                    st.header("📚 Multi-Level Explanations")
                    
                    tab1, tab2, tab3 = st.tabs(["👨‍🔬 Expert", "🎓 Student", "👶 Layman"])
                    
                    with tab1:
                        st.subheader("Expert Level (Technical)")
                        for section, content in result['levels']['expert'].items():
                            st.markdown(f"**{section}**")
                            st.write(content)
                            st.markdown("")
                        
                        st.markdown("---")
                        st.markdown("**Readability:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Flesch", f"{result['readability_scores']['expert']['flesch_reading_ease']:.1f}")
                        with col2:
                            st.metric("Grade", f"{result['readability_scores']['expert']['flesch_kincaid_grade']:.1f}")
                        with col3:
                            st.metric("Fog", f"{result['readability_scores']['expert']['gunning_fog']:.1f}")
                    
                    with tab2:
                        st.subheader("Student Level (Moderate)")
                        for section, content in result['levels']['student'].items():
                            st.markdown(f"**{section}**")
                            st.write(content)
                            st.markdown("")
                        
                        st.markdown("---")
                        st.markdown("**Readability:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Flesch", f"{result['readability_scores']['student']['flesch_reading_ease']:.1f}")
                        with col2:
                            st.metric("Grade", f"{result['readability_scores']['student']['flesch_kincaid_grade']:.1f}")
                        with col3:
                            st.metric("Fog", f"{result['readability_scores']['student']['gunning_fog']:.1f}")
                    
                    with tab3:
                        st.subheader("Layman Level (Simple)")
                        for section, content in result['levels']['layman'].items():
                            st.markdown(f"**{section}**")
                            st.write(content)
                            st.markdown("")
                        
                        st.markdown("---")
                        st.markdown("**Readability:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Flesch", f"{result['readability_scores']['layman']['flesch_reading_ease']:.1f}")
                        with col2:
                            st.metric("Grade", f"{result['readability_scores']['layman']['flesch_kincaid_grade']:.1f}")
                        with col3:
                            st.metric("Fog", f"{result['readability_scores']['layman']['gunning_fog']:.1f}")
                
                elif selected_mode == "comparative":
                    st.header("🔍 Comparative Analysis")
                    
                    # Individual summaries
                    st.subheader("📄 Individual Paper Summaries")
                    for paper_name, summary in result['comparison']['individual_summaries'].items():
                        with st.expander(f"📌 {paper_name}"):
                            st.write(summary)
                    
                    st.markdown("---")
                    
                    # Common themes
                    st.subheader("🤝 Common Themes & Overlaps")
                    st.info(result['comparison']['common_themes'])
                    
                    st.markdown("---")
                    
                    # Differences
                    st.subheader("⚔️ Key Differences & Conflicts")
                    st.warning(result['comparison']['differences'])
                
                else:  # bullets mode
                    st.header("• Bullet Points Summary")
                    
                    for section, lines in result['bullets_by_section'].items():
                        if not lines:
                            continue
                        with st.expander(f"{section}", expanded=True):
                            for line in lines:
                                st.write(line)
                    
                    st.markdown("---")
                    st.header("📊 Readability (All Bullets)")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Flesch", f"{result['readability']['flesch_reading_ease']:.1f}")
                    with col2:
                        st.metric("Grade", f"{result['readability']['flesch_kincaid_grade']:.1f}")
                    with col3:
                        st.metric("Fog", f"{result['readability']['gunning_fog']:.1f}")
                    
                    # Download bullets
                    all_bullets = []
                    for section, lines in result['bullets_by_section'].items():
                        all_bullets.append(f"[{section}]")
                        all_bullets.extend(lines)
                        all_bullets.append("")
                    st.markdown("---")
                    st.download_button(
                        label="📥 Download Bullet Points",
                        data="\n".join(all_bullets),
                        file_name=f"bullets_{uploaded_file.name.replace('.pdf', '.txt')}",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"❌ Error processing paper: {str(e)}")
                st.info("Try uploading a different PDF or check that the file is not corrupted.")

else:
    # Instructions when no file is uploaded
    st.info("👆 Upload a PDF to get started")
    
    st.markdown("""
    ### How to Use
    1. **Upload** a PDF of an academic paper
    2. **Click** the "Simplify Paper" button
    3. **Review** the plain-English summary and readability scores
    4. **Download** the simplified version
    
    ### Tips
    - Works best with standard academic papers (research articles, conference papers)
    - Processing may take 2-5 minutes depending on paper length
    - Choose different models in the sidebar for varied summary styles
    """)

# Footer
st.markdown("---")
st.caption("Built with Streamlit, HuggingFace Transformers, and PyMuPDF")
