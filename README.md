# 📄 Academic Paper Simplifier

Convert complex academic papers into plain-English summaries using AI-powered text simplification.

## 🎯 Features

- **PDF Text Extraction**: Automatically extracts and parses text from academic papers
- **Section-by-Section Analysis**: Breaks down papers into Introduction, Methods, Results, etc.
- **AI-Powered Simplification**: Uses state-of-the-art language models (BART, Pegasus, Flan-T5)
- **Readability Metrics**: Calculates Flesch Reading Ease, Grade Level, and Gunning Fog scores
- **Interactive Web UI**: Simple Streamlit interface for easy interaction
- **Downloadable Summaries**: Export simplified versions as text files

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd paper-simplifier
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

*Note: First-time installation may take 5-10 minutes as it downloads AI models (~1-2GB).*

### Running the App

Start the Streamlit web interface:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

1. **Upload a PDF**: Click "Browse files" and select an academic paper (PDF format)
2. **Choose a Model** (optional): Select from BART, Pegasus, or Flan-T5 in the sidebar
3. **Click "Simplify Paper"**: Wait 2-5 minutes for processing
4. **Review Results**: See key takeaways, section-by-section breakdown, and readability scores
5. **Download**: Export the simplified summary as a text file

## 🏗️ Architecture

```
User Input (PDF)
      ↓
[extractor.py] → Extract text, parse sections
      ↓
[simplifier.py] → AI model simplifies each section
      ↓
[app.py] → Display results + readability metrics
```

## 📁 Project Structure

```
paper-simplifier/
├── app.py              # Streamlit web interface
├── extractor.py        # PDF text extraction & parsing
├── simplifier.py       # AI simplification engine
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🧠 Available Models

| Model | Best For | Style |
|-------|----------|-------|
| **BART** (default) | General papers | Balanced, comprehensive |
| **Pegasus** | Quick summaries | Concise, extractive |
| **Flan-T5** | Flexible needs | Instruction-tuned, better performance |

## 📊 Readability Metrics Explained

### Flesch Reading Ease
- **90-100**: Very easy (5th grade)
- **60-70**: Standard (8th-9th grade)
- **30-50**: Difficult (college level)
- **0-30**: Very difficult (graduate level)

### Grade Level
U.S. school grade level required to understand the text.

### Gunning Fog Index
Years of formal education needed to comprehend on first reading.

## 🛠️ Technical Details

### Dependencies
- `transformers`: HuggingFace models for text simplification
- `torch`: Deep learning framework
- `pymupdf`: PDF text extraction
- `textstat`: Readability calculations
- `streamlit`: Web interface
- `requests`: HTTP requests (for future arXiv integration)

### Performance
- **Processing Time**: 2-5 minutes per paper
- **Memory Usage**: ~2-4GB RAM (model loading)
- **Token Limit**: ~4000 characters per section (model constraint)

## 🔮 Future Enhancements (MVP 2.0)

- [ ] arXiv API integration (fetch papers by URL)
- [ ] Multiple simplification levels (Expert/Student/Layman)
- [ ] Semantic search for similar papers
- [ ] Paper recommendation engine
- [ ] Fine-tuned domain-specific models
- [ ] Batch processing
- [ ] Summary comparison view

## 🐛 Troubleshooting

**Issue**: Model download fails
- **Solution**: Check internet connection, retry installation

**Issue**: "Out of memory" error
- **Solution**: Use smaller model (Pegasus) or reduce input text length

**Issue**: PDF extraction returns garbled text
- **Solution**: Ensure PDF has selectable text (not scanned image)

**Issue**: App crashes during processing
- **Solution**: Check PDF file is not corrupted, try a different paper

## 📝 Example Use Cases

1. **Students**: Quickly understand complex research papers for literature reviews
2. **Researchers**: Get quick overviews of papers in unfamiliar domains
3. **Science Communicators**: Create accessible summaries for general audiences
4. **Educators**: Generate study materials from academic sources

## 🤝 Contributing

This is an MVP project. Suggestions and improvements welcome!

## 📄 License

Open source - use freely for academic and educational purposes.

## 🙏 Acknowledgments

Built with:
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Streamlit](https://streamlit.io/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [textstat](https://github.com/textstat/textstat)

---

**Version**: 1.0.0 (MVP)  
**Build Time**: ~1 week  
**Status**: Production-ready MVP
