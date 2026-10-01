import fitz  # PyMuPDF
import re
from typing import Dict, List


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file or file-like object
        
    Returns:
        Extracted text as a string
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def split_by_sections(text: str) -> Dict[str, str]:
    """
    Split paper text into sections based on common headers.
    
    Args:
        text: Full text of the paper
        
    Returns:
        Dictionary mapping section names to their content
    """
    # Common section headers in academic papers
    section_patterns = [
        r'\n\s*(?:Abstract|ABSTRACT)\s*\n',
        r'\n\s*(?:Introduction|INTRODUCTION)\s*\n',
        r'\n\s*(?:Related Work|RELATED WORK|Literature Review|LITERATURE REVIEW)\s*\n',
        r'\n\s*(?:Method|Methods|METHOD|METHODS|Methodology|METHODOLOGY)\s*\n',
        r'\n\s*(?:Results|RESULTS)\s*\n',
        r'\n\s*(?:Discussion|DISCUSSION)\s*\n',
        r'\n\s*(?:Conclusion|CONCLUSION|Conclusions|CONCLUSIONS)\s*\n',
        r'\n\s*(?:References|REFERENCES)\s*\n',
    ]
    
    sections = {}
    combined_pattern = '|'.join(section_patterns)
    
    # Find all section headers and their positions
    matches = list(re.finditer(combined_pattern, text, re.IGNORECASE))
    
    if not matches:
        # If no sections found, return entire text as "Full Text"
        return {"Full Text": text}
    
    for i, match in enumerate(matches):
        section_name = match.group().strip()
        start_pos = match.end()
        
        # Get content until next section or end of text
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start_pos:end_pos].strip()
        
        sections[section_name] = content
    
    return sections


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing excessive whitespace and artifacts.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive whitespace
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove page numbers (common pattern: standalone numbers)
    text = re.sub(r'\n\d+\n', '\n', text)
    
    return text.strip()


def extract_and_parse_paper(pdf_path: str) -> Dict[str, str]:
    """
    Complete pipeline: extract text from PDF and parse into sections.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary of sections with cleaned content
    """
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    sections = split_by_sections(cleaned_text)
    
    return sections
