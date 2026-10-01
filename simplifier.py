import os
# Ensure critical env vars are set BEFORE importing transformers to avoid hangs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import pipeline, AutoTokenizer
import torch
from functools import lru_cache
from transformers.utils import logging as hf_logging
from typing import List, Dict, Tuple, Optional
import textstat
from enum import Enum
import re
import warnings

# Silence verbose transformers warnings (e.g., Pegasus weight init notices)
hf_logging.set_verbosity_error()
# Suppress noisy third-party warnings
warnings.filterwarnings("ignore", message=".*_register_pytree_node is deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*resume_download.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*tokenizer was incorrectly instantiated with a model max length.*", category=FutureWarning)

# Be conservative on CPU thread usage to avoid starving Streamlit/app threads
try:
    torch.set_num_threads(max(1, min(torch.get_num_threads() or 1, 4)))
except Exception:
    pass


class SimplificationMode(Enum):
    """Different modes of paper simplification."""
    CORE = "core"  # Standard summarization
    ELI5 = "eli5"  # Multi-level explanations
    COMPARATIVE = "comparative"  # Compare multiple papers
    BULLETS = "bullets"  # One-line bullet points


class PaperSimplifier:
    """
    Simplifies academic paper text using pretrained summarization models.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn", mode: SimplificationMode = SimplificationMode.CORE):
        """
        Initialize the simplifier with a specific model and mode.
        
        Args:
            model_name: HuggingFace model identifier (default: BART)
            mode: Simplification mode (CORE, ELI5, or COMPARATIVE)
        """
        print(f"Loading model: {model_name}...")
        self.model_name = model_name
        self.is_t5 = "flan-t5" in model_name.lower() or "t5" in model_name.lower()
        
        # Ensure tokenizer matches model and set explicit max length to avoid warnings
        # Flan-T5/T5 models have a 512 token encoder limit; do not inflate this to 1024
        tokenizer_max_len = 512 if self.is_t5 else 1024
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=tokenizer_max_len, use_fast=True
        )
        self.summarizer = pipeline(
            "summarization", model=model_name, tokenizer=tokenizer
        )
        self.mode = mode
        print(f"Model loaded successfully! Mode: {mode.value}")
    
    def simplify_text(self, text: str, max_length: int = 600, min_length: int = 300, style: Optional[str] = None) -> str:
        """
        Simplify a chunk of text using summarization.
        
        Args:
            text: Input text to simplify
            max_length: Maximum length of summary (tokens)
            min_length: Minimum length of summary (tokens)
            style: Optional audience tag: 'expert' | 'student' | 'layman'
            
        Returns:
            Simplified text
        """
        # Basic guard for very short inputs
        if not text or len(text.strip()) < 10:
            return text
        
        # Limit input to ~1024 tokens (approx by chars); apply to content only
        max_input_chars = 4000
        content = text if len(text) <= max_input_chars else text[:max_input_chars]
        
        # Flan-T5/T5 models need task prefix
        if self.is_t5:
            content = f"summarize: {content}"
            # Guardrails for Flan-T5/T5 generation to avoid long-running decoding
            # Cap output length for Flan-T5-base to keep decoding responsive
            max_length = min(max_length, 256)
            # Keep min_length reasonable and below max_length
            min_length = max(50, min(min_length, max_length - 32))
        
        try:
            result = self.summarizer(
                content,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                truncation=True
            )
            summary = result[0]['summary_text']
            # Only strip if we have style instructions (ELI5 mode)
            if style:
                summary = self._strip_instructional_phrases(summary)
            if style and style.lower() in {"student", "layman"}:
                summary = self._rewrite_for_audience(summary, style.lower())
            # Guard against empty output
            if not summary or len(summary.strip()) < 10:
                return content[:500] + "..."
            return summary
        except Exception as e:
            return f"Error during simplification: {str(e)}"
    
    def simplify_sections(self, sections: Dict[str, str]) -> Dict[str, str]:
        """
        Simplify each section of a paper independently.
        
        Args:
            sections: Dictionary mapping section names to their content
            
        Returns:
            Dictionary mapping section names to simplified content
        """
        simplified = {}
        
        for section_name, content in sections.items():
            # Skip empty sections or references
            if not content or section_name.upper() == "REFERENCES":
                continue
            
            print(f"Simplifying: {section_name}...")
            simplified[section_name] = self.simplify_text(content)
        
        return simplified
    
    def format_as_bullets(self, simplified_sections: Dict[str, str]) -> str:
        """
        Format simplified sections as readable bullet points.
        
        Args:
            simplified_sections: Dictionary of simplified text by section
            
        Returns:
            Formatted string with bullet points
        """
        output = []
        
        for section, content in simplified_sections.items():
            output.append(f"**{section}**")
            
            # Split into sentences for bullet points
            sentences = content.split('. ')
            for sentence in sentences:
                if sentence.strip():
                    output.append(f"• {sentence.strip()}")
            
            output.append("")  # Blank line between sections
        
        return "\n".join(output)
    
    def get_readability_score(self, text: str) -> Dict[str, float]:
        """
        Calculate readability metrics for the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of readability metrics
        """
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "gunning_fog": textstat.gunning_fog(text)
        }
    
    def _strip_instructional_phrases(self, text: str) -> str:
        """Remove any leftover instruction-like sentences from the output."""
        if not text:
            return text
        # More conservative pattern - only remove if it starts the sentence
        patterns = re.compile(
            r"(?i)^(summarize for|explain (like|.*like i'm 5|eli5)|use (simple|clear|short|everyday)|avoid (formulas|jargon))"
        )
        sentences = re.split(r"(?<=[\.!?])\s+", text)
        filtered = [s for s in sentences if not patterns.search(s.strip())]
        cleaned = " ".join(filtered).strip()
        # Collapse duplicate spaces
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned if cleaned else text  # Return original if everything filtered
    
    def _rewrite_for_audience(self, text: str, audience: str) -> str:
        """Heuristically adjust language for student/layman audiences."""
        if not text:
            return text
        repl_common = {
            r"\butilize\b": "use",
            r"\bapproximately\b": "about",
            r"\bdemonstrate\b": "show",
            r"\billustrate\b": "show",
            r"\bobjective\b": "goal",
            r"\bmethodology\b": "method",
            r"\balgorithm\b": "method",
            r"\barchitecture\b": "design",
            r"\bparameters\b": "settings",
            r"\boptimi[sz]e\b": "improve",
            r"\boptimization\b": "improving",
            r"\brobust\b": "reliable",
            r"\bsignificant\b": "important",
            r"\bprior\b": "earlier",
            r"\bsubsequently\b": "then",
            r"\btherefore\b": "so",
            r"\bhowever\b": "but",
            r"\bin order to\b": "to",
            r"\bperform\b": "do",
            r"\bobtain\b": "get",
            r"\bcompute\b": "calculate",
            r"\bnumerous\b": "many",
            r"\bvarious\b": "many"
        }
        repl_layman = {
            r"\bdataset\b": "data",
            r"\bclassification\b": "sorting",
            r"\bclassify\b": "sort",
            r"\bcluster(?:ing)?\b": "group",
            r"\bregression\b": "prediction",
            r"\bfeatures\b": "things we look at",
            r"\blabels\b": "answers",
            r"\bhyperparameters\b": "settings",
            r"\binference\b": "prediction",
            r"\bprecision\b": "exactness",
            r"\brecall\b": "how much we found",
            r"\baccuracy\b": "how many we got right",
            r"\bbaseline\b": "simple starting method",
            r"\bmodel\b": "program",
            r"\bframework\b": "tool",
            r"\binstance\b": "example"
        }
        out = text
        # Apply common replacements
        for pat, rep in repl_common.items():
            out = re.sub(pat, rep, out, flags=re.IGNORECASE)
        if audience == "layman":
            for pat, rep in repl_layman.items():
                out = re.sub(pat, rep, out, flags=re.IGNORECASE)
            # Shorten long sentences by splitting at commas where possible
            sentences = re.split(r"(?<=[\.!?])\s+", out)
            new_sentences = []
            for s in sentences:
                words = s.split()
                if len(words) > 22 and "," in s:
                    parts = [p.strip() for p in s.split(",")]
                    for p in parts:
                        if p:
                            new_sentences.append(p.strip() + ".")
                else:
                    new_sentences.append(s)
            out = " ".join(new_sentences)
        return re.sub(r"\s+", " ", out).strip()
    
    def generate_key_takeaways(self, simplified_sections: Dict[str, str]) -> str:
        """
        Create a detailed "key takeaways" summary from simplified sections.
        
        Args:
            simplified_sections: Dictionary of simplified content
            
        Returns:
            Key takeaways string
        """
        # Combine all simplified content
        all_content = " ".join(simplified_sections.values())
        
        # Generate longer, more detailed summary
        try:
            takeaway = self.simplify_text(all_content, max_length=400, min_length=200)
            return f"**🔑 Key Takeaway:**\n{takeaway}"
        except:
            # Fallback: use first simplified section
            first_section = next(iter(simplified_sections.values()))
            return f"**🔑 Key Takeaway:**\n{first_section[:500]}..."
    
    def simplify_eli5(self, sections: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """
        Generate three levels of explanation: Expert, Student, and Layman.
        
        Args:
            sections: Paper sections to simplify
            
        Returns:
            Dictionary with three levels of simplified content
        """
        levels = {
            "expert": {},
            "student": {},
            "layman": {}
        }
        
        for section_name, content in sections.items():
            if not content or section_name.upper() == "REFERENCES":
                continue
            
            print(f"Generating ELI5 levels for: {section_name}...")
            
            # Expert level: Technical summary (400-600 tokens)
            levels["expert"][section_name] = self.simplify_text(
                content, max_length=600, min_length=400, style="expert"
            )
            
            # Student level: Moderate simplification (250-400 tokens)
            levels["student"][section_name] = self.simplify_text(
                content, max_length=400, min_length=250, style="student"
            )
            
            # Layman level: Heavy simplification (100-200 tokens)
            levels["layman"][section_name] = self.simplify_text(
                content, max_length=200, min_length=100, style="layman"
            )
        
        return levels
    
    def _sentences(self, text: str) -> List[str]:
        if not text:
            return []
        # Normalize newlines and split into sentences
        cleaned = re.sub(r"\s+", " ", text).strip()
        parts = re.split(r"(?<=[\.!?])\s+", cleaned)
        # Remove empty and very short fragments
        return [p.strip() for p in parts if len(p.strip()) > 1]

    def _normalize_sentence_for_match(self, text: str) -> str:
        """Normalize a sentence for loose matching across papers.

        Lowercase, strip punctuation, and collapse whitespace so we can
        identify roughly-shared vs unique sentences across summaries.
        """
        if not text:
            return ""
        normalized = text.lower()
        # Remove punctuation and non-alphanumeric chars
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        # Collapse multiple spaces
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized
    
    def bulletize_sections(self, sections: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Create one-line bullet points per section covering key details.
        Uses summarization first for coverage, then splits into sentences.
        """
        bullets: Dict[str, List[str]] = {}
        for section_name, content in sections.items():
            if not content or section_name.upper() == "REFERENCES":
                continue
            # Produce a detailed summary first for coverage
            summary = self.simplify_text(content, max_length=600, min_length=250)
            lines = [f"• {s}" for s in self._sentences(summary)]
            bullets[section_name] = lines
        return bullets
    
    def compare_papers(self, papers_sections: List[Tuple[str, Dict[str, str]]]) -> Dict:
        """
        Compare multiple papers and identify overlaps/conflicts.
        
        Args:
            papers_sections: List of (paper_name, sections) tuples
            
        Returns:
            Dictionary with comparison analysis
        """
        comparison = {
            "individual_summaries": {},
            "common_themes": [],
            "differences": []
        }
        
        # Simplify each paper
        for paper_name, sections in papers_sections:
            print(f"Simplifying paper: {paper_name}...")
            simplified = self.simplify_sections(sections)
            combined = " ".join(simplified.values())
            
            # Generate overall summary for this paper
            paper_summary = self.simplify_text(combined, max_length=300, min_length=150)
            comparison["individual_summaries"][paper_name] = paper_summary
        
        # Generate comparative analysis
        all_summaries = " ".join(comparison["individual_summaries"].values())
        
        print("Generating comparative analysis...")
        # Ask model to identify commonalities with a high-level summary prompt
        common_prompt = (
            "Summarize the following research papers and focus ONLY on their "
            "COMMON THEMES and SHARED ideas, methods, or findings. "
            "Do not mention any differences, conflicts, or unique aspects.\n\n" 
            f"{all_summaries[:2000]}"
        )
        try:
            common = self.simplify_text(common_prompt, max_length=250, min_length=100)
            comparison["common_themes"] = common
        except Exception:
            comparison["common_themes"] = "Analysis of common themes unavailable."
        
        # Now compute differences/conflicts heuristically so this section is
        # structurally distinct from the common-themes summary.
        #
        # We treat sentences that appear (after normalization) in multiple
        # paper summaries as "common" and sentences that appear in only one
        # paper as "unique" to that paper.
        sentence_index = {}
        paper_sentences: Dict[str, List[str]] = {}
        
        for paper_name, summary in comparison["individual_summaries"].items():
            sents = self._sentences(summary)
            paper_sentences[paper_name] = sents
            for s in sents:
                norm = self._normalize_sentence_for_match(s)
                if not norm:
                    continue
                entry = sentence_index.setdefault(norm, {"text": s, "papers": set()})
                entry["papers"].add(paper_name)
        
        # Build human-readable strings
        common_lines = []
        unique_by_paper: Dict[str, List[str]] = {name: [] for name in paper_sentences}
        
        for norm, entry in sentence_index.items():
            papers_here = entry["papers"]
            if len(papers_here) >= 2:
                # Shared sentence across multiple papers
                common_lines.append(f"• {entry['text']}")
            else:
                # Unique to a single paper
                only_paper = next(iter(papers_here))
                unique_by_paper[only_paper].append(entry["text"])
        
        # If we managed to find explicit shared sentences, prepend them to
        # the model-generated common themes text so the user sees both.
        if common_lines:
            bullet_block = "\n".join(common_lines)
            if comparison["common_themes"]:
                comparison["common_themes"] = f"{comparison['common_themes']}\n\n**Shared points detected across papers:**\n{bullet_block}"
            else:
                comparison["common_themes"] = f"**Shared points detected across papers:**\n{bullet_block}"
        
        # Differences/conflicts: report unique sentences per paper.
        diff_lines: List[str] = []
        for paper_name, unique_sents in unique_by_paper.items():
            if not unique_sents:
                continue
            diff_lines.append(f"**{paper_name} – unique points and potential conflicts:**")
            for s in unique_sents:
                diff_lines.append(f"• {s}")
            diff_lines.append("")
        
        if diff_lines:
            comparison["differences"] = "\n".join(diff_lines)
        else:
            comparison["differences"] = "No clear differences detected based on the generated summaries."
        
        return comparison


@lru_cache(maxsize=6)
def _get_simplifier_cached(model_name: str, mode_value: str) -> "PaperSimplifier":
    mode_enum = SimplificationMode(mode_value)
    return PaperSimplifier(model_name, mode_enum)


def simplify_paper(
    sections: Dict[str, str] = None,
    model_name: str = "facebook/bart-large-cnn",
    mode: str = "core",
    papers_sections: List[Tuple[str, Dict[str, str]]] = None
) -> Dict:
    """
    Complete simplification pipeline with multiple modes.
    
    Args:
        sections: Paper sections from extractor (for CORE/ELI5 modes)
        model_name: Model to use for simplification
        mode: Simplification mode ("core", "eli5", or "comparative")
        papers_sections: List of (name, sections) for comparative mode
        
    Returns:
        Dictionary with simplified content based on mode
    """
    # Reuse model across calls to avoid repeated downloads/initialization
    simplifier = _get_simplifier_cached(model_name, mode)
    
    if mode == "core":
        # Standard simplification
        simplified = simplifier.simplify_sections(sections)
        formatted_output = simplifier.format_as_bullets(simplified)
        key_takeaways = simplifier.generate_key_takeaways(simplified)
        
        full_simplified_text = " ".join(simplified.values())
        readability = simplifier.get_readability_score(full_simplified_text)
        
        return {
            "mode": "core",
            "simplified_sections": simplified,
            "formatted_output": formatted_output,
            "key_takeaways": key_takeaways,
            "readability": readability
        }
    
    elif mode == "eli5":
        # Multi-level explanations
        levels = simplifier.simplify_eli5(sections)
        
        # Calculate readability for each level
        readability_scores = {}
        for level_name, level_sections in levels.items():
            level_text = " ".join(level_sections.values())
            readability_scores[level_name] = simplifier.get_readability_score(level_text)
        
        return {
            "mode": "eli5",
            "levels": levels,
            "readability_scores": readability_scores
        }
    
    elif mode == "comparative":
        # Compare multiple papers
        if not papers_sections:
            raise ValueError("papers_sections required for comparative mode")
        
        comparison = simplifier.compare_papers(papers_sections)
        
        return {
            "mode": "comparative",
            "comparison": comparison
        }
    
    elif mode == "bullets":
        # Bullet points mode
        bullets_by_section = simplifier.bulletize_sections(sections)
        combined_text = "\n".join([line for lines in bullets_by_section.values() for line in lines])
        readability = simplifier.get_readability_score(combined_text)
        return {
            "mode": "bullets",
            "bullets_by_section": bullets_by_section,
            "readability": readability
        }
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
