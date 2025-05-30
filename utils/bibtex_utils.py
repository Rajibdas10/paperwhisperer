import re

def generate_bibtex_from_arxiv(raw_references: str) -> str:
    """
    Basic BibTeX generator for arXiv-style references.
    This version uses regex to identify arXiv IDs and create BibTeX entries.
    """
    entries = []
    refs = raw_references.strip().split("\n")

    for idx, ref in enumerate(refs, 1):
        arxiv_match = re.search(r'arXiv:(\d+\.\d+)', ref)
        if arxiv_match:
            arxiv_id = arxiv_match.group(1)
            bibtex = f"""@article{{ref{idx},
  author    = {{{ref.split('.')[0]}}}, 
  title     = {{{' '.join(ref.split('.')[1:]).strip()}}}, 
  journal   = {{arXiv preprint arXiv:{arxiv_id}}}, 
  year      = {{20{arxiv_id[:2]}}}, 
}}"""
            entries.append(bibtex)
        else:
            entries.append(f"% Could not parse reference: {ref}")

    return "\n\n".join(entries)
