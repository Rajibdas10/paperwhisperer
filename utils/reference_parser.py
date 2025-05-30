import re
import string

def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove common artifacts
    text = re.sub(r'^\d+\.\s*', '', text)  # Remove leading numbers
    text = re.sub(r'^\[\d+\]\s*', '', text)  # Remove [1] style
    return text

def extract_year(text):
    """Extract year from reference text"""
    # Look for 4-digit years (1900-2099)
    year_patterns = [
        r'\b((?:19|20)\d{2})\b',  # Standard 4-digit year
        r'\(((?:19|20)\d{2})\)',  # Year in parentheses
    ]
    
    for pattern in year_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Return the most recent year found
            years = [int(year) for year in matches]
            return str(max(years))
    return None

def extract_authors(text):
    """Extract authors from reference text - improved logic"""
    text = clean_text(text)
    
    # Strategy 1: Look for author patterns at the beginning
    author_patterns = [
        # Pattern: "LastName, FirstName and LastName, FirstName" or "LastName, F. and LastName, F."
        r'^([A-Z][a-z]+(?:,\s*[A-Z][a-z]*\.?\s*)+(?:\s+and\s+[A-Z][a-z]+(?:,\s*[A-Z][a-z]*\.?\s*)+)*)',
        # Pattern: "FirstName LastName and FirstName LastName"
        r'^([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+\s+[A-Z][a-z]+)*(?:\s*,?\s+and\s+[A-Z][a-z]+\s+[A-Z][a-z]+)?)',
        # Pattern: "LastName et al."
        r'^([A-Z][a-z]+\s+et\s+al\.?)',
        # Pattern: Multiple authors with commas, ending with "and" or before year
        r'^([A-Z][^.]+?)(?:\s+\(?\d{4}\)?|\.\s+[A-Z])',
    ]
    
    for pattern in author_patterns:
        match = re.search(pattern, text)
        if match:
            authors = match.group(1).strip()
            # Clean up
            authors = re.sub(r'\.$', '', authors)  # Remove trailing period
            authors = re.sub(r'\s+', ' ', authors)  # Normalize whitespace
            
            # Validate it looks like authors (not a title)
            if not re.search(r'\b(?:the|a|an|on|in|for|with|using|learning|neural|deep|transformer|attention)\b', authors.lower()):
                return authors
    
    # Fallback: Extract everything before the first period or year, but validate
    before_period = re.split(r'[\.\(]\s*(?:19|20)\d{2}', text)[0].strip()
    if before_period and len(before_period) < 150:  # Reasonable length for authors
        before_period = re.sub(r'\.$', '', before_period)
        # Check if it looks like authors (contains names, not article words)
        if re.search(r'[A-Z][a-z]+', before_period) and not re.search(r'\b(?:the|a|an|on|in|for|with|using|learning|neural|deep|transformer|attention|analysis|method|model|approach)\b', before_period.lower()):
            return before_period
    
    return "Unknown Author"

def extract_title(text):
    """Extract title from reference text - improved logic"""
    text = clean_text(text)
    
    # Strategy 1: Remove author part first
    remaining_text = text
    
    # Try to identify where authors end
    author_end_patterns = [
        r'^[^.]+\.\s*',  # Everything before first period
        r'^[^(]+\(\d{4}\)\.\s*',  # Author (year). pattern
        r'^[A-Z][^,]+(?:,\s*[A-Z][^,]*)*\.\s*',  # Author list ending with period
    ]
    
    for pattern in author_end_patterns:
        match = re.search(pattern, text)
        if match:
            potential_title = text[match.end():].strip()
            if len(potential_title) > 10:  # Must have substantial content
                remaining_text = potential_title
                break
    
    # Strategy 2: Look for title patterns
    title_patterns = [
        # Quoted titles
        r'"([^"]+)"',
        r"'([^']+)'", 
        # Title before venue indicators
        r'^([^.]+?)\.?\s+(?:In\s+|Proceedings|Conference|Journal|IEEE|ACM|arXiv)',
        # Title before arXiv
        r'^([^.]+?)(?:\s+arXiv\s*preprint|\s+arXiv:)',
        # Title as first substantial sentence
        r'^([^.]{10,}?)\.(?:\s+[A-Z]|\s*$)',
        # Everything before venue markers
        r'^([^.]+?)(?:\s+In\s+|\s+Proceedings)',
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, remaining_text, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            # Clean title
            title = re.sub(r'^["\']|["\']$', '', title)
            title = re.sub(r'\s+', ' ', title)
            title = re.sub(r'^\W+|\W+$', '', title)
            
            # Validate it's a reasonable title
            if len(title) > 5 and not re.match(r'^\d+$', title) and not title.lower() in ['unknown', 'et al']:
                return title
    
    # Fallback: First substantial chunk that looks like a title
    sentences = re.split(r'[.!?]', remaining_text)
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10 and not re.match(r'^\d+$', sentence):
            # Check if it contains title-like words
            if re.search(r'\b(?:learning|neural|deep|transformer|attention|analysis|method|model|approach|system|framework|algorithm)\b', sentence.lower()):
                return sentence
    
    return "Unknown Title"

def extract_venue(text):
    """Extract venue/journal from reference text"""
    venue_patterns = [
        r'In\s+([^,.\n]+(?:Conference|Workshop|Symposium|Proceedings)[^,.\n]*)',
        r'arXiv\s+preprint\s+(arXiv:\d+\.\d+)',
        r'Journal\s+of\s+([^,.\n]+)',
        r'Proceedings\s+of\s+([^,.\n]+)',
        r'\.\s+([A-Z][^,.\n]*(?:Journal|Review|Letters|Conference)[^,.\n]*)',
        r'In\s+([^,.\n]+)',  # Generic "In" pattern
    ]
    
    for pattern in venue_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            venue = match.group(1).strip()
            # Clean venue
            venue = re.sub(r'^\W+|\W+$', '', venue)
            if len(venue) > 3:
                return venue
    
    return None

def split_references(text):
    """Split text into individual references"""
    # Remove the "References" header
    text = re.sub(r'^\s*References?\s*\n?', '', text, flags=re.IGNORECASE)
    
    references = []
    
    # Strategy 1: Numbered references [1], [2], etc.
    if re.search(r'\[\d+\]', text):
        parts = re.split(r'\n?\s*\[\d+\]\s*', text)
        references = [part.strip() for part in parts if part.strip()]
    
    # Strategy 2: Sequential numbers 1., 2., etc.
    elif re.search(r'^\d+\.\s+', text, re.MULTILINE):
        parts = re.split(r'\n?\s*\d+\.\s+', text)
        references = [part.strip() for part in parts if part.strip()]
    
    # Strategy 3: Author-year format (split by author patterns)
    elif re.search(r'\n[A-Z][a-z]+,?\s+[A-Z]', text):
        parts = re.split(r'\n(?=[A-Z][a-z]+[,\s])', text)
        references = [part.strip() for part in parts if part.strip()]
    
    # Strategy 4: Double newline separation
    else:
        parts = re.split(r'\n\s*\n', text)
        references = [part.strip() for part in parts if part.strip()]
    
    # Filter out very short references
    references = [ref for ref in references if len(ref) > 30]
    
    return references

def generate_bibtex_key(authors, year, title):
    """Generate a clean BibTeX key"""
    # Extract first author's last name
    if authors and authors != "Unknown Author":
        first_author = authors.split(',')[0].split(' and ')[0].strip()
        last_name = first_author.split()[-1] if first_author else "unknown"
    else:
        last_name = "unknown"
    
    # Clean last name
    last_name = re.sub(r'[^a-zA-Z]', '', last_name).lower()
    
    # Extract first meaningful word of title
    if title and title != "Unknown Title":
        # Skip common words
        title_words = title.split()
        meaningful_word = None
        for word in title_words:
            clean_word = re.sub(r'[^a-zA-Z]', '', word).lower()
            if clean_word and clean_word not in ['the', 'a', 'an', 'on', 'in', 'for', 'with', 'using']:
                meaningful_word = clean_word
                break
        title_word = meaningful_word or "paper"
    else:
        title_word = "paper"
    
    year_str = year or "2023"
    
    return f"{last_name}{year_str}{title_word}"

def parse_reference_to_bibtex(reference, index=1):
    """Parse a single reference into BibTeX format"""
    reference = clean_text(reference)
    
    if not reference or len(reference) < 20:
        return f"@misc{{ref{index},\n  note = {{Invalid reference}}\n}}"
    
    # Extract components
    authors = extract_authors(reference)
    title = extract_title(reference)
    year = extract_year(reference)
    venue = extract_venue(reference)
    
    # Generate BibTeX key
    bibtex_key = generate_bibtex_key(authors, year, title)
    
    # Determine publication type
    pub_type = "article"
    if any(keyword in reference.lower() for keyword in ["conference", "proceedings", "workshop", "symposium"]):
        pub_type = "inproceedings"
    elif "arxiv" in reference.lower():
        pub_type = "misc"
    elif any(keyword in reference.lower() for keyword in ["book", "chapter"]):
        pub_type = "book"
    
    # Build BibTeX entry
    bibtex_lines = [f"@{pub_type}{{{bibtex_key},"]
    
    if authors and authors != "Unknown Author":
        bibtex_lines.append(f"  author = {{{authors}}},")
    
    if title and title != "Unknown Title":
        bibtex_lines.append(f"  title = {{{title}}},")
    
    if year:
        bibtex_lines.append(f"  year = {{{year}}},")
    
    if venue:
        if pub_type == "inproceedings":
            bibtex_lines.append(f"  booktitle = {{{venue}}},")
        elif pub_type == "article":
            bibtex_lines.append(f"  journal = {{{venue}}},")
        else:
            bibtex_lines.append(f"  publisher = {{{venue}}},")
    
    # Add arXiv ID if present
    arxiv_match = re.search(r'arXiv:(\d+\.\d+)', reference)
    if arxiv_match:
        bibtex_lines.append(f"  eprint = {{{arxiv_match.group(1)}}},")
        bibtex_lines.append(f"  archivePrefix = {{arXiv}},")
    
    bibtex_lines.append("}")
    
    return "\n".join(bibtex_lines)

def parse_multiple_references_to_bibtex(text):
    """Parse multiple references and convert to BibTeX format"""
    if not text or not text.strip():
        return "% No references provided"
    
    references = split_references(text)
    
    if not references:
        return "% No valid references found"
    
    bibtex_entries = []
    for idx, ref in enumerate(references, 1):
        if ref.strip():
            bibtex_entry = parse_reference_to_bibtex(ref, idx)
            bibtex_entries.append(bibtex_entry)
    
    return "\n\n".join(bibtex_entries)