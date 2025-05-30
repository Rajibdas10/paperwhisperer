import requests
import xml.etree.ElementTree as ET

# Tier 1: Semantic Scholar + DOI to BibTeX
def get_bibtex_from_title(title):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&limit=1&fields=title,externalIds"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if data['data']:
            paper = data['data'][0]
            external_ids = paper.get("externalIds", {})
            doi = external_ids.get("DOI")
            if doi:
                return get_bibtex_from_doi(doi)
    return None

def get_bibtex_from_doi(doi):
    doi = doi.strip()
    url = f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
    response = requests.get(url, headers={"Accept": "application/x-bibtex"})
    if response.status_code == 200:
        return response.text
    return None


# Tier 2: CrossRef fallback (search by title)
def get_crossref_bibtex(title):
    url = f"https://api.crossref.org/works?query.title={title}&rows=1"
    response = requests.get(url)
    if response.status_code == 200:
        items = response.json().get("message", {}).get("items", [])
        if items:
            doi = items[0].get("DOI")
            if doi:
                return get_bibtex_from_doi(doi)
    return None


# Tier 3: ArXiv fallback (if CrossRef + DOI fail)
def get_arxiv_bibtex(title):
    query = title.replace(" ", "+")
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=1"
    response = requests.get(url)
    if response.status_code == 200 and "<entry>" in response.text:
        try:
            root = ET.fromstring(response.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entry = root.find("atom:entry", ns)
            if entry is not None:
                paper_title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                authors = " and ".join(
                    author.find("atom:name", ns).text for author in entry.findall("atom:author", ns)
                )
                published = entry.find("atom:published", ns).text
                year = published[:4]
                arxiv_id = entry.find("atom:id", ns).text.split("/")[-1]
                return f"""@article{{arxiv{arxiv_id},
  title={{ {paper_title} }},
  author={{ {authors} }},
  year={{ {year} }},
  archivePrefix={{arXiv}},
  eprint={{ {arxiv_id} }}
}}"""
        except Exception as e:
            print("ArXiv parsing error:", e)
    return None


# Final wrapper function
def robust_bibtex_fetch(title):
    bib = get_bibtex_from_title(title)
    if bib:
        return bib
    bib = get_crossref_bibtex(title)
    if bib:
        return bib
    bib = get_arxiv_bibtex(title)
    return bib or "❌ BibTeX not found. Please verify title."
