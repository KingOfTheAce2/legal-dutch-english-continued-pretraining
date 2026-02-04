#!/usr/bin/env python3
"""
Legal CPT Data Pipeline v1.0
=============================
Fully automated pipeline to download, clean, chunk and balance legal text
for continued pre-training (CPT) of language models.

Target distribution: 50% Dutch · 25% UK English · 25% US English

Sources (all public domain / CC-0 / Open Government Licence):
  Dutch:
    1. Rechtspraak.nl       – Court decisions           (public domain)
    2. BWB / wetten.overheid – Legislation via SRU + XML (CC-0)
  UK English:
    3. legislation.gov.uk    – UK Acts of Parliament     (Open Government Licence v3)
  US English:
    4. eCFR                  – Federal Regulations       (public domain, US gov work)

Usage:
    pip install -r requirements.txt
    python legal_cpt_pipeline.py                          # defaults: 15M tokens
    python legal_cpt_pipeline.py --target-tokens 20000000 # custom target
    python legal_cpt_pipeline.py --test                   # quick smoke test

The pipeline is resume-capable: re-run safely after interruption.
"""

import os
import sys
import re
import json
import time
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════

# Default target: 20M tokens is a strong CPT baseline for single-GPU QLoRA.
# Rechtspraak.nl alone has 800k+ decisions (500M+ tokens available).
# Adjust upward freely — returns diminish above ~50M for domain CPT.
DEFAULT_TARGET_TOKENS = 20_000_000
DUTCH_RATIO  = 0.50
UK_RATIO     = 0.25
US_RATIO     = 0.25

CHUNK_TARGET_TOKENS = 1024
CHUNK_MAX_TOKENS    = 2048
CHUNK_MIN_TOKENS    = 100

REQUEST_DELAY  = 0.5          # seconds between HTTP requests (be respectful)
MAX_RETRIES    = 3
REQUEST_TIMEOUT = 60

HEADERS = {
    "User-Agent": (
        "LegalCPTPipeline/1.0 "
        "(Academic-Research; legal-domain-CPT-data; contact: see repo)"
    ),
    "Accept": "application/xml, text/html, text/xml, */*",
}

# Atom namespace used by Rechtspraak.nl feeds
ATOM_NS = "http://www.w3.org/2005/Atom"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("legal_cpt")


# ════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (works for NL + EN)."""
    return max(1, len(text) // 4)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def clean_legal_text(text: str) -> str:
    """Normalize whitespace and remove non-text artefacts."""
    # collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # normalize line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    # strip page numbers like "- 3 -" or "page 12"
    text = re.sub(r"(?m)^[ \t]*-\s*\d+\s*-[ \t]*$", "", text)
    text = re.sub(r"(?mi)^[ \t]*page\s+\d+[ \t]*$", "", text)
    # strip stray XML/HTML entities
    text = re.sub(r"&[a-z]+;", " ", text)
    # remove leading/trailing whitespace per line
    lines = [l.strip() for l in text.splitlines()]
    text = "\n".join(lines)
    return text.strip()


def chunk_text(text: str,
               target: int = CHUNK_TARGET_TOKENS,
               max_size: int = CHUNK_MAX_TOKENS,
               min_size: int = CHUNK_MIN_TOKENS) -> list[str]:
    """Split text into chunks on paragraph boundaries."""
    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    current: list[str] = []
    current_tok = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        ptok = estimate_tokens(para)
        if ptok < 20:
            continue  # skip trivial fragments

        # if adding this paragraph exceeds max and we have content, flush
        if current_tok + ptok > max_size and current:
            chunks.append("\n\n".join(current))
            current = []
            current_tok = 0

        current.append(para)
        current_tok += ptok

        if current_tok >= target:
            chunks.append("\n\n".join(current))
            current = []
            current_tok = 0

    if current and current_tok >= min_size:
        chunks.append("\n\n".join(current))

    return chunks


def http_get(url: str, params: dict | None = None,
             delay: float = REQUEST_DELAY,
             timeout: int = REQUEST_TIMEOUT) -> requests.Response | None:
    """GET with retries, backoff, and rate limiting."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(delay)
            resp = requests.get(url, params=params, headers=HEADERS,
                                timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                log.warning("Rate limited on %s – waiting %ds", url, wait)
                time.sleep(wait)
                continue
            if resp.status_code in (404, 410):
                log.debug("Not found: %s", url)
                return None
            log.warning("HTTP %d on %s (attempt %d/%d)",
                        resp.status_code, url, attempt, MAX_RETRIES)
        except requests.RequestException as exc:
            log.warning("Request error on %s: %s (attempt %d/%d)",
                        url, exc, attempt, MAX_RETRIES)
        time.sleep(2 ** attempt)
    return None


def get_all_text(element) -> list[str]:
    """Recursively extract all text from an XML element."""
    parts = []
    if element.text and element.text.strip():
        parts.append(element.text.strip())
    for child in element:
        parts.extend(get_all_text(child))
        if child.tail and child.tail.strip():
            parts.append(child.tail.strip())
    return parts


def local_tag(elem) -> str:
    """Strip namespace from an XML tag."""
    tag = elem.tag
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


# ════════════════════════════════════════════════════════════════════════
# PROGRESS / RESUME
# ════════════════════════════════════════════════════════════════════════

class ProgressTracker:
    """Persist download progress so the pipeline can resume."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {}
        if path.exists():
            with open(path) as f:
                self.data = json.load(f)

    def is_done(self, source: str, doc_id: str) -> bool:
        return doc_id in self.data.get(source, {})

    def mark_done(self, source: str, doc_id: str, tokens: int = 0):
        self.data.setdefault(source, {})[doc_id] = tokens
        self._save()

    def count(self, source: str) -> int:
        return len(self.data.get(source, {}))

    def total_tokens(self, source: str) -> int:
        return sum(self.data.get(source, {}).values())

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f)


# ════════════════════════════════════════════════════════════════════════
# SOURCE 1: Rechtspraak.nl  (Dutch Court Decisions)
# ════════════════════════════════════════════════════════════════════════

RECHTSPRAAK_SEARCH = "https://data.rechtspraak.nl/uitspraken/zoeken"
RECHTSPRAAK_CONTENT = "https://data.rechtspraak.nl/uitspraken/content"


def rechtspraak_search_eclis(date_from: str, date_to: str,
                             max_results: int = 1000,
                             offset: int = 0) -> list[str]:
    """Search the ECLI index. Returns list of ECLI strings."""
    params = {
        "max": min(max_results, 1000),
        "from": offset,
        "date": f"{date_from} TO {date_to}",
        "sort": "DESC",
        "return": "DOC",   # only those with full text
    }
    resp = http_get(RECHTSPRAAK_SEARCH, params=params, delay=0.3)
    if resp is None:
        return []
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        log.warning("XML parse error in Rechtspraak search response")
        return []

    eclis = []
    for entry in root.iter(f"{{{ATOM_NS}}}entry"):
        id_elem = entry.find(f"{{{ATOM_NS}}}id")
        if id_elem is not None and id_elem.text:
            eclis.append(id_elem.text.strip())
    return eclis


def rechtspraak_fetch_text(ecli: str) -> str | None:
    """Download and extract plain text for one ECLI."""
    resp = http_get(RECHTSPRAAK_CONTENT, params={"id": ecli}, delay=0.15)
    if resp is None:
        return None
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return None

    # find <uitspraak> or <conclusie> regardless of namespace
    for elem in root.iter():
        ltag = local_tag(elem)
        if ltag in ("uitspraak", "conclusie"):
            parts = get_all_text(elem)
            text = "\n\n".join(parts)
            if estimate_tokens(text) >= CHUNK_MIN_TOKENS:
                return clean_legal_text(text)
    return None


def download_rechtspraak(out_dir: Path, progress: ProgressTracker,
                         target_tokens: int, years: range | None = None):
    """Download Dutch court decisions from Rechtspraak.nl."""
    source = "rechtspraak"
    out_dir.mkdir(parents=True, exist_ok=True)
    if years is None:
        years = range(2018, 2026)

    log.info("── Rechtspraak.nl: downloading court decisions ──")
    pbar = tqdm(desc="Rechtspraak", unit="doc")
    collected_tokens = progress.total_tokens(source)

    for year in years:
        if collected_tokens >= target_tokens:
            break
        date_from = f"{year}-01-01"
        date_to   = f"{year}-12-31"
        offset = 0

        while collected_tokens < target_tokens:
            eclis = rechtspraak_search_eclis(date_from, date_to,
                                             max_results=1000,
                                             offset=offset)
            if not eclis:
                break

            for ecli in eclis:
                if collected_tokens >= target_tokens:
                    break
                if progress.is_done(source, ecli):
                    pbar.update(1)
                    continue

                text = rechtspraak_fetch_text(ecli)
                if text is None:
                    progress.mark_done(source, ecli, 0)
                    pbar.update(1)
                    continue

                tok = estimate_tokens(text)
                fpath = out_dir / f"{ecli.replace(':', '_')}.txt"
                fpath.write_text(text, encoding="utf-8")
                progress.mark_done(source, ecli, tok)
                collected_tokens += tok
                pbar.update(1)
                pbar.set_postfix(tokens=f"{collected_tokens:,}")

            offset += len(eclis)
            if len(eclis) < 1000:
                break  # last page

    pbar.close()
    log.info("Rechtspraak: %d docs, ~%s tokens",
             progress.count(source), f"{collected_tokens:,}")


# ════════════════════════════════════════════════════════════════════════
# SOURCE 2: wetten.overheid.nl  (Dutch Legislation – BWB)
# ════════════════════════════════════════════════════════════════════════

BWB_SRU_URL = (
    "https://zoekservice.overheid.nl/sru/Search"
    "?operation=searchRetrieve&version=1.2&x-connection=BWB"
)

# Curated list of major Dutch laws — BWB identifiers.
# These are fetched directly from wetten.overheid.nl.
# The SRU search below supplements this list dynamically.
CURATED_BWB = [
    # Grondwet / Constitution
    "BWBR0001840",
    # Burgerlijk Wetboek (Civil Code) Books
    "BWBR0002656",  # BW Boek 1
    "BWBR0003045",  # BW Boek 2
    "BWBR0005291",  # BW Boek 3
    "BWBR0002761",  # BW Boek 4
    "BWBR0005288",  # BW Boek 5
    "BWBR0005289",  # BW Boek 6
    "BWBR0005290",  # BW Boek 7
    "BWBR0006000",  # BW Boek 7A
    "BWBR0005034",  # BW Boek 8
    # Procedural / criminal
    "BWBR0001854",  # Wetboek van Burgerlijke Rechtsvordering
    "BWBR0001903",  # Wetboek van Strafrecht
    "BWBR0001827",  # Wetboek van Strafvordering
    # Administrative
    "BWBR0005537",  # Algemene wet bestuursrecht
    # Financial / regulatory
    "BWBR0020368",  # Wet op het financieel toezicht
    "BWBR0002320",  # Algemene wet inzake rijksbelastingen
    # Company / insolvency
    "BWBR0001860",  # Faillissementswet
    "BWBR0002063",  # Wetboek van Koophandel
    # Labour
    "BWBR0008160",  # Arbeidsomstandighedenwet
    # Data / privacy
    "BWBR0040940",  # Uitvoeringswet AVG
    # Misc important
    "BWBR0001886",  # Gemeentewet
    "BWBR0005416",  # Politiewet 2012
]


def bwb_search_sru(query: str = "dcterms.type == wet",
                   max_records: int = 200) -> list[str]:
    """Search BWB via SRU and return BWB identifiers."""
    params = {
        "maximumRecords": max_records,
        "startRecord": 1,
        "query": query,
    }
    url = BWB_SRU_URL + f"&maximumRecords={max_records}&startRecord=1&query={query}"
    resp = http_get(url, delay=1.0, timeout=90)
    if resp is None:
        return []
    # Extract BWBR identifiers with regex (robust against namespace changes)
    return list(set(re.findall(r"(BWBR\d{7})", resp.text)))


def fetch_wetten_html(bwb_id: str) -> str | None:
    """Fetch legislation text from wetten.overheid.nl as HTML."""
    url = f"https://wetten.overheid.nl/{bwb_id}"
    resp = http_get(url, delay=0.8, timeout=90)
    if resp is None:
        return None
    soup = BeautifulSoup(resp.text, "lxml")

    # The main content is typically in <div class="wettekst"> or
    # in <div id="content">, or in article elements.
    # Try several selectors.
    text_parts = []

    # Try structured article elements
    for sel in ["div.wetartikel", "div.artikel", ".wettekst",
                "#wettenTekst", "#content .tekst", "#content"]:
        elems = soup.select(sel)
        if elems:
            for el in elems:
                txt = el.get_text(separator="\n", strip=True)
                if txt:
                    text_parts.append(txt)
            break

    # Fallback: get text from <main> or <body>
    if not text_parts:
        main = soup.find("main") or soup.find("body")
        if main:
            txt = main.get_text(separator="\n", strip=True)
            if txt:
                text_parts.append(txt)

    text = "\n\n".join(text_parts)
    # Remove common boilerplate
    text = re.sub(r"(?i)informatie.*?wetten\.overheid\.nl.*?\n", "", text)
    text = re.sub(r"(?i)zoek\s+in\s+deze\s+regeling.*?\n", "", text)

    if estimate_tokens(text) < CHUNK_MIN_TOKENS:
        return None
    return clean_legal_text(text)


def download_dutch_legislation(out_dir: Path, progress: ProgressTracker,
                               target_tokens: int):
    """Download Dutch legislation from wetten.overheid.nl."""
    source = "wetten_nl"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("── wetten.overheid.nl: downloading Dutch legislation ──")

    # Combine curated list with SRU search
    bwb_ids = list(CURATED_BWB)
    log.info("Searching BWB SRU for additional laws...")
    for query in ["dcterms.type == wet",
                  "dcterms.type == AMvB",
                  "dcterms.type == ministeriele-regeling"]:
        extras = bwb_search_sru(query, max_records=200)
        bwb_ids.extend(extras)
    bwb_ids = list(dict.fromkeys(bwb_ids))  # deduplicate, keep order
    log.info("Found %d unique BWB identifiers", len(bwb_ids))

    collected_tokens = progress.total_tokens(source)
    pbar = tqdm(bwb_ids, desc="Wetten.nl", unit="law")

    for bwb_id in pbar:
        if collected_tokens >= target_tokens:
            break
        if progress.is_done(source, bwb_id):
            continue
        text = fetch_wetten_html(bwb_id)
        if text is None:
            progress.mark_done(source, bwb_id, 0)
            continue
        tok = estimate_tokens(text)
        fpath = out_dir / f"{bwb_id}.txt"
        fpath.write_text(text, encoding="utf-8")
        progress.mark_done(source, bwb_id, tok)
        collected_tokens += tok
        pbar.set_postfix(tokens=f"{collected_tokens:,}")

    pbar.close()
    log.info("Wetten.nl: %d laws, ~%s tokens",
             progress.count(source), f"{collected_tokens:,}")


# ════════════════════════════════════════════════════════════════════════
# SOURCE 3: legislation.gov.uk  (UK Acts of Parliament)
# ════════════════════════════════════════════════════════════════════════

# Curated important UK Acts (type/year/number)
UK_ACTS = [
    # Companies
    ("ukpga", 2006, 46),   # Companies Act 2006
    # Data protection / privacy
    ("ukpga", 2018, 12),   # Data Protection Act 2018
    # Human rights
    ("ukpga", 1998, 42),   # Human Rights Act 1998
    # Equality
    ("ukpga", 2010, 15),   # Equality Act 2010
    # Consumer
    ("ukpga", 2015, 30),   # Consumer Rights Act 2015
    # FOI
    ("ukpga", 2000, 36),   # Freedom of Information Act 2000
    # Fraud
    ("ukpga", 2006, 35),   # Fraud Act 2006
    # Bribery
    ("ukpga", 2010, 23),   # Bribery Act 2010
    # Employment
    ("ukpga", 1996, 18),   # Employment Rights Act 1996
    # Financial Services
    ("ukpga", 2000, 8),    # Financial Services and Markets Act 2000
    ("ukpga", 2023, 29),   # Financial Services and Markets Act 2023
    # Contract
    ("ukpga", 1999, 31),   # Contracts (Rights of Third Parties) Act 1999
    # Limitation
    ("ukpga", 1980, 58),   # Limitation Act 1980
    # Sale of Goods
    ("ukpga", 1979, 54),   # Sale of Goods Act 1979
    # Insolvency
    ("ukpga", 1986, 45),   # Insolvency Act 1986
    # Arbitration
    ("ukpga", 1996, 23),   # Arbitration Act 1996
    # Criminal
    ("ukpga", 2003, 44),   # Criminal Justice Act 2003
    ("ukpga", 2020, 17),   # Sentencing Act 2020
    ("ukpga", 1968, 60),   # Theft Act 1968
    # Regulatory
    ("ukpga", 2016, 25),   # Investigatory Powers Act 2016
    ("ukpga", 1990, 18),   # Computer Misuse Act 1990
    ("ukpga", 2023, 50),   # Online Safety Act 2023
    # Property
    ("ukpga", 1925, 20),   # Law of Property Act 1925
    ("ukpga", 1954, 56),   # Landlord and Tenant Act 1954
    # Tort
    ("ukpga", 1957, 31),   # Occupiers' Liability Act 1957
    # Partnership
    ("ukpga", 2000, 12),   # Limited Liability Partnerships Act 2000
    # IP
    ("ukpga", 1988, 48),   # Copyright, Designs and Patents Act 1988
    # Health & Safety
    ("ukpga", 1974, 37),   # Health and Safety at Work Act 1974
    # Interpretation
    ("ukpga", 1978, 30),   # Interpretation Act 1978
    # Modern Slavery
    ("ukpga", 2015, 30),   # Modern Slavery Act 2015
    # Environment
    ("ukpga", 2021, 30),   # Environment Act 2021
    # Retained EU Law
    ("ukpga", 2018, 16),   # European Union (Withdrawal) Act 2018
    ("ukpga", 2023, 28),   # Retained EU Law Act 2023
]


def fetch_uk_act_xml(act_type: str, year: int, number: int) -> str | None:
    """Download UK act text via legislation.gov.uk API."""
    url = f"https://www.legislation.gov.uk/{act_type}/{year}/{number}/data.xml"
    resp = http_get(url, delay=0.8, timeout=120)
    if resp is None:
        return None
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        log.warning("XML parse error for %s/%d/%d", act_type, year, number)
        return None

    # Extract all text from the legislation XML
    # The CLML format uses various elements for text content
    text_parts = get_all_text(root)
    text = "\n\n".join(text_parts)

    if estimate_tokens(text) < CHUNK_MIN_TOKENS:
        return None
    return clean_legal_text(text)


def uk_feed_acts(year: int, max_results: int = 50) -> list[tuple[str, int, int]]:
    """Get list of UK acts from legislation.gov.uk feed for a given year."""
    url = f"https://www.legislation.gov.uk/ukpga/{year}/data.feed"
    resp = http_get(url, delay=1.0, timeout=60)
    if resp is None:
        return []
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return []

    acts = []
    for entry in root.iter(f"{{{ATOM_NS}}}entry"):
        id_elem = entry.find(f"{{{ATOM_NS}}}id")
        if id_elem is not None and id_elem.text:
            # URI like: http://www.legislation.gov.uk/id/ukpga/2023/29
            m = re.search(r"/id/(ukpga)/(\d{4})/(\d+)", id_elem.text)
            if m:
                acts.append((m.group(1), int(m.group(2)), int(m.group(3))))
    return acts[:max_results]


def download_uk_legislation(out_dir: Path, progress: ProgressTracker,
                            target_tokens: int):
    """Download UK legislation from legislation.gov.uk."""
    source = "uk_legislation"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("── legislation.gov.uk: downloading UK Acts ──")

    # Combine curated list with feed-discovered acts
    acts = list(UK_ACTS)
    log.info("Searching legislation.gov.uk feeds for additional acts...")
    for year in range(2018, 2026):
        feed_acts = uk_feed_acts(year, max_results=30)
        acts.extend(feed_acts)
    # Deduplicate
    acts = list(dict.fromkeys(acts))
    log.info("Found %d unique UK acts", len(acts))

    collected_tokens = progress.total_tokens(source)
    pbar = tqdm(acts, desc="UK Legislation", unit="act")

    for act_type, year, number in pbar:
        if collected_tokens >= target_tokens:
            break
        doc_id = f"{act_type}_{year}_{number}"
        if progress.is_done(source, doc_id):
            continue
        text = fetch_uk_act_xml(act_type, year, number)
        if text is None:
            progress.mark_done(source, doc_id, 0)
            continue
        tok = estimate_tokens(text)
        fpath = out_dir / f"{doc_id}.txt"
        fpath.write_text(text, encoding="utf-8")
        progress.mark_done(source, doc_id, tok)
        collected_tokens += tok
        pbar.set_postfix(tokens=f"{collected_tokens:,}")

    pbar.close()
    log.info("UK Legislation: %d acts, ~%s tokens",
             progress.count(source), f"{collected_tokens:,}")


# ════════════════════════════════════════════════════════════════════════
# SOURCE 5: eCFR  (US Code of Federal Regulations)
# ════════════════════════════════════════════════════════════════════════

ECFR_TITLES_URL = "https://www.ecfr.gov/api/versioner/v1/titles.json"

# Selected CFR titles most relevant for legal practice
ECFR_PRIORITY_TITLES = [
    12,  # Banks and Banking
    15,  # Commerce and Foreign Trade
    17,  # Commodity and Securities Exchanges
    26,  # Internal Revenue
    28,  # Judicial Administration
    29,  # Labor
    31,  # Money and Finance: Treasury
    40,  # Protection of Environment
    47,  # Telecommunication
    49,  # Transportation
]


def ecfr_get_date() -> str:
    """Get a recent date string for eCFR API."""
    # Use a date a few days in the past to ensure availability
    d = date.today()
    return d.strftime("%Y-%m-%d")


def fetch_ecfr_title(title_number: int) -> str | None:
    """Download full XML for one eCFR title and extract text."""
    api_date = ecfr_get_date()
    url = (
        f"https://www.ecfr.gov/api/versioner/v1/full/"
        f"{api_date}/title-{title_number}.xml"
    )
    log.info("Fetching eCFR title %d (this may take a while)...",
             title_number)
    resp = http_get(url, delay=2.0, timeout=300)  # large files, long timeout
    if resp is None:
        # Fallback: try govinfo bulk data
        url2 = (
            f"https://www.govinfo.gov/bulkdata/ECFR/"
            f"title-{title_number}/ECFR-title{title_number}.xml"
        )
        resp = http_get(url2, delay=2.0, timeout=300)
        if resp is None:
            return None

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        log.warning("XML parse error for eCFR title %d", title_number)
        return None

    # eCFR XML uses DIV1-DIV8 hierarchy for chapters/parts/sections
    # Extract text from all relevant elements
    text_parts = get_all_text(root)
    text = "\n\n".join(text_parts)

    if estimate_tokens(text) < CHUNK_MIN_TOKENS:
        return None
    return clean_legal_text(text)


def download_ecfr(out_dir: Path, progress: ProgressTracker,
                  target_tokens: int):
    """Download US federal regulations from eCFR."""
    source = "ecfr"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("── eCFR: downloading US Federal Regulations ──")

    collected_tokens = progress.total_tokens(source)
    pbar = tqdm(ECFR_PRIORITY_TITLES, desc="eCFR", unit="title")

    for title_num in pbar:
        if collected_tokens >= target_tokens:
            break
        doc_id = f"title_{title_num}"
        if progress.is_done(source, doc_id):
            continue
        text = fetch_ecfr_title(title_num)
        if text is None:
            progress.mark_done(source, doc_id, 0)
            continue
        tok = estimate_tokens(text)
        fpath = out_dir / f"ecfr_{doc_id}.txt"
        fpath.write_text(text, encoding="utf-8")
        progress.mark_done(source, doc_id, tok)
        collected_tokens += tok
        pbar.set_postfix(tokens=f"{collected_tokens:,}")

    pbar.close()
    log.info("eCFR: %d titles, ~%s tokens",
             progress.count(source), f"{collected_tokens:,}")


# ════════════════════════════════════════════════════════════════════════
# CHUNKING & BALANCING
# ════════════════════════════════════════════════════════════════════════

LANG_SOURCE_MAP = {
    "nl": ["rechtspraak", "wetten_nl"],
    "uk": ["uk_legislation"],
    "us": ["ecfr"],
}


def collect_raw_texts(base_dir: Path) -> dict[str, list[dict]]:
    """Read all downloaded text files and assign language labels."""
    texts = {"nl": [], "uk": [], "us": []}

    # Map subdirectory names to languages
    dir_lang_map = {
        "rechtspraak": "nl",
        "wetten_nl": "nl",
        "uk_legislation": "uk",
        "ecfr": "us",
    }

    for subdir, lang in dir_lang_map.items():
        dpath = base_dir / subdir
        if not dpath.exists():
            continue
        for fpath in sorted(dpath.glob("*.txt")):
            text = fpath.read_text(encoding="utf-8")
            if text.strip():
                texts[lang].append({
                    "text": text,
                    "source": subdir,
                    "file": fpath.name,
                })

    return texts


def chunk_and_balance(base_dir: Path, output_path: Path,
                      target_tokens: int):
    """Chunk all texts and balance to target ratios, write JSONL."""
    log.info("── Chunking and balancing ──")

    raw = collect_raw_texts(base_dir)

    # Chunk each language
    all_chunks: dict[str, list[dict]] = {"nl": [], "uk": [], "us": []}
    for lang, docs in raw.items():
        log.info("Chunking %s: %d documents", lang.upper(), len(docs))
        for doc in docs:
            chunks = chunk_text(doc["text"])
            for chunk in chunks:
                all_chunks[lang].append({
                    "text": chunk,
                    "meta": {
                        "lang": lang,
                        "source": doc["source"],
                        "tokens": estimate_tokens(chunk),
                    },
                })

    # Count tokens per language
    token_counts = {}
    for lang, chunks in all_chunks.items():
        token_counts[lang] = sum(c["meta"]["tokens"] for c in chunks)
        log.info("  %s: %d chunks, ~%s tokens",
                 lang.upper(), len(chunks), f"{token_counts[lang]:,}")

    # Balance to target ratios
    ratios = {"nl": DUTCH_RATIO, "uk": UK_RATIO, "us": US_RATIO}
    target_per_lang = {lang: int(target_tokens * ratio)
                       for lang, ratio in ratios.items()}

    final_chunks = []
    for lang in ["nl", "uk", "us"]:
        available = all_chunks[lang]
        target_tok = target_per_lang[lang]
        running = 0
        for chunk in available:
            if running >= target_tok:
                break
            final_chunks.append(chunk)
            running += chunk["meta"]["tokens"]
        log.info("  %s selected: ~%s tokens (target: %s)",
                 lang.upper(), f"{running:,}", f"{target_tok:,}")

    # Shuffle deterministically
    import random
    random.seed(42)
    random.shuffle(final_chunks)

    # Write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in final_chunks:
            record = {"text": chunk["text"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Also write a version with metadata for reference
    meta_path = output_path.with_suffix(".meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for chunk in final_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    total_tok = sum(c["meta"]["tokens"] for c in final_chunks)
    log.info("Final dataset: %d chunks, ~%s tokens → %s",
             len(final_chunks), f"{total_tok:,}", output_path)
    log.info("Metadata file: %s", meta_path)

    return final_chunks


# ════════════════════════════════════════════════════════════════════════
# STATISTICS & REPORTING
# ════════════════════════════════════════════════════════════════════════

def print_stats(chunks: list[dict]):
    """Print a summary of the final dataset."""
    stats = defaultdict(lambda: {"chunks": 0, "tokens": 0})
    for c in chunks:
        lang = c["meta"]["lang"]
        source = c["meta"]["source"]
        key = f"{lang}/{source}"
        stats[key]["chunks"] += 1
        stats[key]["tokens"] += c["meta"]["tokens"]
        stats[lang]["chunks"] += 1
        stats[lang]["tokens"] += c["meta"]["tokens"]

    total_tok = sum(c["meta"]["tokens"] for c in chunks)

    print("\n" + "=" * 65)
    print("  LEGAL CPT DATASET — FINAL STATISTICS")
    print("=" * 65)

    for lang in ["nl", "uk", "us"]:
        s = stats[lang]
        pct = (s["tokens"] / total_tok * 100) if total_tok else 0
        print(f"\n  {lang.upper()} — {s['tokens']:>12,} tokens "
              f"({pct:5.1f}%)  [{s['chunks']:,} chunks]")
        for key, val in sorted(stats.items()):
            if key.startswith(f"{lang}/"):
                src_name = key.split("/", 1)[1]
                print(f"      {src_name:<20s} {val['tokens']:>10,} tokens  "
                      f"[{val['chunks']:,} chunks]")

    print(f"\n  TOTAL — {total_tok:>12,} tokens  [{len(chunks):,} chunks]")
    print("=" * 65 + "\n")


# ════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════

def run_pipeline(output_dir: str, target_tokens: int, test: bool = False):
    """Run the complete CPT data pipeline."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    progress = ProgressTracker(base / "progress.json")

    # Calculate per-source targets (with headroom for balancing)
    # We download more than needed, then balance in the chunking step
    headroom = 1.3  # download 30% extra to ensure enough for balancing
    nl_target = int(target_tokens * DUTCH_RATIO * headroom)
    uk_target = int(target_tokens * UK_RATIO * headroom)
    us_target = int(target_tokens * US_RATIO * headroom)

    if test:
        log.info("TEST MODE: downloading minimal samples")
        nl_target = 50_000
        uk_target = 25_000
        us_target = 25_000

    # Distribute Dutch tokens across sources
    # Rechtspraak gets the lion's share (most volume available: 800k+ decisions)
    rechtspraak_target = int(nl_target * 0.70)
    wetten_target = int(nl_target * 0.30)

    # ── Phase 1: Download ──
    log.info("=" * 60)
    log.info("  PHASE 1: DOWNLOAD")
    log.info("=" * 60)
    log.info("Target: %s total tokens (NL=%s UK=%s US=%s)",
             f"{target_tokens:,}",
             f"{int(target_tokens * DUTCH_RATIO):,}",
             f"{int(target_tokens * UK_RATIO):,}",
             f"{int(target_tokens * US_RATIO):,}")

    # Dutch sources
    download_rechtspraak(base / "rechtspraak", progress, rechtspraak_target,
                         years=range(2024, 2017, -1) if not test
                         else range(2024, 2023, -1))

    download_dutch_legislation(base / "wetten_nl", progress, wetten_target)

    # UK source
    download_uk_legislation(base / "uk_legislation", progress, uk_target)

    # US source
    download_ecfr(base / "ecfr", progress, us_target)

    # ── Phase 2: Chunk & Balance ──
    log.info("")
    log.info("=" * 60)
    log.info("  PHASE 2: CHUNK & BALANCE")
    log.info("=" * 60)

    final_jsonl = base / "legal_cpt_train.jsonl"
    chunks = chunk_and_balance(base, final_jsonl, target_tokens)

    # ── Phase 3: Report ──
    print_stats(chunks)

    # License summary
    print("LICENSE SUMMARY")
    print("-" * 65)
    print("  Rechtspraak.nl      Public domain (Dutch gov works)")
    print("  wetten.overheid.nl   CC-0 (Basiswettenbestand)")
    print("  legislation.gov.uk   Open Government Licence v3.0")
    print("  eCFR                 Public domain (US gov works)")
    print("-" * 65)
    print(f"\nOutput: {final_jsonl.absolute()}")
    print(f"Ready for CPT training! Each line is: " + '{"text": "..."}')
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Legal CPT Data Pipeline — download & prepare legal text "
                    "for continued pre-training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python legal_cpt_pipeline.py                      # 15M tokens, default
  python legal_cpt_pipeline.py --target-tokens 20000000
  python legal_cpt_pipeline.py --test               # quick smoke test
  python legal_cpt_pipeline.py --output-dir /data/legal_cpt
        """,
    )
    parser.add_argument(
        "--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS,
        help=f"Total target tokens (default: {DEFAULT_TARGET_TOKENS:,})",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./legal_cpt_data",
        help="Output directory (default: ./legal_cpt_data)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Quick test mode: download minimal samples to verify APIs work",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print(r"""
    ╔═══════════════════════════════════════════════════════╗
    ║        Legal CPT Data Pipeline v1.1                  ║
    ║  Dutch 50% · UK English 25% · US English 25%         ║
    ║  4 sources: public domain / CC-0 / OGL               ║
    ╚═══════════════════════════════════════════════════════╝
    """)

    run_pipeline(
        output_dir=args.output_dir,
        target_tokens=args.target_tokens,
        test=args.test,
    )


if __name__ == "__main__":
    main()
