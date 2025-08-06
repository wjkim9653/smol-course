import os
import re
from pathlib import Path
from collections import Counter
from tabulate import tabulate
from pprint import pprint
import json
import fitz
import pdfplumber
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract
import easyocr
import tempfile
from tqdm import tqdm
import logging
# Basic configuration for logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to show info, warning, error, etc.
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("pdfminer").setLevel(logging.ERROR)


def extract_text_from_pdf(pdf_path, method="pymupdf", use_tqdm=False):
    """
    Conduct a PDF -> Text Extraction based on given pdf file, using specified library
    """
    method = method.lower()
    if method == "pymupdf":
        doc = fitz.open(pdf_path)
        if use_tqdm:
            text = "\n".join([page.get_text() for page in tqdm(doc, desc="🔥 PyMuPDF Extracting...")])  # '\n' -> ensures separation between pages
        else:
            text = "\n".join([page.get_text() for page in doc])  # '\n' -> ensures separation between pages
    elif method == "pdfminer":
        logging.info("🔥 PDFMiner Extracting... (this method doesn't support progress bar)") if use_tqdm else None
        text = pdfminer_extract(pdf_path)
    elif method == "pypdf2":
        reader = PdfReader(pdf_path)
        if use_tqdm:
            text = "\n".join([page.extract_text() for page in tqdm(reader.pages, desc="PyPDF2 Extracting...")])
        else:
            text = "\n".join([page.extract_text() for page in reader.pages])
    elif method == "pdfplumber":
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            if use_tqdm:
                for page in tqdm(pdf.pages, desc="🔥 PDFPlumber Extracting..."):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            else:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
    elif method == "easyocr":
        reader = easyocr.Reader(['en'], gpu=True)  # gpu -> for faster ocr
        text = ""
        with fitz.open(pdf_path) as doc:
            if use_tqdm:
                for page_num in tqdm(range(len(doc)), desc="🔥 EasyOCR Extracting..."):
                    pix = doc.load_page(page_num).get_pixmap(dpi=300)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        pix.save(tmp.name)
                        result = reader.readtext(tmp.name, detail=0)
                        text += "\n".join(result) + "\n"
                        os.unlink(tmp.name)
            else:
                for page_num in range(len(doc)):
                    pix = doc.load_page(page_num).get_pixmap(dpi=300)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        pix.save(tmp.name)
                        result = reader.readtext(tmp.name, detail=0)
                        text += "\n".join(result) + "\n"
                        os.unlink(tmp.name)
            
    else:
        raise ValueError(f"Unsupported library method: {method}")
    
    # Some simple post-processing...
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])  # remove excessive newlines and blank lines
    return text

def analyze_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    lines = text.splitlines()
    num_lines = len(lines)
    empty_lines = sum(1 for line in lines if not line.strip())
    total_length = sum(len(line) for line in lines)
    avg_line_len = total_length / num_lines if num_lines > 0 else 0

    words = re.findall(r'\b\w+\b', text.lower())  # word counting based on blanks
    word_count = len(words)
    unique_words = len(set(words))
    unique_ratio = unique_words / word_count if word_count > 0 else 0

    non_ascii_chars = sum(1 for c in text if ord(c) > 127)
    non_ascii_ratio = non_ascii_chars / len(text) if len(text) > 0 else 0

    return {
        "lines": num_lines,
        "words": word_count,
        "avg_line_len": round(avg_line_len, 2),
        "empty_lines": empty_lines,
        "unique_word_ratio": round(unique_ratio, 4),
        "non_ascii_ratio": round(non_ascii_ratio, 4),
    }

def slide_and_chunk_text(text, window_size=1000, stride=500):
    """
    utilizes sliding window technique w/ stride in order to create chunked text from given source text (so to be used as context when generating synthetic dataset)
    """
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+window_size]
        if len(chunk.strip()) > 200:  # omit too short chunks w/ lots of blank spaces
            chunks.append(chunk.strip())
        i += stride  # overlap w/ set stride
    return chunks

def build_prompt_chunks(chunks):
    """
    takes chunked text (list of str) as input and creates a dict w/ "context" and "instruction" keys
    (so to be used as input to pipeline)
    """
    dataset = []
    for chunk in chunks:
        dataset.append({
            "context": chunk,
            "instruction": "Based on the above content, generate a high school-level multiple-choice chemistry question in the MMLU format. The output should follow this structure: {question: ..., choices: [...], answer: ...}"
        })
    return dataset

if __name__ == "__main__":
    raw_pdf_dir = "../data/raw_pdf"  # directory for original pdf files
    ocr_pdf_dir = "../data/ocr_pdf"  # directory for ocr-processed pdf files
    pdf_sources = {
        "raw_pdf_paths": [
            os.path.join(raw_pdf_dir, f)
            for f in os.listdir(raw_pdf_dir)
            if f.lower().endswith('.pdf')
        ],
        "ocr_pdf_paths": [
            os.path.join(ocr_pdf_dir, f)
            for f in os.listdir(ocr_pdf_dir)
            if f.lower().endswith('.pdf')
        ]
    }

    raw_text_dir = "../data/raw_text"  # directory to store extracted .txt files
    pipeline_prompt_dir = "../data/pipeline_input_prompts"  # dir to store pipeline prompts built from extracted texts
    extraction_methods = ["pymupdf", "pdfminer", "pypdf2", "pdfplumber"] #, "easyocr"]  # pdf -> text extraction libraries to compare
    
    # 🔍 Extract texts from pdfs (w/ various source pdfs and extraction methods)
    for pdf_source in pdf_sources.keys():
        for extraction_method in extraction_methods:
            logging.info(f"{pdf_source} 내 PDF 파일들에 대한 추출 진행 (라이브러리: {extraction_method})")
            all_extracted_text = ""
            all_pipeline_prompts = []
            for pdf_file in pdf_sources.get(pdf_source):
                # ✅ Step 1: Extract Text from PDF files
                logging.info(f"🗂️ Extracting {pdf_file} ...")
                text = extract_text_from_pdf(pdf_file, extraction_method)
                all_extracted_text += text 
                all_extracted_text += "\n"
                # print(text[:300])

                # ✅ Step 2: Turn chunked text into pipeline input formatted dict
                chunks = slide_and_chunk_text(text=text, window_size=3000, stride=500)
                prompt_chunks = build_prompt_chunks(chunks=chunks)
                all_pipeline_prompts.extend(prompt_chunks)

            # Save the extracted text
            textfile_path = os.path.join(raw_text_dir, f"text_from_{pdf_source}_with_{extraction_method}.txt")
            with open(textfile_path, "w") as f:
                f.write(all_extracted_text)
            

            # Save the input prompts built for the pipeline
            pipeline_promptfile_path = os.path.join(pipeline_prompt_dir, f"pipeline_input_prompt_from_{pdf_source}_with_{extraction_method}.jsonl")
            with open(pipeline_promptfile_path, "w") as f:
                for item in all_pipeline_prompts:
                    f.write(json.dumps(item, ensure_ascii=False) +'\n')
            
            logging.info(f"✅ 총 {len(all_pipeline_prompts)}개의 context 블록이 생성되었습니다.")
            # pprint(all_pipeline_prompts[:5])

    # 🧪 Compare the Extracted Files
    analysis_results = []
    for pdf_source in pdf_sources.keys():
        for extraction_method in extraction_methods:
            file_path = os.path.join("../data/raw_text", f"text_from_{pdf_source}_with_{extraction_method}.txt")
            stats = analyze_text_file(file_path=file_path)
            analysis_results.append({
                "PDF Source": pdf_source,
                "Extraction Method": extraction_method,
                "# of Lines": stats["lines"],
                "# of Words": stats["words"],
                "Avg Line Len": stats["avg_line_len"],
                "# of Empty Lines": stats["empty_lines"],
                "Unique Words Ratio": stats["unique_word_ratio"],
                "Non-ASCII Ratio": stats["non_ascii_ratio"]
            })

    # headers = ["PDF Source", "Extraction Method", "# of Lines", "# of Words", "Avg Line Len", "# of Empty Lines", "Unique Words Ratio", "Non ASCII Ratio"]
    print(tabulate(analysis_results, headers="keys", tablefmt="github"))