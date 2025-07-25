import os
import json
import fitz
from pprint import pprint

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
        text += "\n"  # ensure separation between pages
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])  # remove excessive newlines and blank lines
    return text

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
    data_dir = "../data"
    file_paths = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith('.pdf')
    ]

    all_pipeline_prompts = []
    for pdf_file in file_paths:
        # ✅ Step 1: Extract Text from PDF files
        text = extract_text_from_pdf(pdf_file)
        # print(f"🗂️ --- {pdf_file} ---")
        # print(text[:300])

        # ✅ Step 2: Turn chunked text into pipeline input formatted dict
        chunks = slide_and_chunk_text(text=text, window_size=3000, stride=500)
        prompt_chunks = build_prompt_chunks(chunks=chunks)
        all_pipeline_prompts.extend(prompt_chunks)

    with open("../data/mmlu_prompt_input.jsonl", "w") as f:
        for item in all_pipeline_prompts:
            f.write(json.dumps(item, ensure_ascii=False) +'\n')
            
    print(f"✅ 총 {len(all_pipeline_prompts)}개의 context 블록이 생성되었습니다.")
    pprint(all_pipeline_prompts[:5])
