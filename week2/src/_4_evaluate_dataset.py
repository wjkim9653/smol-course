import argparse
import json
import re
from lexical_diversity import lex_div as ld
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from util import *

def main(input_path):
    jsonl_path = input_path

    num_samples = 0
    input_lengths = []
    output_lengths = []
    outputs = []

    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc="Reading"):
            qa = json.loads(line.strip())
            '''a line from input jsonl file
            {
                "question": "What is the number of significant figures in the measurement 20–21?",
                "choices": ["1", "2", "3", "4"],
                "answer": "B"
            }
            '''
            # if (qa) and all(k in qa for k in ("question", "choices", "answer")) and (type(qa["answer"]) is str) and (type(qa["choices"] is list) and len(qa["choices"])==4):
            if (qa) and all(k in qa for k in ("question", "choices", "answer")) and (type(qa["answer"]) is str) and (type(qa["choices"] is list) and len(qa["choices"])>1):
                num_samples += 1

                # question 길이 (토큰 수)
                q_tokens = word_tokenize(qa["question"])
                input_lengths.append(len(q_tokens))

                # choices + answer 길이
                # output_text = " ".join(qa["choices"]) + " " + qa["answer"]
                output_text = " ".join(str(choice) for choice in qa["choices"]) + " " + qa["answer"]
                o_tokens = word_tokenize(output_text)
                output_lengths.append(len(o_tokens))

                outputs.extend(q_tokens + o_tokens)
            else:
                print("⚠️ Invalid or unparsable qa_item:", qa)

    avg_input_len = sum(input_lengths) / num_samples if num_samples else 0
    avg_output_len = sum(output_lengths) / num_samples if num_samples else 0

    mattr = ld.mattr(outputs, window_length=50) if outputs else 0
    hdd = ld.hdd(outputs) if outputs else 0
    mtld = ld.mtld(outputs) if outputs else 0

    print("\n📊 Evaluation Results")
    print(f"# Samples             : {num_samples}")
    print(f"Avg Input Length     : {avg_input_len:.2f} tokens")
    print(f"Avg Output Length    : {avg_output_len:.2f} tokens")
    print(f"Lexical Diversity (MATTR): {mattr:.4f}")
    print(f"Lexical Diversity (HD-D) : {hdd:.4f}")
    print(f"Lexical Diversity (MTLD) : {mtld:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evaluation for Generated Synthetic Dataset")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file dir for pipeline output(generated QAs)")
    args = parser.parse_args()
    main(args.input)