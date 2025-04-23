"""
Preprocess the VoltAI dataset to parquet format
"""

import argparse
import os
import json
import re

from datasets import Dataset

SYSTEM_PROMPT = """You are a technical assistant specializing in electronics, semiconductors, and microcontrollers. Provide accurate and reliable information to engineers.

## Task Overview

You will be provided with:
1. **A Technical Question:** A specific inquiry related to electronics, semiconductors, or microcontrollers enclosed in `<question>` and `</question>` tags.
2. **Source Documents:** A set of datasheet excerpts and related documents that may contain relevant information to answer the question that are enclosed in `<document>` and `</document>` tags with a unique id field.

## Your Responsibilities

### 1. Step-by-Step Reasoning
Reason through the documents to find the answer to the question.
Start your reasoning with a `<think>` tag:

- **Ask good questions:**
  - Breakdown the question into smaller, manageable questions.
  - Questions should be clear and directly related to the user's question.
- **Reason through the documents:**
  - Reason through the provided documents and then briefly answer each question.
  - Only use the documents that are relevant to the question, there might be some documents that are not relevant to the question.
  - Look for inconsistencies, contradictions, and any other issues with the information provided.
- **Put it all together:**
  - When you have enough information to answer the user's question, finish your reasoning with a `</think>` tag.

### 3. Answer Formulation
Wrap your final response in `<answer>` tags:

- **Structure:**
  - **One-Sentence Summary:** Start with a one-sentence answer enclosed in `<mark>` and `</mark>` tags.
  - **Detailed Explanation:** Follow with comprehensive explanations covering all main points that you discovered while reasoning through the documents.
- **Content Requirements:**
  - **Accuracy:** Base answers solely on the provided context and documents. Do not fabricate information. If the information is not provided in the documents, mention that in your answer.
  - **Comprehensiveness:** Ensure no critical information is omitted.
  - **Citations:** Use inline references in the format `<ref id="document_id"></ref>`.
Finally, finish your answer with a `</answer>` tag."""

qa_data = json.load(open(os.path.join(args.data_dir, "gemini_qa_data.json")))
qc_data = json.load(open(os.path.join(args.data_dir, "qc_data.json")))
all_data = json.load(open(os.path.join(args.data_dir, "all_data.json")))
retrieval_ids = json.load(open(os.path.join(args.data_dir, "all_data_similar.json")))


def extract_ref_ids(example):
	pattern = r'<ref id=\"([^"]+)\"></ref>'
	refs = re.findall(pattern, example['gemini_answer'])
	if len(refs) == 0:
		pattern = r'<ref id=\'([^"]+)\'></ref>'
		refs = re.findall(pattern, example['gemini_answer'])
	if len(refs) == 0:
		pattern = r'<ref id=([^"\'>]+)></ref>'
		refs = re.findall(pattern, example['gemini_answer'])
	example['ref_ids'] = list(set(refs))
	print(f"found {len(example['ref_ids'])} references")
	return example


def make_map_fn(split):
	def process_fn(example, idx):
		data = {
			"data_source": data_source,                
			"prompt": [
					{"role": "system", "content": SYSTEM_PROMPT},
					{"role": "user",   "content": f"<question>{example['question']}</question>" + "\n".join(
							[
								f"<document id={chunk_id}>{all_data[chunk_id]}</document>" 
								for chunk_id in qc_data.get(example["question"], [])
								if chunk_id in all_data
							]
						),
					}
			],
			"ability": "math",
			"reward_model": {"style": "rule", "ground_truth": None},
			"extra_info": {
				"split": split,
				"index": idx,
				"answer": example["gemini_answer"].strip(),
				"question": example["question"],
				"ref_ids": example.get(
					"ref_ids", []
				),  # reference IDs, defaulting to empty list if not present
				"retrieved_ids": retrieval_ids[example["original_chunk_id"]],					
			}
		}
		return data

	return process_fn

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--local_dir", default="data/")
	parser.add_argument("--data_dir", default="data/")
	args = parser.parse_args()

	data_source = "voltai/aurix"

	filtered_data = []
	for sample in qa_data:
		if len(qc_data.get(sample["question"], [])) == 0:
			continue
		sample = extract_ref_ids(sample)
		if len(sample["ref_ids"]) == 0:
			continue
		filtered_data.append(sample)

	qa_data = filtered_data
	print("Size of filtered data:", len(qa_data))
	
	# Convert the list of dicts into a HuggingFace Dataset.
	dataset = Dataset.from_list(qa_data)
	dataset = dataset.map(function=make_map_fn("train"), with_indices=True)

	# Split the dataset into train and test sets (95% train, 5% test)
	dataset = dataset.train_test_split(test_size=0.05, train_size=0.95, seed=407, shuffle=True)
	train_dataset = dataset["train"]
	test_dataset = dataset["test"]

	local_dir = args.local_dir
	train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
	test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
