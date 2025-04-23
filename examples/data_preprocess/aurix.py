"""
Preprocess the VoltAI dataset to parquet format
"""

import argparse
import os
import json
import re

from datasets import Dataset
from prompts import SYSTEM_PROMPT


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
	qa_data = json.load(open(os.path.join(args.data_dir, "gemini_qa_data.json")))
	qc_data = json.load(open(os.path.join(args.data_dir, "qc_data.json")))
	all_data = json.load(open(os.path.join(args.data_dir, "all_data.json")))
	retrieval_ids = json.load(open(os.path.join(args.data_dir, "all_data_similar.json")))	

	filtered_data = []
	for sample in qa_data:
		if len(qc_data.get(sample["question"], [])) == 0:
			continue
		sample["ref_ids"] = extract_ref_ids(sample["gemini_answer"])
		if len(sample["ref_ids"]) == 0:
			continue
		filtered_data.append(sample)

	print("Size of original data:", len(qa_data))
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
