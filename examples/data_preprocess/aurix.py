"""
Preprocess the VoltAI dataset to parquet format
"""

import argparse
import os
import json
import re
import random

from datasets import Dataset
from prompts import SYSTEM_PROMPT

def extract_ref_ids(answer):
	pattern = r'<ref id=\"([^"]+)\"></ref>'
	ref_ids = re.findall(pattern, answer)
	if len(ref_ids) == 0:
		pattern = r'<ref id=\'([^"]+)\'></ref>'
		ref_ids = re.findall(pattern, answer)
	if len(ref_ids) == 0:
		pattern = r'<ref id=([^"\'>]+)></ref>'
		ref_ids = re.findall(pattern, answer)
	print(f"found {len(ref_ids)} references")
	return ref_ids


def make_map_fn(split, negative_chunks, qa_model):
	def process_fn(example, idx):
		if qa_model == "gemini":
			answer_field = "gemini_answer"
		elif qa_model == "openai":
			answer_field = "answer"
		else:
			raise ValueError(f"Invalid qa_model: {qa_model}")
		retrieved_chunk_ids = [s for s in qc_data.get(example["question"], []) if s in all_data][:negative_chunks]
		source_chunk_ids = [
			s for s in all_data_similar[example["original_chunk_id"]] 
			if s in all_data
		] + [example["original_chunk_id"]] if example["original_chunk_id"] in all_data else []
		chunk_ids = source_chunk_ids + retrieved_chunk_ids
		random.shuffle(chunk_ids)
		chunk_ids = list(set(chunk_ids))
		print(f"Context has {len(chunk_ids)} chunks")
		chunk_id_mapping = {
			chunk_id: i for i, chunk_id in enumerate(chunk_ids)
		}
		data = {
			"data_source": data_source,                
			"prompt": [
					{"role": "system", "content": SYSTEM_PROMPT},
					{"role": "user",   "content": f"<question>{example['question']}</question>" + "\n".join(
							[
								f"<document id={chunk_id_mapping[chunk_id]}>{all_data[chunk_id]}</document>" 
								for chunk_id in chunk_ids if chunk_id in chunk_id_mapping
							]
						),
					}
			],
			"ability": "math",
			"reward_model": {"style": "rule", "ground_truth": None},
			"extra_info": {
				"split": split,
				"index": idx,
				"answer": example[answer_field].strip(),
				"question": example["question"],
				"ref_ids": [str(chunk_id_mapping[chunk_id]) for chunk_id in example["ref_chunk_ids"] if chunk_id in chunk_id_mapping],
				"ref_chunk_ids": example["ref_chunk_ids"]
			}
		}
		return data

	return process_fn

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--local_dir", default="data/")
	parser.add_argument("--data_dir", default="data/")
	parser.add_argument("--negative_chunks", default=95, type=int)
	parser.add_argument("--qa_model", default="gemini")
	args = parser.parse_args()

	data_source = "voltai/aurix"
	qa_data = json.load(open(os.path.join(args.data_dir, f"{args.qa_model}_qa_data.json")))
	qc_data = json.load(open(os.path.join(args.data_dir, "qc_data.json")))
	all_data = json.load(open(os.path.join(args.data_dir, "all_data.json")))
	all_data_similar = json.load(open(os.path.join(args.data_dir, "all_data_similar.json")))	

	filtered_data = []
	for sample in qa_data:
		if len(qc_data.get(sample["question"], [])) == 0:
			continue
		sample["ref_chunk_ids"] = extract_ref_ids(sample["answer"])
		if len(sample["ref_chunk_ids"]) == 0:
			continue
		filtered_data.append(sample)

	print("Size of original data:", len(qa_data))
	qa_data = filtered_data
	print("Size of filtered data:", len(qa_data))
	
	# Convert the list of dicts into a HuggingFace Dataset.
	dataset = Dataset.from_list(qa_data)
	dataset = dataset.map(function=make_map_fn("train", args.negative_chunks, args.qa_model), with_indices=True)

	# Split the dataset into train and test sets (95% train, 5% test)
	dataset = dataset.train_test_split(test_size=0.05, train_size=0.95, seed=407, shuffle=True)
	train_dataset = dataset["train"]
	test_dataset = dataset["test"]

	local_dir = args.local_dir
	train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
	test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
