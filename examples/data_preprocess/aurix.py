"""
Preprocess the VoltAI dataset to parquet format
"""

import argparse
import os
import json

from datasets import Dataset

reasoning_start = "<think>"
reasoning_end   = "</think>"
solution_start = "<answer>"
solution_end = "</answer>"

SYSTEM_PROMPT = f"""You are an expert hardware and electronics engineer that answers questions by looking at the retrieved information from documents.
Multiple documents analysis is provided to you with their corresponding ids.
Think about the question and the retrieved information.
Provide a detailed analysis of the retrieved information and how it relates to the question and place it between {reasoning_start} and {reasoning_end}.
Then, provide your final answer between {solution_start}{solution_end}.
When composing your final answer, you must cite the sources you used by adding a citation in the following format:
<ref id="cited_document_id"></ref>
The citation should be a valid document id from the retrieved documents.
It should follow the sentence or paragraph that you are citing, and should only be in the answer part."""

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--local_dir", default="data/aurix")
	parser.add_argument("--data_dir", default="data/aurix")
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
		filtered_data.append(sample)

	qa_data = filtered_data
	print("Size of filtered data:", len(qa_data))

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
	
	# Convert the list of dicts into a HuggingFace Dataset.
	dataset = Dataset.from_list(qa_data)
	dataset = dataset.map(function=make_map_fn("train"), with_indices=True)

	# Split the dataset into train and test sets (95% train, 5% test)
	dataset = dataset.train_test_split(test_size=26, train_size=30, seed=407, shuffle=True)
	train_dataset = dataset["train"]
	test_dataset = dataset["test"]

	local_dir = args.local_dir
	train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
	test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
