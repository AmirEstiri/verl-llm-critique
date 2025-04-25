from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from dotenv import load_dotenv

load_dotenv()

import os
from tqdm import tqdm
import glob
import re
import json
from prompts import SYSTEM_PROMPT, EVAL_CORRECTNESS_PROMPT

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI

from dataclasses import dataclass

@dataclass
class CorrectnessScore:
	analysis: str
	score: float

openai_scorer = ChatOpenAI(model="o3-mini", temperature=1.0).with_structured_output(CorrectnessScore)
def answer_correctness(answer, gt_answer):
	# Extract the answer part
	answer_pattern = r"<answer>\s*(.*?)\s*</answer>"
	answer_match = re.search(answer_pattern, answer, re.DOTALL)
	if answer_match:
		answer = answer_match.group(1)

	prompt = ChatPromptTemplate.from_messages([
		SystemMessagePromptTemplate.from_template(EVAL_CORRECTNESS_PROMPT),
		HumanMessagePromptTemplate.from_template("Groundtruth Answer: {gt_answer}\nAnswer: {answer}"),
	])
	pipeline = prompt | openai_scorer
	response = pipeline.invoke({"gt_answer": gt_answer, "answer": answer})
	return response.get("score", 0.0)

eval_data = json.load(open("data/Neal-Simplified.json"))
eval_chunks = json.load(open("data/Neal-Simplified_chunks.json"))
all_data = json.load(open("data/all_data.json"))

# Initialize tokenizer and model
model_path = "Qwen/Qwen2-7B-Instruct"
checkpoint_dir = "/root/verl-llm-critique/checkpoints/verl_grpo_aurix/qwen2_7b_qa_reasoning/hf_80"
tokenizer = AutoTokenizer.from_pretrained(model_path)

llm = LLM(
	model=model_path,
	tokenizer=model_path,
    tensor_parallel_size=4
)

sampling_params = SamplingParams(
	temperature=0.1,
	repetition_penalty=1.05,
	max_tokens=130000,
)


scores = []
for sample in tqdm(eval_data):
	question = sample["input"]["query"]
	gt_answer = sample["expected"]["groundtruth_answer"]
	document_ids = eval_chunks[question]

	context = f"<question>{question}</question>" + "\n".join([f"<document id={doc_id}>{all_data[doc_id]}</document>" for doc_id in document_ids])
	tokens = tokenizer.encode(context, truncation=True, max_length=120000)
	context = tokenizer.decode(tokens)

	# Format input with system prompt and question
	conversation = [
		{"role": "system", "content": SYSTEM_PROMPT},
		{"role": "user", "content": context}
	]

	# Generate the assistant’s reply:
	chat_outputs = llm.chat(conversation, sampling_params=sampling_params)
	answer = chat_outputs[0].outputs[0].text

	# Extract the answer part
	answer_pattern = r"<answer>\s*(.*?)\s*</answer>"
	answer_match = re.search(answer_pattern, answer, re.DOTALL)
	if answer_match:
		answer = answer_match.group(1)
	else:
		answer_pattern = r"<answer>(.*?)$"
		answer_match = re.search(answer_pattern, answer, re.DOTALL)
		if answer_match:
			answer = answer_match.group(1)	    

	score = answer_correctness(answer, gt_answer)
	
	scores.append({
		"question": question,
		"generated_response": answer,
		"ground_truth": gt_answer,
		"correctness": score
	})

	json.dump(scores, open("evals/eval_grpo.json", "w"), indent=4)

print(f"Average score for Qwen2-7B-Instruct: {sum([s['correctness'] for s in scores]) / len(scores)}")