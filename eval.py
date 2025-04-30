from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import asyncio
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

os.makedirs("evals", exist_ok=True)

@dataclass
class CorrectnessScore:
	analysis: str
	score: float

openai_scorer = ChatOpenAI(model="o3-mini", temperature=1.0).with_structured_output(CorrectnessScore)
async def answer_correctness(answer, gt_answer):
	try:
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
		response = await pipeline.ainvoke({"gt_answer": gt_answer, "answer": answer})
		return float(response.get("score", 0.0))
	except Exception as e:
		print("Error in answer_correctness:", e)
		return 0.0

eval_data = json.load(open("data/Neal-Simplified.json"))
eval_chunks = json.load(open("data/Neal-Simplified_chunks.json"))
all_data = json.load(open("data/all_data.json"))

# Initialize tokenizer and model
model_path = "Qwen/Qwen2.5-32B-Instruct"
checkpoint_dir = "/root/verl-llm-critique/checkpoints/verl_grpo_aurix/qwen2_7b_qa_reasoning/hf_80"
tokenizer = AutoTokenizer.from_pretrained(model_path)

llm = LLM(
	model=model_path,
	tokenizer=model_path,
    tensor_parallel_size=8,
	gpu_memory_utilization=0.9,
	enforce_eager=True
)

sampling_params = SamplingParams(
	temperature=0.2,
	max_tokens=130000,
)

async def evalute_correctness(all_scores, batch):
	questions = [sample["input"]["query"] for sample in batch]
	gt_answers = [sample["expected"]["groundtruth_answer"] for sample in batch]
	document_ids_list = [eval_chunks[question] for question in questions]

	contexts = [f"<question>{question}</question>" + "\n".join([f"<document id={doc_id}>{all_data[doc_id]}</document>" for doc_id in document_ids]) for question, document_ids in zip(questions, document_ids_list)]
	tokens = [tokenizer.encode(context, truncation=True, max_length=120000) for context in contexts]
	contexts = [tokenizer.decode(tokens) for tokens in tokens]

	# Format input with system prompt and question
	conversations = [[
		{"role": "system", "content": SYSTEM_PROMPT},
		{"role": "user", "content": context}
	] for context in contexts]

	# Generate the assistant’s reply:
	chat_outputs = llm.chat(conversations, sampling_params=sampling_params)
	answers = [chat_output.outputs[0].text for chat_output in chat_outputs]

	# Extract the answer part
	tasks = []
	for answer, gt_answer in zip(answers, gt_answers):
		answer_pattern = r"<answer>\s*(.*?)\s*</answer>"
		answer_match = re.search(answer_pattern, answer, re.DOTALL)
		if answer_match:
			answer = answer_match.group(1)
		else:
			answer_pattern = r"<answer>(.*?)$"
			answer_match = re.search(answer_pattern, answer, re.DOTALL)
			if answer_match:
				answer = answer_match.group(1)

		tasks.append(asyncio.create_task(answer_correctness(answer, gt_answer)))
	
	scores = await asyncio.gather(*tasks)
	json.dump(all_scores + scores, open("evals/eval_grpo.json", "w"), indent=4)
		
	return [
		{
			"question": question,
			"generated_response": answer,
			"ground_truth": gt_answer,
			"correctness": score
		} for question, answer, gt_answer, score in zip(questions, answers, gt_answers, scores)
	]

batch_size = 100
all_scores = []
for i in range(0, len(eval_data), batch_size):
	batch = eval_data[i:i+batch_size]
	all_scores += asyncio.run(evalute_correctness(all_scores, batch))

print(f"Average score for Qwen2.5-32B-Instruct: {sum([s['correctness'] for s in all_scores]) / len(all_scores)}")