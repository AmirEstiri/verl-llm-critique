from dotenv import load_dotenv

import re
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from dataclasses import dataclass

load_dotenv()

@dataclass
class CorrectnessScore:
	analysis: str
	score: float

reasoning_start = "<think>"
reasoning_end   = "</think>"
solution_start = "<answer>"
solution_end = "</answer>"
highlight_start = "<mark>"
highlight_end = "</mark>"

from prompts import EVAL_CORRECTNESS_PROMPT

def extract_ref_ids(answer):
	pattern = r'<ref id=\"([^"]+)\"></ref>'
	ref_ids = re.findall(pattern, answer)
	if len(ref_ids) == 0:
		pattern = r'<ref id=\'([^"]+)\'></ref>'
		ref_ids = re.findall(pattern, answer)
	if len(ref_ids) == 0:
		pattern = r'<ref id=([^"\'>]+)></ref>'
		ref_ids = re.findall(pattern, answer)
	return ref_ids

def all_reward_functions(tokenizer, data_source, solution_str, ground_truth, extra_info=None):
	return {
		"approximate_formatting": 1.0 * reward_output_approximate_formatting(tokenizer, data_source, solution_str, ground_truth, extra_info),
		"xml_tags_penalty": 1.0 * reward_output_xml_tags_penalty(tokenizer, data_source, solution_str, ground_truth, extra_info),
		"exact_formatting": 2.0 * reward_output_exact_formatting(tokenizer, data_source, solution_str, ground_truth, extra_info),
		"references_formatting": 1.0 * reward_references_formatting(tokenizer, data_source, solution_str, ground_truth, extra_info),
		"references_correctness": 2.0 * reward_references_correctness(tokenizer, data_source, solution_str, ground_truth, extra_info),
		"correctness": 5.0 * reward_answer_correctness_openai(tokenizer, data_source, solution_str, ground_truth, extra_info),
		"length": 1.0 * reward_output_length(tokenizer, data_source, solution_str, ground_truth, extra_info),
	}


def reward_output_xml_tags_penalty(tokenizer, data_source, solution_str, ground_truth, extra_info=None):
	reward = 0.0
	response = solution_str

	allowed_tags = [reasoning_start, reasoning_end, solution_start, solution_end, highlight_start, highlight_end, "document"]

	# Extract all XML tags from the text
	xml_tags = re.findall(r'<[^>]+>', response)
	
	# Count unique tags
	unique_tags = list(set(xml_tags))
	wrong_tags = [tag for tag in unique_tags if tag not in allowed_tags]
	
	if len(wrong_tags) > 0:
		reward = 1.0 - len(wrong_tags) / len(unique_tags)
	else:
		reward = 1.0
	
	return reward
	

def reward_output_approximate_formatting(tokenizer, data_source, solution_str, ground_truth, extra_info=None):
	reward = 0.0
	response = solution_str
	# Count how many keywords are seen - we penalize if too many!
	# If we see 1, then plus some points!
	reward += 0.25 if response.count(reasoning_start) == 1 else 0.0
	reward += 0.25 if response.count(reasoning_end)   == 1 else 0.0
	reward += 0.25 if response.count(solution_start)  == 1 else 0.0
	reward += 0.25 if response.count(solution_end)    == 1 else 0.0
	return reward


def reward_output_exact_formatting(tokenizer, data_source, solution_str, ground_truth, extra_info=None):
	match_format = re.compile(
		rf"{reasoning_start}.*?{reasoning_end}\s*{solution_start}.*?{solution_end}", 
		flags = re.MULTILINE | re.DOTALL
	)
	reward = 0.0
	response = solution_str
	# Match if format is seen exactly!
	if match_format.search(response) is not None: reward = 1.0
	return reward

def reward_references_formatting(tokenizer, data_source, solution_str, ground_truth, extra_info=None):
	reward = 0.0
	response = solution_str

	with open("logs/answers.log", "a") as f:
		f.write(solution_str + "\n\n\n" + "#" * 100 + "\n\n\n")

	# Extract the answer part
	answer_pattern = f"{solution_start}\s*(.*?)\s*{solution_end}"
	answer_match = re.search(answer_pattern, response, re.DOTALL)
	
	# Calculate reward for thinking part
	if answer_match:
		response = answer_match.group(1)
	
	# Extract all reference IDs from the response
	found_refs = extract_ref_ids(response)
	
	# Check if references are properly formatted
	if found_refs:
		# Check if references are in the expected format (alphanumeric IDs)
		valid_ref_pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
		valid_refs = 0
		
		for ref_id in found_refs:
			if valid_ref_pattern.match(ref_id):
				valid_refs += 1
		
		# Calculate score linearly between 0.0 and 2.0 based on proportion of valid refs
		reward = valid_refs / len(found_refs)

	return reward


def reward_references_correctness(tokenizer, data_source, solution_str, ground_truth, extra_info=None):
	reward = 0.0
	response = solution_str
	ref_ids = extract_ref_ids(response)

	# Extract the answer part
	answer_pattern = f"{solution_start}\s*(.*?)\s*{solution_end}"
	answer_match = re.search(answer_pattern, response, re.DOTALL)
	
	# Calculate reward for thinking part
	if answer_match:
		response = answer_match.group(1)
	else:
		response = ""

	# Extract all reference IDs from the response
	pattern = r'<ref id="([^"]+)"></ref>'
	found_refs = re.findall(pattern, response)
	
	# Calculate reward based on reference accuracy
	if len(ref_ids) > 0:
		# Reward for correct references
		correct_refs = [ref for ref in found_refs if ref in ref_ids]
		incorrect_refs = [ref for ref in found_refs if ref not in ref_ids]
		# Reward for precision (correct refs / total refs found)
		precision = len(correct_refs) / max(len(found_refs), 1)
		# Reward for recall (correct refs / total expected refs)
		recall = len(correct_refs) / len(ref_ids)
		
		# Combined F1-like score
		if precision + recall > 0:
			f1 = 2 * precision * recall / (precision + recall)
			reward = f1
	
	return reward


openai_scorer = ChatOpenAI(model="o3-mini", temperature=1.0).with_structured_output(CorrectnessScore)
def reward_answer_correctness_openai(tokenizer, data_source, solution_str, ground_truth, extra_info=None):
	answer = solution_str
	gt_answer = ground_truth

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

	prompt = ChatPromptTemplate.from_messages([
		SystemMessagePromptTemplate.from_template(EVAL_CORRECTNESS_PROMPT),
		HumanMessagePromptTemplate.from_template("Groundtruth Answer: {gt_answer}\nAnswer: {answer}"),
	])
	pipeline = prompt | openai_scorer
	response = pipeline.invoke({"gt_answer": gt_answer, "answer": answer})
	with open("logs/correctness.log", "a") as f:
		f.write(str(response) + "\n")
	return response.get("score", 0.0) / 4.0


def reward_output_length(tokenizer, data_source, solution_str, ground_truth, extra_info=None):
	reward = 0.0
	response = solution_str
	
	# Extract the thinking part
	think_pattern = f"{reasoning_start}\s*(.*?)\s*{reasoning_end}"
	think_match = re.search(think_pattern, response, re.DOTALL)
	
	# Extract the answer part
	answer_pattern = f"{solution_start}\s*(.*?)\s*{solution_end}"
	answer_match = re.search(answer_pattern, response, re.DOTALL)
	
	# Calculate reward for thinking part
	if think_match:
		thinking_text = think_match.group(1)
		thinking_tokens = len(tokenizer.encode(thinking_text))
	
		reward += 0.5 * min(thinking_tokens / 3000, 1.0)
	
	# Calculate reward for answer part
	if answer_match:
		answer_text = answer_match.group(1)
		answer_tokens = len(tokenizer.encode(answer_text))
		
		if answer_tokens <= 1000:
			reward += 0.5
	
	return reward