import re


reasoning_start = "<think>"
reasoning_end   = "</think>"
solution_start = "<answer>"
solution_end = "</answer>"

EVAL_CORRECTNESS_PROMPT = """You are a senior hardware engineer and your task is to evaluate an answer to a technical question by comparing it to a reference answer. Sometimes answers have attachments, so illustrate something about the answer.
You are given the answer as input and the ground truth answer as the reference answer. Your task is to evaluate the correctness of the answer.
Correctness is a measure of how accurate the answer is compared to the ground truth answer. You will look out for factual inaccuracies, contradictions, and errors in the answer.
For now, additional information that is not present in the ground truth answer should not be considered as incorrect. Finally, sometimes the attachments are not the same (also with answers) but you should evaluate the correctness as a whole. If the answer is a bit different, then it makes sense to have different attachments that support that claim. Of course, we want the answer as a whole to be as close as possible to the grountruth answer.
Note that sometimes the groundtruth answer is not really answer, but only suggestions. These are usually expected to be an answer, if the question is simply not well defined, and therefore it is better to give suggestions on what to do next, and not give incorrect answers.
You should assign a score between 0 and 4, where 0 means that the answer is completely incorrect and/or contradicts the ground truth answer, and 4 means that the answer is completely correct.
You will also provide an analysis of your scoring and why you gave the score you did. Make sure to compare and contrast the input answer with the ground truth.
You will output the score as a number between 0 and 4 as a float.

Your output should be in the following format:
Example output: 
<analysis>
... your analysis of the answer, at most one paragraph ...
</analysis>
<score>
... your score between 0 and 4 ...
</score>"""


def all_reward_functions(data_source, solution_str, ground_truth, extra_info=None):
    return sum(
        [
			reward_output_approximate_formatting(data_source, solution_str, ground_truth, extra_info),
			reward_output_exact_formatting(data_source, solution_str, ground_truth, extra_info),
			reward_references_formatting(data_source, solution_str, ground_truth, extra_info),
    	]
    )

def reward_output_approximate_formatting(data_source, solution_str, ground_truth, extra_info=None):
    reward = 0
    response = solution_str
    # Count how many keywords are seen - we penalize if too many!
    # If we see 1, then plus some points!
    reward += 0.25 if response.count(reasoning_start) == 1 else 0.0
    reward += 0.25 if response.count(reasoning_end)   == 1 else 0.0
    reward += 0.25 if response.count(solution_start)  == 1 else 0.0
    reward += 0.25 if response.count(solution_end)    == 1 else 0.0
    return reward


def reward_output_exact_formatting(data_source, solution_str, ground_truth, extra_info=None):
	match_format = re.compile(
		rf"{reasoning_start}\s*.*?\s*{reasoning_end}\s*.*?\s*"\
		rf"{solution_start}\s*.*?\s*{solution_end}\s*.*?\s*", 
		flags = re.MULTILINE | re.DOTALL
	)
	reward = 0
	response = solution_str
	# Match if format is seen exactly!
	if match_format.search(response) is not None: reward = 2.0
	return reward

def reward_references_formatting(data_source, solution_str, ground_truth, extra_info=None):
	reward = 0.0
	response = solution_str

	# Extract the answer part
	answer_pattern = f"{solution_start}\s*(.*?)\s*{solution_end}"
	answer_match = re.search(answer_pattern, response, re.DOTALL)
	
	# Calculate reward for thinking part
	if answer_match:
		response = answer_match.group(1)
	else:
		response = ""
	
	# Extract all reference IDs from the response
	ref_pattern = re.compile(r'<ref id="([^"]+)"></ref>', re.DOTALL)
	found_refs = ref_pattern.findall(response)
	
	# Check if references are properly formatted
	if found_refs:
		# Check if references are in the expected format (alphanumeric IDs)
		valid_ref_pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
		valid_refs = 0
		
		for ref_id in found_refs:
			if valid_ref_pattern.match(ref_id):
				valid_refs += 1
		
		# Calculate score linearly between 0.0 and 2.0 based on proportion of valid refs
		reward = 1.0 * (valid_refs / len(found_refs))

	return reward


def reward_references_correctness(data_source, solution_str, ground_truth, extra_info=None):
    reward = 0
    response = solution_str
    retrieved_ids = extra_info["retrieved_ids"]
    ref_ids = extra_info["ref_ids"]

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
            reward += f1 * 1.5  # Scale to similar range as other rewards

        # Incorrect ids must be in the retrieved ids
        if len(retrieved_ids) > 0:
            reward += 0.5 * len([ref for ref in incorrect_refs if ref in retrieved_ids]) / len(retrieved_ids)
	
    return reward

# TODO: Implement this reward function
# def reward_answer_correctness(**kwargs):
# 	prompts = []
# 	for completion, question, gt_answer in zip(completions, kwargs['question'], kwargs['gt_answer']):
# 		# Apply the chat template to the prompt.
# 		answer = completion[-1]["content"]

# 		eval_prompt = {
# 			"messages": [
# 				{"role": "system", "content": EVAL_CORRECTNESS_PROMPT},
# 				{
# 					"role": "user",
# 					"content": f"Groundtruth Answer: {gt_answer}\nAnswer: {answer}",
# 				},
# 			],
# 		}
# 		eval_prompt = apply_chat_template(eval_prompt, tokenizer=tokenizer)[
# 			"text"
# 		]
# 		prompts.append(eval_prompt)

# 	# Get the model responses in batch (each response should ideally be "Yes" or "No")
# 	responses = eval_fn(prompts)
# 	responses_text = [response.outputs[0].text for response in responses]

# 	# Evaluate each response and mark as correct if "yes" appears in the answer (case-insensitive)
# 	rewards = []
# 	for response_text in responses_text:
# 		try:
# 			# Extract the score from the response text
# 			pattern = r"<score>\s*(\d+(?:\.\d+)?)\s*</score>"
# 			match = re.search(pattern, response_text, re.DOTALL)
# 			if match:
# 				reward = float(match.group(1))
# 			else:
# 				reward = 0.0
# 			rewards.append(reward)
# 		except ValueError:
# 			rewards.append(0.0)

# 	return rewards

# TODO: Implement this reward function
# def reward_output_length(data_source, solution_str, ground_truth, extra_info=None):
# 	rewards = []
# 	for completion in completions:
# 		reward = 0.0
# 		response = completion[0]["content"]
		
# 		# Extract the thinking part
# 		think_pattern = f"{reasoning_start}\s*(.*?)\s*{reasoning_end}"
# 		think_match = re.search(think_pattern, response, re.DOTALL)
		
# 		# Extract the answer part
# 		answer_pattern = f"{solution_start}\s*(.*?)\s*{solution_end}"
# 		answer_match = re.search(answer_pattern, response, re.DOTALL)
		
# 		# Calculate reward for thinking part
# 		if think_match:
# 			thinking_text = think_match.group(1)
# 			thinking_tokens = len(tokenizer.encode(thinking_text))
			
# 			if 0 <= thinking_tokens <= 2048:
# 				reward += 0.5
		
# 		# Calculate reward for answer part
# 		if answer_match:
# 			answer_text = answer_match.group(1)
# 			answer_tokens = len(tokenizer.encode(answer_text))
			
# 			if 0 <= answer_tokens <= 1024:
# 				reward += 0.5
		
# 		rewards.append(reward)
	
# 	return rewards