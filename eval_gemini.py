import os
import json
import re
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

os.makedirs("evals", exist_ok=True)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from prompts import SYSTEM_PROMPT, EVAL_CORRECTNESS_PROMPT

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from dataclasses import dataclass

@dataclass
class CorrectnessScore:
	analysis: str
	score: float

openai_scorer = ChatOpenAI(model="o3-mini", temperature=1.0).with_structured_output(CorrectnessScore)
def answer_correctness(answer, gt_answer):
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

gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro-preview-03-25", temperature=0.1, max_tokens=1000000)

scores = []
for sample in tqdm(eval_data):
	question = sample["input"]["query"]
	gt_answer = sample["expected"]["groundtruth_answer"]
	document_ids = eval_chunks[question]

	print(f"Retrieved documents: {len(document_ids)}")
	documents = "\n".join([f"<document id={doc_id}>{all_data[doc_id]}</document>" for doc_id in document_ids])
	prompt = ChatPromptTemplate.from_messages([
		SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
		HumanMessagePromptTemplate.from_template("Question: {question}\nDocuments: {documents}"),
	])
	gemini_pipeline = prompt | gemini_model
	response = gemini_pipeline.invoke({"question": question, "documents": documents})    
	answer = response.content

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
	
	scores.append(
		{
			"question": question,
			"answer": answer,
			"ground_truth": gt_answer,
			"correctness": score
		}
	)
	json.dump(scores, open("evals/eval-gemini.json", "w"), indent=4)

print(f"Average score for gemini-2.5-pro: {sum([s['correctness'] for s in scores]) / len(scores)}")


