import os
import glob
import json

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from prompts import SYSTEM_PROMPT, EVAL_CORRECTNESS_PROMPT

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from dataclasses import dataclass

@dataclass
class CorrectnessScore:
	analysis: str
	score: float

openai_scorer = ChatOpenAI(model="gpt-4.1-nano", temperature=0.1).with_structured_output(CorrectnessScore)
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

gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro-preview-03-25", temperature=0.1, max_tokens=1000000)

scores = []
for sample in eval_data:
	question = sample["input"]["query"]
	gt_answer = sample["expected"]["groundtruth_answer"]
	document_ids = eval_chunks[question]

	print(f"Retrieved {len(document_ids)} documents")

	documents = "\n".join([f"<document id={doc_id}>{all_data[doc_id]}</document>" for doc_id in document_ids])
	prompt = ChatPromptTemplate.from_messages([
		SystemMessagePromptTemplate.from_template(EVAL_CORRECTNESS_PROMPT),
		HumanMessagePromptTemplate.from_template("Question: {question}\nDocuments: {documents}"),
	])
	gemini_pipeline = prompt | gemini_model
	response = gemini_pipeline.invoke({"question": question, "documents": documents})    
	answer = response.content

	score = answer_correctness(answer, gt_answer)

	with open("eval-gemini.log", "a") as f:
		f.write(f"Question: {question}\n")
		f.write(f"Generated response: {answer}\n")
		f.write(f"Ground truth: {gt_answer}\n")
		f.write(f"Score: {score}\n")
		f.write("-" * 80 + "\n")
	
	scores.append({
		"question": question,
		"generated_response": answer,
		"ground_truth": answer,
		"score": score
	})
	json.dump(scores, open("eval-gemini.json", "w"), indent=4)




