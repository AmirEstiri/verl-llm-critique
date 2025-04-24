from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from dotenv import load_dotenv

load_dotenv()

import os
import glob
import json
from prompts import SYSTEM_PROMPT, EVAL_CORRECTNESS_PROMPT

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI

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

# Initialize tokenizer and model
model_path = "Qwen/Qwen2-7B-Instruct"
checkpoint_dir = "checkpoints/verl_grpo_aurix/qwen2_7b_qa_reasoning/global_step_20/actor/"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", trust_remote_code=True)
llm = LLM(
    model=checkpoint_dir,              # Path to your folder
    trust_remote_code=True,                   # Allow custom Qwen code
    engine_args={
        "load_format": "pt",                  # Load weights from .pt files
        "config_format": "hf",                # Load config in Hugging Face format
    }
)
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=0.8,
    repetition_penalty=1.05,
    max_tokens=10000
)


scores = []
for sample in eval_data:
    question = sample["input"]["query"]
    gt_answer = sample["expected"]["groundtruth_answer"]
    document_ids = eval_chunks[question]
    print(f"Retrieved {len(document_ids)} documents")
    
    # Format input with system prompt and question
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<question>{question}</question>" + "\n".join([f"<document id={doc_id}>{all_data[doc_id]}</document>" for doc_id in document_ids])}
    ]

    # Generate the assistant’s reply:
    chat_outputs = llm.chat(conversation, sampling_params=sampling_params)

    # Print the reply text:
    for out in chat_outputs:
        print(out.outputs[0].text)

    break

    score = answer_correctness(response, answer)

    with open("eval.log", "a") as f:
        f.write(f"Question: {question}\n")
        f.write(f"Generated response: {response}\n")
        f.write(f"Ground truth: {answer}\n")
        f.write(f"Score: {score}\n")
        f.write("-" * 80 + "\n")
    
    scores.append({
        "question": question,
        "generated_response": response,
        "ground_truth": answer,
        "score": score
    })

json.dump(scores, open("eval.json", "w"), indent=4)




