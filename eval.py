from transformers import AutoTokenizer, AutoModelForCausalLM

import os
import glob
import json
from prompts import SYSTEM_PROMPT, EVAL_CORRECTNESS_PROMPT

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

def load_model_weights(model):
    checkpoint_dir = "checkpoints/verl_grpo_aurix/qwen2_7b_qa_reasoning/global_step_20/actor/"
    # Find all .pt files in the checkpoint directory
    weight_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    
    # Load each weight file into the model
    for weight_file in weight_files:
        state_dict = torch.load(weight_file)
        model.load_state_dict(state_dict, strict=False)
        
    return model

eval_data = json.load(open("data/Neal-Simplified.json"))
eval_chunks = json.load(open("data/Neal-Simplified_chunks.json"))
all_data = json.load(open("data/all_data.json"))

# Initialize tokenizer and model
model_path = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Load the fine-tuned weights
model = load_model_weights(model)
model = model.cuda()
model.eval()

scores = []
for sample in eval_data:
    question = sample["input"]["query"]
    gt_answer = sample["expected"]["groundtruth_answer"]
    document_ids = eval_chunks[question]
    # Format input with system prompt and question
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<question>{question}</question>" + "\n".join([f"<document id={doc_id}>{all_data[doc_id]}</document>" for doc_id in document_ids])}
    ]
    
    # Convert prompt to string format expected by model
    prompt_str = ""
    for msg in prompt:
        if msg["role"] == "system":
            prompt_str += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
        else:
            prompt_str += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
    prompt_str += "<|im_start|>assistant\n"

    # Tokenize input
    inputs = tokenizer(prompt_str, return_tensors="pt").to("cuda")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
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




