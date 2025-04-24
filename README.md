## Preparation
1. `pip install langchain langchain-openai google-generativeai langchain-google-genai python-dotenv`
2. Add `OPENAI_API_KEY` and `GOOGLE_API_KEY` to .env
3. Download the base model from HF: `huggingface-cli download Qwen/Qwen2-7B-Instruct`
4. For long-context training, add this to the `config.json` of model:  

"rope_scaling": {
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "type": "yarn"
}  
5. Run `sh grpo_aurix.sh`

6. Convert to HF format:  
`python3 scripts/model_merger.py --hf_model_path Qwen/Qwen2-7B-Instruct --local_dir=checkpoints/verl_grpo_aurix/qwen2_7b_qa_reasoning/global_step_80/actor/ --target_dir checkpoints/verl_grpo_aurix/qwen2_7b_qa_reasoning/hf_80/ --backend fsdp`

## TODOs 
Enable LoRA
