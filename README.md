# GRPO Setup
0. Assuming you have either installed all packages or deployed using verl docker.  
1. `pip install -r requirements_grpo.txt`
2. Add `OPENAI_API_KEY` and `GOOGLE_API_KEY` to .env
3. Download the base model from HF: `huggingface-cli download Qwen/Qwen2.5-32B-Instruct`
4. To enable long context training, add this to the `config.json` of model:  

```
"rope_scaling": {
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "type": "yarn"
}
```  

Path to config file: `~/.cache/huggingface/hub/models--*/snapshots/*/config.json`  

5. Run `sh grpo_aurix.sh`

6. Convert checkpoint to HF format:  
`python3 scripts/model_merger.py --hf_model_path Qwen/Qwen2-7B-Instruct --local_dir=checkpoints/*/*/global_step_x/actor/ --target_dir checkpoints/hf_eval_ckpt --backend fsdp`

7. Run evaluation on sample eval dataset: `python3 eval.py`  
For baseline, run `python3 eval_gemini.py` to get gemini numbers.  

8. Results on golden Eval dataset:  
Gemini-2.5-Pro: 70%  
Qwen/Qwen2.5-14B-Instruct: 45.3%  
Qwen/Qwen2.5-32B-Instruct: 45.1%  

## TODOs 
Enable LoRA