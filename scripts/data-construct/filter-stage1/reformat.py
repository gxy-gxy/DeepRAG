import json

def format_history(history):
    formatted = ""
    for step in history:
        if not isinstance(step, dict):
            formatted += f"So the final answer is: {step}"
        elif "docs" not in step:
            formatted += f"Follow up: {step['follow_up']}\nIntermediate answer: {step['answer']}\n"
        else:
            formatted += f"Follow up: {step['follow_up']}\nLet's search the question in Wikipedia.\nContext:\n{step['docs']}\nIntermediate answer: {step['answer']}\n"
        
    return formatted
    

instruction = open("/mnt/gemininjceph2/geminicephfs/pr-others-prctrans/xinyanguan/baseline/SRAG/construct-srag/sft/instruction.txt").read()
fout = open("/mnt/gemininjceph2/geminicephfs/pr-others-prctrans/xinyanguan/baseline/SRAG/construct-srag-qwen/sft/hotpotqa_wikihop_processed.jsonl", "w")
with open("/mnt/gemininjceph2/geminicephfs/pr-others-prctrans/xinyanguan/baseline/SRAG/construct-srag-qwen/sft/hotpotqa_wikihop.jsonl") as f:
    for line in f:
        data = json.loads(line)
        data_processed = [
            {"role": "human", "content": instruction+data["question"]},
            {"role": "assistant", "content": format_history(data["prediction"])}
        ]
        fout.write(json.dumps(data_processed)+"\n")
