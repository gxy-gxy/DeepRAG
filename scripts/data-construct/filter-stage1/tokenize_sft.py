# simple_tokenize.py

import json
from datasets import Dataset
from transformers import AutoTokenizer
from typing import Union, List, Dict


def format_history_and_tokenize(history, tokenizer):
    formatted = ""
    tokenized = []
    labels = []
    for step in history:
        if not isinstance(step, dict):
            s = f"So the final answer is: {step}"
            encoded_s = tokenizer.encode(s)
            if encoded_s[0]==tokenizer.bos_token_id:
                encoded_s = encoded_s[1:]
            formatted += s
            tokenized.extend(encoded_s)
            labels.extend(encoded_s)
        elif "docs" not in step:
            s = f"Follow up: {step['follow_up']}\nIntermediate answer: {step['answer']}\n"
            encoded_s = tokenizer.encode(s)
            if encoded_s[0]==tokenizer.bos_token_id:
                encoded_s = encoded_s[1:]
            formatted += s
            tokenized.extend(encoded_s)
            labels.extend(encoded_s)
        else:
            s = f"Follow up: {step['follow_up']}\nLet's search the question in Wikipedia.\n"
            encoded_s = tokenizer.encode(s)
            if encoded_s[0]==tokenizer.bos_token_id:
                encoded_s = encoded_s[1:]
            formatted += s
            tokenized.extend(encoded_s)
            labels.extend(encoded_s)
            s = f"Context:\n{step['docs']}\n"
            encoded_s = tokenizer.encode(s)
            if encoded_s[0]==tokenizer.bos_token_id:
                encoded_s = encoded_s[1:]
            formatted += s
            tokenized.extend(encoded_s)
            labels.extend([-100]*len(encoded_s))
            
            s = f"Intermediate answer: {step['answer']}\n"
            encoded_s = tokenizer.encode(s)
            if encoded_s[0]==tokenizer.bos_token_id:
                encoded_s = encoded_s[1:]
            formatted += s
            tokenized.extend(encoded_s)
            labels.extend(encoded_s)
    tokenized.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id)
            
    return tokenized, labels
    

instruction = open("ICL-prompt/instruction.txt").read()
processed_data = []


tokenizer = AutoTokenizer.from_pretrained("hf_models/Qwen2.5-7B-Instruct")
with open("construct/sft/hotpotqa_wikihop.jsonl") as f:
    for line in f:
        data = json.loads(line)
        input_tokens = tokenizer.apply_chat_template([
            # {"role": "human", "content": instruction+data["question"]}], add_generation_prompt=True)
            {"role": "user", "content": instruction+data["question"]}], add_generation_prompt=True)
        # breakpoint()
        output_tokens, output_labels = format_history_and_tokenize(data["prediction"], tokenizer)
        input_labels = [-100]*len(input_tokens)
        processed_data.append({
            "input_ids": input_tokens + output_tokens,
            "labels": input_labels + output_labels  
        })
        
# print the first item
print("input_ids")
print(tokenizer.decode(processed_data[0]["input_ids"]))

print("labels")
labels = [token_id for token_id in processed_data[0]["labels"] if token_id != -100]
print(tokenizer.decode(labels))

# 将processed_data转换为Dataset格式
# 首先将数据整理成字典格式，每个key对应一个列表
dataset_dict = {
    "input_ids": [],
    "labels": []
}
# breakpoint()
for item in processed_data:
    dataset_dict["input_ids"].append(item["input_ids"])
    dataset_dict["labels"].append(item["labels"])
    if len(item["input_ids"]) != len(item["labels"]):
        breakpoint()

# 创建Dataset对象
tokenized_dataset = Dataset.from_dict(dataset_dict)

# 保存到磁盘
tokenized_dataset.save_to_disk("construct/sft/tokenized_dataset")
