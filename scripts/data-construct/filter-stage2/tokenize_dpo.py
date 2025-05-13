from collections import defaultdict
import json
from transformers import AutoTokenizer
from datasets import Dataset


instruction = open("ICL-prompt/instruction.txt").read()
tokenizer = AutoTokenizer.from_pretrained("hf_models/Qwen2.5-7B-Instruct")

def cal_retrieve(path):
    cnt = 0
    for cot in path:
        if "docs"  in cot:
            cnt += 1
    return cnt

def process_item(line):
    all_results = line['all_results']
    # 对于每个follow_up，都应该有两个answer，一个是根据doc回答的，一个是直接回答的，
    #   如果直接回答的后续搜索分支有正确答案，那么chosen 是直接回答，reject 是根据doc回答
    #   否则， chosen是根据doc回答，reject是直接回答
    
    correct_cot_set = set()
    next_node = defaultdict(list)
    best_path = None
    for result in all_results:

        pred, score = result[:-1], result[-1]
        if score == 1:
            if best_path is None:
                best_path = pred
            else:
                if cal_retrieve(best_path) > cal_retrieve(pred):
                    best_path = pred
    if best_path is None:
        return [], {}
    
    pairs = []
        
        # print(len(next_node[key]))
    type_counts = {"direct": 0, "doc": 0}
    # print(best_path)
    for idx, cot in enumerate(best_path):
        if isinstance(cot, str):
            break
        if "docs" in cot:
            pair = {
                "question": line['question'],
                "chosen": best_path[:idx] + [{"follow_up": line['question'], "answer": cot["answer"], "docs": cot["docs"]}],
                "rejected": best_path[:idx] + [{"follow_up": line['question'], "answer": cot["answer"]}],
            }
            pairs.append(pair)
            type_counts["doc"] += 1
        else:
            pair = {
                "question": line['question'],
                "chosen": best_path[:idx] + [{"follow_up": line['question'], "answer": cot["answer"]}],
                "rejected": best_path[:idx] + [{"follow_up": line['question'], "answer": cot["answer"], "docs": ""}],
            }
            pairs.append(pair)
            type_counts["direct"] += 1
    
    return pairs, type_counts
    
    
def reformat_pair(all_pairs):
    """
    [
  {
    "conversations": [
      {
        "from": "human",
        "value": "xxx"
      }
    ],
    "chosen": {
      "from": "gpt",
      "value": "xxx"
    },
    "rejected": {
      "from": "gpt",
      "value": "xxx"
    }
  },
  """

    def format_history_and_tokenize(history, tokenizer):
        formatted = ""
        tokenized = []
        labels = []
        for step in history[:-1]:
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

        if "docs" in history[-1]:
            s = f"Follow up: {history[-1]['follow_up']}\nLet's search the question in Wikipedia."
            encoded_s = tokenizer.encode(s)
            if encoded_s[0]==tokenizer.bos_token_id:
                encoded_s = encoded_s[1:]
            formatted += s
            tokenized.extend(encoded_s)
            labels.extend(encoded_s)
        else:
            s = f"Follow up: {history[-1]['follow_up']}\nIntermediate answer:"
            encoded_s = tokenizer.encode(s)
            if encoded_s[0]==tokenizer.bos_token_id:
                encoded_s = encoded_s[1:]
            formatted += s
            tokenized.extend(encoded_s)
            labels.extend(encoded_s)
            
        
        # tokenized.append(tokenizer.eos_token_id)
        # labels.append(tokenizer.eos_token_id)
                
        return tokenized, labels
        

    final_data = []
    
    for pair in all_pairs:

        input_tokens = tokenizer.apply_chat_template([
            {"role": "user", "content": instruction+pair["question"]}], add_generation_prompt=True)
        input_labels = [-100]*len(input_tokens)
        chosen_input_tokens, chosen_labels = format_history_and_tokenize(pair['chosen'], tokenizer)
        rejected_input_tokens, rejected_labels = format_history_and_tokenize(pair['rejected'], tokenizer)
        
        final_data.append({
            "chosen_input_ids": input_tokens+chosen_input_tokens,
            "chosen_labels": input_labels+chosen_labels,
            "chosen_attention_mask": [1]* len(input_labels+chosen_labels),
            "rejected_input_ids": input_tokens+rejected_input_tokens,
            "rejected_labels": input_labels+rejected_labels,
            "rejected_attention_mask": [1]* len(input_labels+rejected_labels),
        })
    return final_data
  

all_pairs = []
total_counts = {"direct": 0, "doc": 0}

input_folders = ["construct/dpo/hotpotqa/0", "construct/dpo/wikihop/0"]

for input_folder in input_folders:
    input_file = f"{input_folder}/output.jsonl"
    with open(input_file) as f:
        data = [json.loads(line) for line in f]
        for line in data:
            pairs, counts = process_item(line)
            if pairs != []:
                all_pairs.append(pairs)
            for key in counts:
                total_counts[key] += counts[key]

# 计算总数和百分比
total = sum(total_counts.values())
percentages = {k: (v / total) * 100 for k, v in total_counts.items()}

print("Type counts:")
for k, v in total_counts.items():
    print(f"{k}: {v} ({percentages[k]:.2f}%)")

flatten_pairs = []
for pairs in all_pairs:
    for pair in pairs:
        flatten_pairs.append(pair)

reformated_pair = reformat_pair(flatten_pairs)

print('chosen',"="*20)    
# print the first item
print("input_ids")
print(tokenizer.decode(reformated_pair[0]["chosen_input_ids"]))

print("labels")
labels = [token_id for token_id in reformated_pair[0]["chosen_labels"] if token_id != -100]
print(tokenizer.decode(labels))

print('rejected',"="*20)
# print the first item
print("input_ids")
print(tokenizer.decode(reformated_pair[0]["rejected_input_ids"]))

print("labels")
labels = [token_id for token_id in reformated_pair[0]["rejected_labels"] if token_id != -100]
print(tokenizer.decode(labels))

dataset_dict = {
    "chosen_input_ids": [],
    "chosen_labels": [],
    "rejected_input_ids": [],
    "rejected_labels": [],
    "chosen_attention_mask": [],
    "rejected_attention_mask": []
}
# breakpoint()
for item in reformated_pair:
    dataset_dict["chosen_input_ids"].append(item["chosen_input_ids"])
    dataset_dict["chosen_labels"].append(item["chosen_labels"])
    dataset_dict["chosen_attention_mask"].append(item["chosen_attention_mask"])
    dataset_dict["rejected_input_ids"].append(item["rejected_input_ids"])
    dataset_dict["rejected_labels"].append(item["rejected_labels"])
    dataset_dict["rejected_attention_mask"].append(item["rejected_attention_mask"])
    if all(x==-100 for x in item["rejected_labels"]) or all(x==-100 for x in item["chosen_labels"]):
        print(f"warining: all-100")

    if len(item["chosen_input_ids"]) != len(item["chosen_labels"]) or len(item["chosen_labels"])!=len(item["chosen_attention_mask"]):
        breakpoint()
        
    if len(item["rejected_input_ids"]) != len(item["rejected_labels"]) or len(item["rejected_labels"])!=len(item["rejected_attention_mask"]):
        breakpoint()

# breakpoint()

# 创建Dataset对象
tokenized_dataset = Dataset.from_dict(dataset_dict)


# 保存到磁盘
tokenized_dataset.save_to_disk("construct/dpo/tokenized_dataset")
