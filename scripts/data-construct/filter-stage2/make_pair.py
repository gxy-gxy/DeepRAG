from collections import defaultdict
import json

instruction = open("ICL-prompt/instruction.txt").read()


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
    def format_history(history):
        formatted = ""
        for step in history[:-1]:
            if not isinstance(step, dict):
                formatted += f"So the final answer is: {step}"
            elif "docs" not in step:
                formatted += f"Follow up: {step['follow_up']}\nIntermediate answer: {step['answer']}\n"
            else:
                formatted += f"Follow up: {step['follow_up']}\nLet's search the question in Wikipedia.\nContext:\n{step['docs']}\nIntermediate answer: {step['answer']}\n"
        if "docs" in history[-1]:
            formatted += f"Follow up: {history[-1]['follow_up']}\nLet's search the question in Wikipedia."
        else:
            formatted += f"Follow up: {history[-1]['follow_up']}\nIntermediate answer:"
        return formatted

    final_data = []
    
    for pair in all_pairs:
        final_data.append(
            {
                "conversations": [
                {
                    "from": "user",
                    "value": instruction+pair['question']
                }
                ],
                "chosen": {
                "from": "assistant",
                "value": format_history(pair['chosen'])
                },
                "rejected": {
                "from": "assistant",
                "value": format_history(pair['rejected'])
                }
            }
        )
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
# 筛选掉 xxx 1. xxx的case

with open(f"construct/dpo/v3/pairs-v3.jsonl", 'w') as f:
    for pair in reformated_pair:
        f.write(json.dumps(pair) + '\n')






# write
# with open(xxx) 