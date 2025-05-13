import json

def read_quesiton_set():
    question_set = {}
    for file in ["data/hotpot/sft_new.jsonl", "data/wikihop/sft_new.jsonl"]:
        with open(file) as f:
            for line in f:
                data = json.loads(line)
                question_set[data.get("qid",data["_id"])] = data["question"]
    return question_set

output_file_list = ["construct/sft/wikihop/0/output.jsonl", "construct/sft/hotpotqa/0/output.jsonl"]

fout = open("construct/sft/hotpotqa_wikihop.jsonl", "w")


question_set = read_quesiton_set()
for output_file in output_file_list:
    with open(output_file) as f:
        for line in f:
            data = json.loads(line)
            if data["prediction"] is not None and isinstance(data["prediction"][-1], str) and data["prediction"][-1].startswith("<answer long>") and "</answer long>" in data["prediction"][-1]:
                del data["all_results"]
                data["question"] = question_set[data["qid"]]
                should_remove = False
                # 如果intermediate answer 是1.xxx开头，去掉
                for i, step in enumerate(data["prediction"]):
                    if isinstance(step, dict) and "answer" in step:
                        if step["answer"].startswith("1. ") or step["answer"].startswith("2. ") or step["answer"].startswith("3. "):
                            should_remove = True
                            break
                if not should_remove:
                    fout.write(json.dumps(data)+"\n")
