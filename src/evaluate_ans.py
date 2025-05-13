import os
import json
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from data import FreshQA, WikiMultiHopQA, HotpotQA, IIRC
from transformers import AutoTokenizer, AutoModelForCausalLM 

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

no_Gen = ["single-retrieval-sft", "adaptive-retrieval-sft", "baseline-sft", "answer-aware", "adaptive-retrieval","srag-sample","srag-sftv2", "srag-allretrieve", "srag-nonretrieve", "iter-drag"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--remote_url", type=str, default=None)
    tmp = parser.parse_args()
    with open(os.path.join(tmp.dir, "config.json"), "r") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.output_dir = tmp.dir
    args.remote_url = tmp.remote_url
    return args

def extract_answer(cot):
    import re
    cot = cot.split("<end>")[0]
    if "<answer short>" in cot:
        return cot.split("<answer short>")[-1].split("</answer short>")[0]
    elif "<answer long>" in cot:
        return cot.split("<answer long>")[-1].split("</answer long>")[0]
    else:
        cot = cot.split("</answer long>")[0].split("</answer short>")[0]
    
    if cot.endswith("<|im_end|>"):
        cot = cot[:-len("<|im_end|>")]
    if cot.endswith("<|eot_id|>"):
        cot = cot[:-len("<|eot_id|>")]
    pattern = r'<answer>([^<]+)</answer>(?!.*<answer>)'
    match = re.findall(pattern, cot)
    # print(cot)
    if len(match)>0:
        last_answer = match[-1]
        # print(last_answer)
        return last_answer
    else:
        return cot

def regenerate_answer(cot, tokenizer, model, case, demo):
    # print("##### origin #####")
    # print(cot)
    split_words = ["Question:", "#10000000", "Note:"]
    # split_words = ["Question:", "#10000000", "\n"]
    for word in split_words:
        pos = cot.find(word)
        if pos != -1 and pos > 0:
            cot = cot[:pos]
    if "the answer is" in cot:
        return cot
    cot = cot.rstrip().removesuffix("<|eot_id|>")
    
    cot += " So the answer is "
    prompt = "".join([d["case"]+"\n" for d in demo])
    prompt += case + " " + cot
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    input_length = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    outputs = model.generate(
        input_ids = input_ids, 
        attention_mask = attention_mask, 
        max_new_tokens = 20)
    generated_tokens = outputs[:, input_length:]
    text = tokenizer.decode(generated_tokens[0])
    text = cot + text.strip()
    for word in split_words:
        pos = text.find(word)
        if pos != -1:
            text = text[:pos] 
    # print("##### prompt #####")
    # print(prompt)
    # print("##### output #####")
    # print(text)
    # print("##### pred #####")
    return text

model_name = None
def regenerate_remote_answer(cot, tokenizer, model, case, demo):
    global model_name
    # print("##### origin #####")
    # print(cot)
    split_words = ["Question:", "#10000000", "Note:"]
    # split_words = ["Question:", "#10000000", "\n"]
    for word in split_words:
        pos = cot.find(word)
        if pos != -1 and pos > 0:
            cot = cot[:pos]
    
    if "the answer is" in cot:
        return cot 
    cot = cot.rstrip().removesuffix("<|eot_id|>")
    cot += " So the answer is "
    prompt = "".join([d["case"]+"\n" for d in demo])
    prompt += case + " " + cot
    text = model.completions.create(model=model_name,prompt=prompt,max_tokens=20).choices[0].text
    # input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # input_ids = input_ids.to(model.device)
    # input_length = input_ids.shape[1]
    # attention_mask = torch.ones_like(input_ids)
    # outputs = model.generate(
    #     input_ids = input_ids, 
    #     attention_mask = attention_mask, 
    #     max_new_tokens = 20)
    # generated_tokens = outputs[:, input_length:]
    # text = tokenizer.decode(generated_tokens[0])
    text = cot + text.strip()
    for word in split_words:
        pos = text.find(word)
        if pos != -1:
            text = text[:pos] 
    # print("##### prompt #####")
    # print(prompt)
    # print("##### output #####")
    # print(text)
    # print("##### pred #####")
    return text


def main():
    args = get_args()
    logger.info(f"{args}")
    
    if args.dataset == '2wikimultihopqa':
        data = WikiMultiHopQA(args.data_path, args.split)
    elif args.dataset == 'hotpotqa':
        data = HotpotQA(args.data_path, args.split)
    elif args.dataset == 'iirc':
        data = IIRC(args.data_path)
    elif args.dataset == 'freshqa':
        data = FreshQA(args.data_path, args.split)
    else:
        raise NotImplementedError
    data.format(fewshot=args.fewshot)

    dataset = {}
    for i in range(len(data.dataset)):
        t = data.dataset[i]
        dataset[t["qid"]] = [
            t["answer"], 
            t["answer_id"] if "answer_id" in t else None,
            t["case"] if "case" in t else None
        ]

    metrics = ["EM", "F1", "Precision", "Recall"]
    if "use_counter" not in args or args.use_counter:
        count_list = ["retrieve_count", "generate_count", "hallucinated_count", "token_count", "sentence_count"]
        metrics += count_list
    value = [[] for _ in range(len(metrics))]
    with open(os.path.join(args.output_dir, "output.jsonl"), "r") as fin:
        lines = fin.readlines()
    
    need_generate = args.dataset in ['2wikimultihopqa', "hotpotqa", "iirc", "strategyqa"] 
    if args.method == "baseline-sft":
        need_generate = False
    if need_generate and args.method not in no_Gen:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if args.remote_url is None:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto",
                                                     trust_remote_code = "falcon" in args.model_name_or_path)
        else:
            from openai import OpenAI
            # openai_api_base = args.remote_url
            # openai_api_key = "EMPTY"
            # model = OpenAI(
            #     api_key=openai_api_key,
            #     base_url=openai_api_base,
            # )
            # global model_name
            # model_name = model.models.list().data[0].id

        demo = data.dataset[0]["demo"]
    pred_out = open(f"{args.output_dir}/details.txt", "w")
    
    for line in tqdm(lines):
        rd = json.loads(line)
        qid = rd["qid"]
        pred = rd["prediction"]
        rd["retrieve_count"] = sum([1 for p in pred if isinstance(p, dict) and "docs" in p]) if pred is not None else 0
        if pred is None:
            pred = ""
        if isinstance(pred, list):
            if pred[-1] == False or pred[-1] is None:
                pred = pred[-2]
            else:
                pred = pred[-1]
        if isinstance(pred, dict):
            pred = pred["answer"]
        
            
        ground_truth, ground_truth_id, case = dataset[qid]

        if pred.endswith("<|im_end|>"):
            pred = pred[:-len("<|im_end|>")]
        if pred.endswith("<|eot_id|>"):
            pred = pred[:-len("<|eot_id|>")]

        if args.method in no_Gen:
            pred = extract_answer(pred).split("<|eot_id|>")[0]
            # pred = pred.split(',')[0]
        elif args.method == "non-retrieval":
            pred = data.get_real_prediction(pred)
        else:
            # if need_generate:
            #     if args.remote_url is None:
            #         pred = regenerate_answer(pred, tokenizer, model, case, demo) 
            #     else:
            #         pred = regenerate_remote_answer(pred, tokenizer, model, case, demo) 
            
            pred = data.get_real_prediction(pred)
            pred = pred.split("So the answer is")[0].split(".")[0]


        em_ret = data.exact_match_score(
            pred, 
            ground_truth, 
            ground_truth_id
        )
        f1_ret = data.f1_score(
            pred, 
            ground_truth, 
            ground_truth_id
        )
        value[0].append(em_ret["correct"])
        for i, k in enumerate(f1_ret.keys()):
            value[i+1].append(f1_ret[k])

        if "use_counter" not in args or args.use_counter:
            for i, k in enumerate(count_list):
                value[i+4].append(rd[k])
        detail = {
            "qid": qid, 
            "final_pred": pred,
            "EM": str(em_ret["correct"]), 
            "F1": str(f1_ret["f1"]) 
        }
        pred_out.write(json.dumps(detail)+"\n")

    ret = []
    for i, metric in enumerate(metrics):
        val = np.array(value[i])
        ret.append([metric, val.mean()])
    # 按em=0和em=retrieve_count
    em_0 = np.array(value[4])[np.array(value[0]) == 0]
    em_1 = np.array(value[4])[np.array(value[0]) == 1]
    # print(em_1)
    print(f"em=0: {em_0.mean()}, em=1: {em_1.mean()}")
    df = pd.DataFrame(ret)
    print(df)
    df.to_csv(f"{args.output_dir}/result.tsv", index=False, header=False)


if __name__ == "__main__":
    main()