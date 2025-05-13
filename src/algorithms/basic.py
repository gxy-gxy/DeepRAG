import collections
import os
import random
import re
import string
import numpy as np
import logging
import spacy
from sympy import Basic
import torch
from math import exp
from scipy.special import softmax
from retriever import BM25, DPR, SGPT, GTR
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from vllm import LLM, SamplingParams
nlp = spacy.load("en_core_web_sm")

logging.basicConfig(level=logging.INFO) 
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)



import os
import time
import logging
import hmac
import hashlib
import json
import requests
import sys

def chat_wxcompletion_openai(prompt, temperature, max_length):
    # internal api, unavaible
    pass

class BasicGenerator:
    def __init__(self, model_name_or_path, vllm=False, temperature=0.0, remote_url=None):
        logger.info(f"Loading model from {model_name_or_path} with vllm={vllm} remote_url={remote_url}")
        self.vllm = vllm
        self.remote_url = remote_url
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_config = AutoConfig.from_pretrained(model_name_or_path,
                    trust_remote_code = "falcon" in model_name_or_path, use_flash_attn=True)
        self.model_name_or_path = model_name_or_path
        if self.remote_url is not None:
            if 'gpt4' not in self.remote_url:
                logger.info("Using remote moel, skip loading")
                from openai import OpenAI
                openai_api_base = self.remote_url
                print(self.remote_url)
                openai_api_key = "EMPTY"
                self.client = OpenAI(
                    api_key=openai_api_key,
                    base_url=openai_api_base[0],
                )
                models = self.client.models.list()
                self.model = models.data[0].id
                # for model in models.data:
                #     if "lora" in model.id:
                #         self.model = model.id
                print("model id: ", self.model)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        elif self.vllm:
            logger.info(f"Using vllm with tensor parallel size {torch.cuda.device_count()}")
            self.model = LLM(model_name_or_path, enable_prefix_caching=True, dtype='bfloat16', tensor_parallel_size=torch.cuda.device_count())
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", 
                    trust_remote_code = "falcon" in model_name_or_path, torch_dtype='bfloat16')
        # if self.model_config.model_type == "llama":
        #     self.space_token = "▁"
        # else:
        #     self.space_token = self.tokenizer.tokenize(' ')[0]
        self.space_token = self.tokenizer.tokenize(' ')[0]
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def vanilla_generate(self, input_text, max_length, return_logprobs=False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        if return_logprobs:
            outputs = self.model.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                max_new_tokens = max_length, 
                return_dict_in_generate = True, 
                output_scores = True,
            )
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            generated_tokens = outputs.sequences[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0]) # text = "".join(tokens)
            tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            return text, tokens, logprobs
        
        else:
            outputs = self.model.generate(
                input_ids = input_ids, 
                max_new_tokens = max_length, 
                attention_mask = attention_mask,
            )
            generated_tokens = outputs[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0])
            return text, None, None
        
    def vllm_generate(self, input_text, max_length, return_logprobs=False, stop_words=None):
        if return_logprobs:
            raise NotImplementedError("vllm does not support return_logprobs")
        if stop_words is None:
            stop_words = [self.tokenizer.eos_token]
        sampling_params = SamplingParams(max_tokens=max_length, stop=stop_words, temperature=self.temperature,include_stop_str_in_output=True)
                                        #  , skip_special_tokens=False)
        outputs = self.model.generate(
            input_text,
            sampling_params,
            use_tqdm=False
        )
        return outputs[0].outputs[0].text, None, None
    
    def gpt4_generate(self, input_text, max_length, return_logprobs=False, stop_words=None):
        return chat_wxcompletion_openai(input_text, self.temperature, max_length), None, None

    def remote_generate(self, input_text, max_length, return_logprobs=False, stop_words=None):
        # for i in range(10):
        #     try:
            # breakpoint()
            random_url = random.choice(self.remote_url) if isinstance(self.remote_url, list) else self.remote_url
            self.client.base_url = random_url
            token_len = self.tokenizer.encode(input_text, return_tensors="pt").shape[1]
            max_tokens  = min(max_length, self.model_config.max_position_embeddings - token_len)
            if max_tokens <= 0:
                return "", None, None
            completion = self.client.completions.create(
                    model=self.model,
                    prompt=input_text,
                    max_tokens=max_tokens,
                    stop=stop_words,
                    stream=False,
                    top_p=0.95,
                    temperature=self.temperature,
                    logprobs=return_logprobs)

            stop_reason = completion.choices[0].stop_reason if hasattr(completion.choices[0], "stop_reason") else completion.choices[0].matched_stop
            ret_str = completion.choices[0].text
            logprobs = completion.choices[0].logprobs.token_logprobs if return_logprobs else None
            tokens = completion.choices[0].logprobs.tokens if return_logprobs else None
            if stop_reason is None:
                ret_str += self.tokenizer.eos_token
            elif stop_words is None:
                pass
            elif stop_reason in stop_words:
                ret_str += stop_reason
            return ret_str, tokens, logprobs
            # except:
            #     self.remote_url.remove(random_url)
        

     
    def generate(self, input_text, max_length, return_logprobs=False, stop_words=None):
        if self.remote_url:
            if "gpt4" in self.remote_url:
                return self.gpt4_generate(input_text, max_length, return_logprobs=return_logprobs, stop_words=stop_words)
            return self.remote_generate(input_text, max_length, return_logprobs=return_logprobs, stop_words=stop_words)
        elif self.vllm:
            return self.vllm_generate(input_text, max_length, stop_words=stop_words)
        else:
            return self.vanilla_generate(input_text, max_length, return_logprobs)
        
        
    def generate_attn(self, input_text, max_length, solver="max", use_entropy = False, use_logprob = False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            max_new_tokens = max_length, 
            return_dict_in_generate = True, 
            output_scores = True,
            do_sample=False,
        )
        generated_tokens = outputs.sequences[:, input_length:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
        text = self.tokenizer.decode(generated_tokens[0])

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) or generated_tokens[0][i] == 13 or tokens[i-1] == '</s>' or tokens[i-1] == '<|im_end|>':
                range_.append([i, i])
            else:
                range_[-1][-1] += 1

        # attention
        atten = self.model(generated_tokens, output_attentions=True).attentions[-1][0]
        if solver == "max": 
            mean_atten, _ = torch.max(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
        elif solver == "avg":
            mean_atten = torch.sum(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
            for i in range(mean_atten.shape[0]):
                mean_atten[i] /= (mean_atten.shape[0] - i)
        elif solver == "last_token":
            mean_atten = torch.mean(atten[:, -1], dim=0)
        else:
            raise NotImplementedError
        if mean_atten.shape[0] > 1 and (tokens[0] == '</s>' or tokens[0] == '<|im_end|>'):
            mean_atten = mean_atten / sum(mean_atten[1:]).item()
        # mean_atten = mean_atten[tl:tr]
            
        # regular tokens
        seqlist = []
        attns = []
        for r in range_:
            tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(self.space_token, "")
            value = sum(mean_atten[r[0]: r[1]+1]).item()
            seqlist.append(tokenseq)
            attns.append(value)

        # -log prob
        if use_logprob:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            seqlogprobs = []
            for r in range_:
                logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqlogprobs.append(logprobseq)
        else:
            seqlogprobs = None

        # entropy
        if use_entropy:
            tmp = []
            for v in outputs.scores:
                tmp.append(v.cpu())
            tmp = np.stack(tmp)
            softmax_probs = softmax(tmp, axis=-1)
            entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
            entropies = [v[0] for v in entropies]
            seqentropies = []
            for r in range_:
                entropyseq = sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqentropies.append(entropyseq) 
        else:
            seqentropies = None 

        return text, seqlist, attns, seqlogprobs, seqentropies


class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        try:
            ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
            self.token += len(ids)
        except:
            self.token += 1
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }
         

class BasicRAG:
    def __init__(self, args):
        args = args.__dict__ 
        for k, v in args.items():
            setattr(self, k, v)
        self.generator = BasicGenerator(self.model_name_or_path, self.vllm, self.temperature, self.remote_url)
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                # gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                self.retriever = BM25(
                    tokenizer = self.generator.tokenizer, 
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name, 
                    engine = "elasticsearch",
                )
            elif self.retriever_type == "SGPT":
                self.retriever = SGPT(
                    model_name_or_path = self.sgpt_model_name_or_path, 
                    sgpt_encode_file_path = self.sgpt_encode_file_path,
                    passage_file = self.passage_file
                )
            elif self.retriever_type == "DPR":
                self.retriever = DPR()
                # self.retriever = None
            elif self.retriever_type == "GTR":
                self.retriever = GTR()
            else:
                self.retriever = None
                # raise NotImplementedError
        
        self.counter = Counter()

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            _docs_ids, docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk, 
                max_query_length = max_query_length,
            )
            return docs[0]
        elif self.retriever_type == "SGPT":
            docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk,
            )
            return docs[0] 
        elif self.retriever_type == "DPR":
            docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk,
            )
            return docs[0] 
        else:
            raise NotImplementedError
    
    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else "" 
    
    def inference(self, question, demo, case):
        # non-retrieval
        
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length, stop_words=["<|im_end|>","Question","<|eod_id|>","</s>"])
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class SingleRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, demo, case):
        
        docs = self.retrieve(question, topk=self.retrieve_topk)
        # 对 topk 个 passage 生成 prompt
        prompt = "".join([d["case"]+"\n" for d in demo])
        
        prompt += "Context:\n"
        for i, doc in enumerate(docs):
            prompt += f"[{i+1}] {doc}\n"
        # prompt += "\nQuestion:" + case.split("Question:")[-1].strip().removesuffix("Answer:")
        prompt += "Answer in the same format as before.\n"
        prompt += case
        
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        # text = ""
        return text

class SingleRAGSFT(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference_qwen(self, question, demo, case):
        
        docs = self.retrieve(question, topk=self.retrieve_topk)
        # 对 topk 个 passage 生成 prompt
        
        # prompt = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. Your response should be as detailed as possible.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>observation\n"
        observation = "Context:\n"
        for i, doc in enumerate(docs):
            observation += f"[{i+1}] {doc}\n"
        # prompt += observation
        prompt = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. Your response should be as detailed as possible.<|im_end|>\n<|im_start|>user{observation}\n\nQuestion: {question}<|im_end|>\n"

        prompt += "<|im_start|>assistant\n<answer short>"

        text, _, _ = self.generator.generate(prompt, self.generate_max_length, stop_words=["<|im_end|>, </answer short>"])
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text
    
    def inference(self, question, demo, case):
        if "qwen" in self.model_name_or_path.lower():
            return self.inference_qwen(question, demo, case)
        elif "llama" in self.model_name_or_path.lower():
            return self.inference_llama(question, demo, case)
        else:
            raise NotImplementedError
    
    def inference_llama(self, question, demo, case):
        
        docs = self.retrieve(question, topk=self.retrieve_topk)
        # 对 topk 个 passage 生成 prompt
        
        prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>observation<|end_header_id|>\n\n"
        observation = "Context:\n"
        for i, doc in enumerate(docs):
            observation += f"[{i+1}] {doc}\n"
        prompt += observation
        
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n<answer short>"

        text, _, _ = self.generator.generate(prompt, self.generate_max_length, stop_words=["<|eot_id|>, </answer short>"])
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class TokenRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        tid = 0
        for sid, sent in enumerate(sentences):
            pos = 0
            tr = tid
            while tr < len(tokens):
                apr = sent[pos:].find(tokens[tr])
                if apr == -1:
                    break
                pos = apr + len(tokens[tr])
                tr += 1
            probs = [1 - exp(v) for v in logprobs[tid:tr+1]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                # # 这里改成了替换掉最大的那个，而不是所有的
                # max_prob = 0
                # for prob, tok in zip(probs, tokens[tid:tr+1]):
                #     max_prob = max(prob, max_prob)
                for prob, tok in zip(probs, tokens[tid:tr+1]):
                    apr = curr[pos:].find(tok) + pos
                    if prob > self.hallucination_threshold:
                    # if prob == max_prob:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True
            tid = tr + 1
        
        # No hallucination
        return text, None, False
    
    def inference(self, question, demo, case):
        # 
        text = ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += case + " " + text
            new_text, tokens, logprobs = self.generator.generate(
                prompt, 
                self.generate_max_length, 
                return_logprobs=True
            )
            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            ptext, curr, hallucination = self.modifier(new_text, tokens, logprobs)
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                if self.query_formulation == "direct":
                    retrieve_question = curr.replace("[xxx]", "")
                elif self.query_formulation == "forward_all":
                    tmp_all = [question, text, ptext]
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                prompt += case + " " + text + " " + ptext.strip()
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text
    
    
class BaselineSFTRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, demo, case):
        
        # 对 topk 个 passage 生成 prompt
        prompt = self.generator.tokenizer.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True, tokenize=False)+"<answer short>"
        text, _, _ = self.generator.generate(prompt, self.generate_max_length, stop_words=["<|eot_id|>, </answer short>"])
        if self.use_counter:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text

