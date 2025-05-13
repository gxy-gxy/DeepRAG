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
from retriever import BM25, DPR, SGPT
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO) 
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

from algorithms.basic import BasicGenerator, BasicRAG

from collections import deque
class IterDRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
        if args.dataset == "hotpotqa":
            self.max_depth = 6
        else:
            self.max_depth = 10
        self.found_answer = False
        self.final_answer = None
        self.follow_up_generator = self.generator
        
    def inference(self, question, demo, case, golden_answer=None):
        # print(f"golden_answer: {golden_answer}")
        self.found_answer = False
        self.final_answer = None
        icl = open("ICL-prompt/iterDRAG.txt").read()
        # self.base_prompt = icl + 'Question: ' + question
        self.base_prompt = icl
        return self._bfs_reasoning(question, golden_answer)
    
    def print_state(self,reasoning_history):
        """
        去掉doc的问答
        D->R->D->...
        D：直接生成 R：检索生成
        """
        chain = []
        for step in reasoning_history:
            if "docs" in step:
                chain.append(f"R:{step['follow_up']}|||{step['answer']}")
            else:
                chain.append(f"D:{step['follow_up']}|||{step['answer']}")
        print("CHAIN",chain)


        
    def _bfs_reasoning(self, question, golden_answer):
        verbose = False
        best_path = None
        self.question = question
        # 使用队列存储待探索的状态
        queue = deque([([], 0)])  # (reasoning_history, depth)
        
        while queue:
            reasoning_history, depth = queue.popleft()
            if verbose:
                self.print_state(reasoning_history)
            # 检查深度限制
            if depth >= self.max_depth:
                final_answer = self._gen_final_answer(reasoning_history)
                if verbose:
                    print(f"FINAL ANSWER: {final_answer}")
                best_path = reasoning_history + [final_answer]
                return best_path
            
            # 生成follow-up问题
            follow_up = self._generate_follow_up(question, reasoning_history)
            
            
            # 检查是否需要生成最终答案
            if "So the final" in follow_up:
                final_answer = self._gen_final_answer(reasoning_history)
                if verbose:
                    print(f"FINAL ANSWER: {final_answer}")
                best_path = reasoning_history + [final_answer]
                return best_path
                    

            intermediate_answer = self._generate_intermediate_answer(question, reasoning_history, follow_up)
            
            # 创建新的推理历史
            new_history = reasoning_history + [{
                "follow_up": follow_up,
                "answer": intermediate_answer
            }]
            
            # 将直接推理路径加入队列
            queue.append((new_history, depth + 1))
        
        return []

    def _generate_follow_up(self, question, history):
        """生成follow-up问题"""
        history_text = self._format_history(history)
        prompt = self._create_chat_prompt(
            self.follow_up_generator,
            self.base_prompt if question in history_text else self.base_prompt+"\nQuestion: "+question,
            # history_text 
            "Follow up:" if not history else history_text
        ).strip("\n").removesuffix("<|eot_id|>").removesuffix("<|im_end|>")

        if history:
            prompt += "\n"
        
        for i in range(10):
            follow_up, _, _ = self.follow_up_generator.generate(
                prompt,
                max_length=self.generate_max_length,
                stop_words=["<|im_end|>", "<|eot_id|>", "Intermediate answer:", "So the final", "Let"]
            )
            
            if follow_up.strip() == "":
                continue
            
            return self._clean_generated_text(follow_up)
        return ""
    
    def _generate_intermediate_answer(self, question, history, follow_up):
        """生成中间答案"""
        retrieved_text =  self.retrieve(follow_up, topk=self.retrieve_topk)
        self.context = "\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(retrieved_text))
        history_text = self._format_history(history)
        if history_text == "":
            history_text = "Context:\n<Retrieved documents>\n" + self.context + "\n"
        prompt = self._create_chat_prompt(
            self.generator,
            self.base_prompt,
            history_text + f"Follow up: {follow_up}\nIntermediate answer: "
        ).strip("\n").removesuffix("<|eot_id|>").removesuffix("<|im_end|>")
        # print("##########################")
        # print("INTERMEDIATE PROMPT",prompt.split("##########################")[-1])
    
        answer, _, _ = self.generator.generate(
            prompt,
            max_length=self.generate_max_length,
            stop_words=["<|im_end|>", "<|eot_id|>", "Follow up:", "So the final answer"]
        )
        return self._clean_generated_text(answer)
    
    def _gen_final_answer(self, history):
        history_text = self._format_history(history)
        prompt = self._create_chat_prompt(
            self.generator,
            self.base_prompt,
            history_text + f"\nSo the final answer is: "
        ).strip("\n").removesuffix("<|eot_id|>").removesuffix("<|im_end|>")
        final_answer, _, _ = self.generator.generate(
            prompt,
            max_length=self.generate_max_length,
            stop_words=["<|im_end|>", "<|eot_id|>", "Follow up:"]
        )

        return self._clean_generated_text(final_answer)

    def _try_retrieval_path(self, follow_up, history):
        """尝试检索路径"""
        retrieved_text = self.retrieve(follow_up, topk=self.retrieve_topk)
        conversation_history = "\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(retrieved_text))
        history_text = self._format_history(history)
        
        prompt = self._create_chat_prompt(
            self.follow_up_generator,
            self.base_prompt,
            history_text + f"Follow up: {follow_up.strip()}\nLet's search the question in Wikipedia.\nContext:\n{conversation_history}\nIntermediate answer: "
        ).strip("\n").removesuffix("<|eot_id|>").removesuffix("<|im_end|>")
        
        answer, _, _ = self.follow_up_generator.generate(
            prompt,
            max_length=self.generate_max_length,
            stop_words=["<|im_end|>", "<|eot_id|>", "Follow up:", "So the final answer"]
        )
        # breakpoint()
        return self._clean_generated_text(answer), conversation_history
    
    def _create_chat_prompt(self, generator, content, assistant_content="", add_generation_prompt=False):
        """创建对话提示"""
        prompt = generator.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": content},
                {"role": "assistant", "content": assistant_content}
            ],
            add_generation_prompt=add_generation_prompt,
            tokenize=False
        )
        
        return prompt.removesuffix("<|eot_id|>").removesuffix("<|im_end|>")
    
    def _clean_generated_text(self, text):
        """清理生成的文本"""
        return (text.strip().removeprefix("Follow up:").split("Follow up:")[0]
                .split("Intermediate answer:")[0]
                .split("So the final answer")[0]
                .split("<|eot_id|>")[0]
                .split("<|im_end|>")[0]
                .strip())
    
    def normalize_answer(self, s):
        if "<answer short>" in s:
            s = s.split("<answer short>")[1].strip().split("</answer short>")[0].strip()
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _format_history(self, history):
        if not history:
            return ""
        
        formatted = f"Context:\n<Retrieved documents>\n{self.context}\n"
        formatted += f"Question: {self.question}\n"
        for step in history:
            if "docs" not in step:
                formatted += f"\nFollow up: {step['follow_up'].strip()}\nIntermediate answer: {step['answer'].strip()}"
            else:
                formatted += f"\nFollow up: {step['follow_up'].strip()}\nLet's search the question in Wikipedia.\nContext:\n{step['docs']}\nIntermediate answer: {step['answer'].strip()}"
            
        return formatted + "\n"