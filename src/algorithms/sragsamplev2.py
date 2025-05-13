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
logging.basicConfig(level=logging.INFO) 
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

from algorithms.basic import BasicGenerator, BasicRAG

from collections import deque
from queue import PriorityQueue  # Add this import at the top

class PrioritizedItem:
    def __init__(self, priority, item):
        self.priority = priority
        self.item = item
    
    def __lt__(self, other):
        return self.priority < other.priority

class SRAGSampleV2(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
        if args.dataset == "hotpotqa":
            self.max_depth = 6
        else:
            self.max_depth = 10
        self.found_answer = False
        self.final_answer = None
        self.follow_up_generator = BasicGenerator(model_name_or_path=self.model_name_or_path, vllm=self.vllm, remote_url=self.follow_up_remote_url)
        self.retrieve_template = open("ICL-prompt/intermediate_answer.txt").read()
        
    def inference(self, question, **kwargs):
        """
        Main inference method that accepts a question and additional parameters as kwargs
        """
        # Extract commonly used parameters with defaults
        golden_answer = kwargs.get('answer')
        qa_pairs = kwargs.get('qa_pairs', [])
        contexts = kwargs.get('ctxs', [])
        supporting_facts = kwargs.get('supporting_facts', [])
        
        # print(f"golden_answer: {golden_answer}")
        self.found_answer = False
        self.final_answer = None
        icl = open(f"ICL-prompt/{self.dataset}/question_decompose.txt").read()
            
        self.base_prompt = icl + 'Question: ' + question
        self.question = question
        
        ret = self._bfs_reasoning(question, golden_answer)
        if ret[0] is None:
            return self._golden_doc_reasoning(question, golden_answer, contexts, supporting_facts)
        else:
            return ret

    
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

    def _verify_answer(self, answer, golden_answer):
        if isinstance(golden_answer, list):
            for _golden_answer in golden_answer:
                if self.normalize_answer(_golden_answer) in self.normalize_answer(answer) or self.normalize_answer(answer) in self.normalize_answer(_golden_answer):
                    return 1
            return 0
        else:
            if self.normalize_answer(golden_answer) in self.normalize_answer(answer) or self.normalize_answer(answer) in self.normalize_answer(golden_answer):
                return 1
            else:
                return 0
    
    def _bfs_reasoning(self, question, golden_answer):
        verbose = False
        all_results = []
        best_path = None
        
        pq = PriorityQueue()
        # 使用 PrioritizedItem 包装数据，确保只比较深度值
        pq.put(PrioritizedItem(0, ([], 0)))
        
        while not pq.empty():
            current = pq.get()
            reasoning_history, depth = current.item
            
            if verbose:
                self.print_state(reasoning_history)
            # 检查深度限制
            if depth >= self.max_depth:
                final_answer = self._gen_final_answer(reasoning_history)
                if verbose:
                    print(f"FINAL ANSWER: {final_answer}")
                verify_score = self._verify_answer(final_answer, golden_answer)
                if verify_score > 0.7:
                    if best_path is None:
                        if not isinstance(golden_answer, list):
                            final_answer = re.sub(r"(<answer short>).*?(</answer short>)", lambda match: f"{match.group(1)}{golden_answer}{match.group(2)}", final_answer)
                        best_path = reasoning_history + [final_answer]
                        return best_path, all_results
                    else:
                        all_results.append(reasoning_history + [final_answer])
                else:
                    all_results.append(reasoning_history + [final_answer, verify_score])
                continue
            
            # 生成follow-up问题
            follow_up = self._generate_follow_up(question, reasoning_history)
            
            # follow-up会是空吗？
            # 检查是否需要生成最终答案
            if "So the final" in follow_up:
                final_answer = self._gen_final_answer(reasoning_history)
                if verbose:
                    print(f"FINAL ANSWER: {final_answer}")
                verify_score = self._verify_answer(final_answer, golden_answer)
                if verify_score > 0.7:
                    if best_path is None:
                        # <answer long>Scott Derrickson and Ed Wood are both American.</answer long><answer short>Yes</answer short>
                        if not isinstance(golden_answer, list):
                            final_answer = re.sub(r"(<answer short>).*?(</answer short>)", lambda match: f"{match.group(1)}{golden_answer}{match.group(2)}", final_answer)
                        
                        best_path = reasoning_history + [final_answer]
                        return best_path, all_results
                    else:
                        all_results.append(reasoning_history + [final_answer])
                else:
                    all_results.append(reasoning_history + [final_answer, verify_score])
                    
                continue # 当前节点不用扩展
                
            # 生成中间答案
            intermediate_answer = self._generate_intermediate_answer(question, reasoning_history, follow_up)
            
            # 什么时候是空？
            if not intermediate_answer:
                continue
                
            # 创建新的推理历史
            new_history = reasoning_history + [{
                "follow_up": follow_up,
                "answer": intermediate_answer
            }]
            
            # 更新 put 操作，使用 PrioritizedItem
            pq.put(PrioritizedItem(depth + 1, (new_history, depth + 1)))
            
            # 尝试检索路径
            retrieval_answer, conversation_history = self._try_retrieval_path(follow_up, reasoning_history)

            retrieval_history = reasoning_history + [{
                "follow_up": follow_up,
                "answer": retrieval_answer,
                "docs": conversation_history
            }]
            if retrieval_answer == "[No related Info]":
                all_results.append(retrieval_history)
                if verbose:
                    print(f"Final Answer: [No related Info]")
                continue
            
            pq.put(PrioritizedItem(depth + 1, (retrieval_history, depth + 1)))
                
        return best_path, all_results
    
    def _extract_gold_doc(self,contexts, supporting_facts):
        """
        "supporting_facts": [
            [
                "Arthur's Magazine",
                0
            ],
            [
                "First for Women",
                0
            ]
        ],
         "context": [
            [
                "Radio City (Indian radio station)",
                [
                    "Radio City is India's first private FM radio station and was started on 3 July 2001.",
                ]
            ],
            [
                "History of Albanian football",
                [
                    "Football in Albania existed before the Albanian Football Federation (FSHF) was created."
                ]
            ],
        """
        golden_doc = []
        # Create a dict mapping title to list of relevant sentence indices
        supporting_facts_dict = collections.defaultdict(list)
        for title, sent_idx in supporting_facts:
            supporting_facts_dict[title].append(sent_idx)
        
        # Filter contexts to only include supporting facts
        for doc_title, sentences in contexts:
            if doc_title in supporting_facts_dict:
                # Only keep sentences that are marked as supporting facts
                relevant_sentences = [
                    sent for idx, sent in enumerate(sentences) 
                    if idx in supporting_facts_dict[doc_title]
                ]
                if relevant_sentences:  # Only add if there are relevant sentences
                    for sent in relevant_sentences:
                        golden_doc.append(f"Title: {doc_title} Text:{sent}")
                    
        return golden_doc
        
        
    def _golden_doc_reasoning(self, question, golden_answer, contexts, supporting_facts):        
        all_golden_doc = self._extract_gold_doc(contexts, supporting_facts)
        verbose = False
        all_results = []
        best_path = None
        
        pq = PriorityQueue()
        # 使用 PrioritizedItem 包装数据，确保只比较深度值
        pq.put(PrioritizedItem(0, ([], 0)))
        
        while not pq.empty():
            current = pq.get()
            reasoning_history, depth = current.item
            
            if verbose:
                self.print_state(reasoning_history)
            # 检查深度限制
            if depth >= self.max_depth:
                final_answer = self._gen_final_answer(reasoning_history)
                if verbose:
                    print(f"FINAL ANSWER: {final_answer}")
                verify_score = self._verify_answer(final_answer, golden_answer)
                if verify_score > 0.7:
                    if best_path is None:
                        final_answer = re.sub(r"(<answer short>).*?(</answer short>)", lambda match: f"{match.group(1)}{golden_answer}{match.group(2)}", final_answer)
                        best_path = reasoning_history + [final_answer]
                        return best_path, all_results
                    else:
                        all_results.append(reasoning_history + [final_answer])
                else:
                    all_results.append(reasoning_history + [final_answer, verify_score])
                continue
            
            # 生成follow-up问题
            follow_up = self._generate_follow_up(question, reasoning_history)
            
            # follow-up会是空吗？
            
            # 检查是否需要生成最终答案
            if "So the final" in follow_up:
                final_answer = self._gen_final_answer(reasoning_history)
                if verbose:
                    print(f"FINAL ANSWER: {final_answer}")
                verify_score = self._verify_answer(final_answer, golden_answer)
                if verify_score > 0.7:
                    if best_path is None:
                        final_answer = re.sub(r"(<answer short>).*?(</answer short>)", lambda match: f"{match.group(1)}{golden_answer}{match.group(2)}", final_answer)
                        best_path = reasoning_history + [final_answer]
                        return best_path, all_results
                    else:
                        all_results.append(reasoning_history + [final_answer])
                else:
                    all_results.append(reasoning_history + [final_answer, verify_score])
                    
                continue # 当前节点不用扩展
                
            # 生成中间答案
            intermediate_answer = self._generate_intermediate_answer(question, reasoning_history, follow_up)
            
            # 什么时候是空？
            if not intermediate_answer:
                continue
                
            # 创建新的推理历史
            new_history = reasoning_history + [{
                "follow_up": follow_up,
                "answer": intermediate_answer
            }]
            
            # 更新 put 操作，使用 PrioritizedItem
            pq.put(PrioritizedItem(depth + 1, (new_history, depth + 1)))
            
            # 尝试检索路径
            retrieval_answer, conversation_history = self._try_retrieval_path(follow_up, reasoning_history, docs=all_golden_doc)

            retrieval_history = reasoning_history + [{
                "follow_up": follow_up,
                "answer": retrieval_answer,
                "docs": conversation_history
            }]
            if retrieval_answer == "[No related Info]":
                all_results.append(retrieval_history)
                if verbose:
                    print(f"Final Answer: [No related Info]")
                continue
            
            pq.put(PrioritizedItem(depth + 1, (retrieval_history, depth + 1)))
        if best_path is None and len(all_results)>0:
            best_path = all_results[-1]
                
        return best_path, all_results
    
    def _generate_follow_up(self, question, history):
        """生成follow-up问题"""
        history_text = self._format_history(history)
        prompt = self._create_chat_prompt(
            self.follow_up_generator,
            self.base_prompt,
            "Follow up:" if not history else history_text
        ).strip("\n").removesuffix("<|eot_id|>").removesuffix("<|im_end|>")
        # print("FOLLOW UP PROMPT",prompt)

        for i in range(10):
            follow_up, _, _ = self.follow_up_generator.generate(
                prompt,
                max_length=self.generate_max_length,
                stop_words=["<|im_end|>", "<|eot_id|>", "Intermediate answer:", "So the final"]
            )
            if follow_up.strip() == "":
                continue
            return self._clean_generated_text(follow_up)
        return ""
    
    def _generate_intermediate_answer(self, question, history, follow_up):
        """生成中间答案"""
        history_text = self._format_history(history)
        prompt = self._create_chat_prompt(
            self.generator,
            self.base_prompt,
            history_text + f"\nFollow up: {follow_up}\nIntermediate answer: "
        ).strip("\n").removesuffix("<|eot_id|>").removesuffix("<|im_end|>")
        # print("##########################")
        # print("INTERMEDIATE PROMPT",prompt.split("##########################")[-1])
    
        answer, _, _ = self.generator.generate(
            prompt,
            max_length=self.generate_max_length,
            stop_words=["<|im_end|>", "<|eot_id|>", "Follow up:", "So the final answer"]
        )
        # print("INTERMEDIATE ANSWER",answer)
        # breakpoint()
        return self._clean_generated_text(answer)
    
    def _gen_final_answer(self, history):
        history_text = self._format_history(history)
        prompt = self._create_chat_prompt(
            self.generator,
            self.base_prompt,
            history_text + f"So the final answer is: "
        ).strip("\n").removesuffix("<|eot_id|>").removesuffix("<|im_end|>")
        final_answer, _, _ = self.generator.generate(
            prompt,
            max_length=self.generate_max_length,
            stop_words=["<|im_end|>", "<|eot_id|>", "Follow up:"]
        )
        # print("FINAL ANSWER",final_answer)

        return self._clean_generated_text(final_answer)

    def _try_retrieval_path(self, follow_up, history, docs = None):
        """尝试检索路径"""
        if docs is None:
            retrieved_text = self.retrieve(follow_up, topk=self.retrieve_topk)
        else:
            retrieved_text = docs
            
        conversation_history = "\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(retrieved_text))
        
        prompt = self._create_chat_prompt(
            self.follow_up_generator,
            self.retrieve_template + f"Question: {follow_up}\nContext:\n{conversation_history}\n\nQuestion: {follow_up}",
            "Answer:"
        ).strip("\n").removesuffix("<|eot_id|>").removesuffix("<|im_end|>")
        
        answer, _, _ = self.follow_up_generator.generate(
            prompt,
            max_length=self.generate_max_length,
            stop_words=["<|im_end|>", "<|eot_id|>","[No related Info]"]
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
        # breakpoint()
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
            
        formatted = ""
        for step in history:
            if step["answer"]:
                formatted += f"\nFollow up: {step['follow_up']}\nIntermediate answer: {step['answer']}"
            else:
                formatted += f"\nFollow up: {step['follow_up']}"
        return formatted + "\n"
        
