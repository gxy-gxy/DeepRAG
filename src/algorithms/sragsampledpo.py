import collections
import os
import re
import string
import logging
from queue import PriorityQueue

logging.basicConfig(level=logging.INFO) 
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

from algorithms.basic import BasicGenerator, BasicRAG
from algorithms.sragsamplev2 import SRAGSampleV2

from collections import deque


class PrioritizedItem:
    def __init__(self, priority, item):
        self.priority = priority
        self.item = item
    
    def __lt__(self, other):
        return self.priority < other.priority


class SRAGSampleDPO(SRAGSampleV2):
    def __init__(self, args):
        super().__init__(args)
        if args.dataset == "hotpotqa":
            self.max_depth = 6
        else:
            self.max_depth = 10
        self.found_answer = False
        self.final_answer = None
        self.follow_up_generator = self.generator
        
    def inference(self, question, **kwargs):
        # print(f"golden_answer: {golden_answer}")
        self.found_answer = False
        self.final_answer = None
        
        golden_answer = kwargs.get('answer')
        icl = open("ICL-prompt/instruction.txt").read()
        self.base_prompt = icl + 'Question: ' + question
        
        return self._bfs_reasoning(question, golden_answer)
        
    def _bfs_reasoning(self, question, golden_answer):
        verbose = False
        all_results = []
        best_path = None
        
        pq = PriorityQueue()
        # Use PrioritizedItem to wrap data, ensuring only depth is compared
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
                    all_results.append(reasoning_history + [final_answer, 1])
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
                        final_answer = re.sub(r"(<answer short>).*?(</answer short>)", lambda match: f"{match.group(1)}{golden_answer}{match.group(2)}", final_answer)
                        
                        best_path = reasoning_history + [final_answer]
                    all_results.append(reasoning_history + [final_answer, 1])
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
            
            # Update put operation using PrioritizedItem
            pq.put(PrioritizedItem(depth + 1, (new_history, depth + 1)))
            
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

    def _generate_follow_up(self, question, history):
        """生成follow-up问题"""
        history_text = self._format_history(history)
        prompt = self._create_chat_prompt(
            self.follow_up_generator,
            self.base_prompt,
            "Follow up:" if not history else history_text
        ).strip("\n").removesuffix("<|eot_id|>").removesuffix("<|im_end|>")
        if history:
            prompt += "\n"
        
        # print("FOLLOW UP PROMPT",prompt)
        
        for i in range(10):
            follow_up, _, _ = self.follow_up_generator.generate(
                prompt,
                max_length=self.generate_max_length,
                stop_words=["<|im_end|>", "<|eot_id|>", "Intermediate answer:", "So the final", "Let"]
            )

            if follow_up.strip() == "":
                continue
            
            # print("FOLLOW UP ANSWER",follow_up)
            return self._clean_generated_text(follow_up)
        return ""
    
    def _generate_intermediate_answer(self, question, history, follow_up):
        """生成中间答案"""
        history_text = self._format_history(history)
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
        # print("INTERMEDIATE ANSWER",answer)

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
        # print("FINAL ANSWER",final_answer)

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
                .split("Let")[0]
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
            if "docs" not in step:
                formatted += f"\nFollow up: {step['follow_up'].strip()}\nIntermediate answer: {step['answer'].strip()}"
            else:
                formatted += f"\nFollow up: {step['follow_up'].strip()}\nLet's search the question in Wikipedia.\nContext:\n{step['docs']}\nIntermediate answer: {step['answer'].strip()}"
            
        return formatted + "\n"