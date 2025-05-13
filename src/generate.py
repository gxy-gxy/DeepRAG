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



from algorithms.sragsamplev2 import SRAGSampleV2
from algorithms.sragsftv2 import SRAGSFTV2
from algorithms.sragsampledpo import SRAGSampleDPO
from algorithms.basic import *
from algorithms.iterDRAG import *