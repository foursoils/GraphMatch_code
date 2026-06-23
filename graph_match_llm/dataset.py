"""
graph_match_llm - 数据集模块
==============================
负责：
  1. 读取带 CoT 的 parquet（训练集）或普通 data_with_graph parquet（评估集）
  2. 构建 claim/doc 的 PyG 图（复用 ablation/check/dataset.py 的图构建逻辑）
  3. 返回 collate 后的 batch，供 DataLoader 使用

训练集 parquet 字段：id, claim, doc, label, gt_trial, graph_claim, graph_doc
评估集 parquet 字段：id, claim, doc, label, graph_claim, graph_doc
"""

import os
import sys
import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)


# ---------------------------------------------------------------------------
# 导入公共数据处理工具
# ---------------------------------------------------------------------------
from utils.dataset_utils import PairData, textualize_graph, build_pair_data, load_precomputed_embeddings
from utils.path_utils import log_rank0



# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LLMGraphDataset(Dataset):
    """
    通用 Dataset，支持训练集（含 gt_trial CoT）和评估集（无 gt_trial）。

    Args:
        parquet_path:     parquet 文件路径
        embed_model_path: 节点/边 embedding 模型路径
        tokenizer:        LLM tokenizer（用于构建 prompt，不做实际 tokenize）
        max_txt_len:      文本部分最大 token 数（instruction truncation）
        is_train:         True=训练模式（读取 gt_trial），False=评估模式
        train_target:     "cot_and_answer" | "answer_only"
        device:           embedding 模型推理设备
    """

    SYSTEM_PROMPT = (
        "You are an expert fact-checker. "
        "Given a document and a claim, reason step by step and determine "
        "whether the document supports the claim."
    )

    def __init__(
        self,
        parquet_path:     str,
        embed_model_path: str,
        tokenizer,
        max_txt_len:      int  = 1024,
        is_train:         bool = True,
        train_target:     str  = "cot_and_answer",
        device:           str  = "cuda",
        embed_cache_path: str  = None,
    ):
        super().__init__()
        self.is_train     = is_train
        self.train_target = train_target
        self.max_txt_len  = max_txt_len
        self.tokenizer    = tokenizer
        self.device       = device

        log_rank0(f"[Dataset] 加载数据: {parquet_path}")
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)

        # 加载提示词
        sys_path = os.path.join(_PROJ_ROOT, "prompts", "hallu_detect", "system_prompt.txt")
        user_path = os.path.join(_PROJ_ROOT, "prompts", "hallu_detect", "user_prompt.txt")
        if os.path.exists(sys_path):
            with open(sys_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read().strip()
        else:
            self.system_prompt = self.SYSTEM_PROMPT

        if os.path.exists(user_path):
            with open(user_path, 'r', encoding='utf-8') as f:
                self.user_prompt_template = f.read().strip()
        else:
            self.user_prompt_template = (
                "<doc>\n{{doc}}\n</doc>\n\n<claim>\n{{claim}}\n</claim>"
            )

        # 优先使用显式并加载预计算 Embeddings
        self.embeddings_dict, self.use_precomputed, self.embeddings_path = load_precomputed_embeddings(
            parquet_path=parquet_path,
            embed_model_path=embed_model_path,
            embed_cache_path=embed_cache_path
        )

        if not self.use_precomputed:
            raise FileNotFoundError(
                f"[Dataset Error] 找不到预计算的图 embedding 缓存文件（路径：{self.embeddings_path}）。"
                f"出于显存保护考虑，在此已禁止回退到在线实时计算模式。请检查配置文件中的 train_embed_file/val_embed_file 设置是否正确。"
            )

    def __len__(self):
        return len(self.df)

    # ---- 图构建 ----

    def _build_pair_data(self, claim_graph_str: str, doc_graph_str: str, sample_id) -> PairData:
        """将 claim 图和 doc 图打包为 PairData（GMN 格式）。"""
        return build_pair_data(
            claim_graph_str=claim_graph_str,
            doc_graph_str=doc_graph_str,
            sample_id=sample_id,
            embeddings_dict=self.embeddings_dict,
            use_precomputed=True,
        )

    # ---- Prompt 构建 ----

    def _build_instruction(self, doc: str, claim: str) -> str:
        """构建用户 instruction（不含 CoT/答案部分）。"""
        return self.user_prompt_template.replace("{{doc}}", doc).replace("{{claim}}", claim)

    def _build_target(self, label: int, cot: str = "") -> str:
        """构建训练目标文本（response 部分）。"""
        label_text = "1" if label == 1 else "0"
        if self.train_target == "cot_and_answer" and cot:
            return f"{cot.strip()}\n{label_text}"
        return label_text

    # ---- __getitem__ ----

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        sample_id = row['id']
        doc       = str(row.get('doc',   ''))
        claim     = str(row.get('claim', ''))
        label     = int(row.get('label', 0))

        instruction = self._build_instruction(doc, claim)

        item = {
            'index':       idx,
            'id':          sample_id,
            'instruction': instruction,
            'label':       label,
            'label_text':  "1" if label == 1 else "0",
            'graph_pair':  self._build_pair_data(
                str(row.get('graph_claim', '')),
                str(row.get('graph_doc',   '')),
                sample_id,
            ),
        }

        if self.is_train:
            cot = str(row.get('gt_trial', '')) if self.train_target == "cot_and_answer" else ""
            item['target'] = self._build_target(label, cot)

        return item


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def llm_graph_collate_fn(batch):
    ids          = [x['id']          for x in batch]
    instructions = [x['instruction'] for x in batch]
    labels       = [x['label']       for x in batch]
    label_texts  = [x['label_text']  for x in batch]
    # GMN 使用 PairData batch，follow_batch 确保生成 x_s_batch / x_t_batch
    graph_pairs  = Batch.from_data_list(
        [x['graph_pair'] for x in batch],
        follow_batch=['x_s', 'x_t'],
    )

    out = {
        'id':          ids,
        'instruction': instructions,
        'label':       labels,
        'label_text':  label_texts,
        'graph_pair':  graph_pairs,
    }
    if 'target' in batch[0]:
        out['target'] = [x['target'] for x in batch]
    if 'index' in batch[0]:
        out['index'] = [x['index'] for x in batch]
    return out
