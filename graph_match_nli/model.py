"""
NLI-Graph 融合幻觉检测模型

架构（对应设计图）:
  1. GMN 子图匹配（复用 graph_match/model.py 的 GMNPropLayer）
       - 输入: Claim 图 G_C、Doc 子图 G_D（SentenceTransformer 嵌入）
       - 输出: 节点级对齐向量 [N, node_hidden_dim]、图级全局向量 [B, node_hidden_dim]
  2. NLI 编码器（DeBERTa-v3-base，二分类微调版）
       - 前 k 层正常运行，得到隐状态 h_k [B, seq, 768]
       - Cross-Attention: query=h_k, key/value=节点嵌入（投影到 768）
       - LayerNorm(h_k + F(h_k))  ← 节点信息注入 + 残差
       - LayerNorm(h  + h_graph_global_proj)  ← 图级全局注入 + 残差
       - 后 L-k 层继续运行
  3. NLI 分类头（复用预训练权重）
       - [CLS] 向量 → Linear → 2 类（entailment=支持 / not_entailment=幻觉）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel, AutoConfig




from utils.gmn import GMNEncoder


# ────────────────────────────────────────────────────────────────────────────
# Cross-Attention 注入层：将 GMN 节点嵌入注入 DeBERTa 隐状态
# ────────────────────────────────────────────────────────────────────────────
class GraphCrossAttentionLayer(nn.Module):
    """
    query: h_k  [B, seq_len, 768]（DeBERTa 第 k 层输出）
    key/value: 节点嵌入（claim + doc 合并）[total_nodes, node_hidden_dim]

    对 batch 中第 i 个样本，仅用该样本对应的节点作为 key/value。
    """
    def __init__(self, hidden_size: int = 768, node_hidden_dim: int = 256,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # 节点嵌入投影到 DeBERTa 隐层维度
        self.node_proj = nn.Linear(node_hidden_dim, hidden_size)

        # Multi-head Cross-Attention
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

        # Tanh Gating parameter initialized to 0
        self.alpha_micro = nn.Parameter(torch.zeros(1))

    def forward(self, h_k: torch.Tensor,
                node_emb: torch.Tensor,
                node_batch: torch.Tensor) -> torch.Tensor:
        """
        h_k:        [B, seq_len, hidden_size]
        node_emb:   [total_nodes, node_hidden_dim]  （claim + doc 节点合并）
        node_batch: [total_nodes]  节点归属 of batch index

        返回: [B, seq_len, hidden_size]，已加残差 + LayerNorm
        """
        B, seq_len, _ = h_k.shape
        device = h_k.device

        # 节点投影
        node_h = self.node_proj(node_emb)  # [total_nodes, hidden_size]

        # 逐样本做 cross-attention，再 stack 回 batch
        # 用 padding 方式统一长度，避免 for 循环开销
        # 1) 找每个 batch 的节点数
        max_nodes = int((node_batch.bincount()).max().item())

        # 2) 构建 padded key/value [B, max_nodes, hidden_size]
        kv_pad = torch.zeros(B, max_nodes, self.hidden_size, device=device)
        key_padding_mask = torch.ones(B, max_nodes, dtype=torch.bool, device=device)  # True=ignore

        for i in range(B):
            idx = (node_batch == i).nonzero(as_tuple=True)[0]
            n = idx.size(0)
            kv_pad[i, :n] = node_h[idx]
            key_padding_mask[i, :n] = False  # 有效节点不 mask

        # 3) Multi-head Cross-Attention
        Q = self.q_proj(h_k)                          # [B, seq_len, hidden_size]
        K = self.k_proj(kv_pad)                        # [B, max_nodes, hidden_size]
        V = self.v_proj(kv_pad)                        # [B, max_nodes, hidden_size]

        # reshape 为多头
        def split_heads(t):
            b, s, d = t.shape
            return t.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
            # → [B, num_heads, s, head_dim]

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        scale = self.head_dim ** 0.5
        attn = torch.matmul(Q, K.transpose(-1, -2)) / scale  # [B, heads, seq, max_nodes]

        # 对 padding 位置填 -inf
        # key_padding_mask: [B, max_nodes] → [B, 1, 1, max_nodes]
        attn = attn.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
        )
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)                           # [B, heads, seq, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, seq_len, self.hidden_size)
        out = self.out_proj(out)                              # [B, seq_len, hidden_size]

        # Tanh Gating + 残差 + LayerNorm
        return self.norm(h_k + torch.tanh(self.alpha_micro) * out)


# ────────────────────────────────────────────────────────────────────────────
# 主模型：NLIGraphClassifier
# ────────────────────────────────────────────────────────────────────────────
class NLIGraphClassifier(nn.Module):
    """
    GMN + DeBERTa-v3-base（NLI 微调版）融合幻觉检测模型

    注入位置 inject_layer_k: DeBERTa 第 k 层（0-indexed）之后插入 Cross-Attention
    """
    def __init__(self,
                 nli_model_path: str,
                 node_input_dim: int = 1024,
                 edge_input_dim: int = 1024,
                 node_hidden_dim: int = 256,
                 num_prop_layers: int = 3,
                 inject_layer_k: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 freeze_nli_layers: int = 0):
        """
        :param nli_model_path:    本地 DeBERTa NLI 模型路径
        :param inject_layer_k:    在 DeBERTa 第 k 层后注入图信息（共 12 层，建议 4~8）
        :param freeze_nli_layers: 冻结 DeBERTa 前 N 层（0 = 全部参与训练）
        """
        super().__init__()
        self.inject_layer_k = inject_layer_k

        # ── GMN 编码器 ────────────────────────────────────────────────────
        self.gmn = GMNEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            node_hidden_dim=node_hidden_dim,
            num_prop_layers=num_prop_layers,
            dropout=dropout,
        )

        # ── DeBERTa 编码器（只取 encoder body，不含分类头）─────────────────
        config = AutoConfig.from_pretrained(nli_model_path)
        self.nli_encoder = AutoModel.from_pretrained(nli_model_path, config=config)
        hidden_size = config.hidden_size  # 768

        # 可选：冻结前 N 层
        if freeze_nli_layers > 0:
            for i, layer in enumerate(self.nli_encoder.encoder.layer):
                if i < freeze_nli_layers:
                    for p in layer.parameters():
                        p.requires_grad = False

        # ── Cross-Attention 注入层 ────────────────────────────────────────
        self.cross_attn = GraphCrossAttentionLayer(
            hidden_size=hidden_size,
            node_hidden_dim=node_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ── 图级全局向量投影 + 注入 ───────────────────────────────────────
        self.graph_global_proj = nn.Linear(node_hidden_dim, hidden_size)
        self.graph_global_norm = nn.LayerNorm(hidden_size)

        # Tanh Gating parameter for macro-level injection initialized to 0
        self.alpha_macro = nn.Parameter(torch.zeros(1))

        # ── 分类头（复用 NLI 预训练逻辑，2 类）──────────────────────────
        # DeBERTa NLI 模型的分类头结构: Linear(768, 2)
        # 这里重建，权重可从预训练模型加载
        self.dropout_cls = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

        # 尝试复用预训练分类头权重
        self._try_load_classifier_weights(nli_model_path)

    def _try_load_classifier_weights(self, model_path: str):
        """尝试从预训练 NLI 模型加载分类头权重"""
        try:
            from transformers import AutoModelForSequenceClassification
            pretrained = AutoModelForSequenceClassification.from_pretrained(model_path)
            # DeBERTa 分类头在 classifier 或 pooler+classifier
            if hasattr(pretrained, 'classifier'):
                cls_state = pretrained.classifier.state_dict()
                # 若权重形状匹配则加载
                if cls_state['weight'].shape == self.classifier.weight.shape:
                    self.classifier.load_state_dict(cls_state)
                    print("✅ 已复用预训练 NLI 分类头权重")
            del pretrained
        except Exception as e:
            print(f"⚠️  分类头权重复用失败（将随机初始化）: {e}")

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor,
                graph_batch) -> torch.Tensor:
        """
        input_ids / attention_mask / token_type_ids: DeBERTa tokenizer 输出
            格式: [CLS] doc tokens [SEP] claim tokens [SEP]
            shape: [B, seq_len]
        graph_batch: PyG PairData batch（含 x_s/x_t/edge_index_s/edge_index_t 等）

        返回: logits [B, 2]（entailment=支持, not_entailment=幻觉）
        """
        B = input_ids.size(0)

        # ── Step 1: GMN 处理图，获取节点级 + 图级嵌入 ───────────────────
        node_c, node_d, _, batch_c, batch_d = self.gmn(graph_batch)

        # 用 global_mean_pool 计算 claim 与 doc 图全局特征并做差值
        g_c = global_mean_pool(node_c, batch_c, size=B)  # [B, node_hidden_dim]
        g_d = global_mean_pool(node_d, batch_d, size=B)  # [B, node_hidden_dim]
        delta_h_g = g_c - g_d                            # [B, node_hidden_dim]

        # 合并 claim + doc 节点，统一做 cross-attention
        node_all = torch.cat([node_c, node_d], dim=0)       # [N_c+N_d, node_hidden_dim]
        # batch index 偏移（两图节点 batch index 相同，直接 cat 即可）
        node_batch_all = torch.cat([batch_c, batch_d], dim=0)

        # ── Step 2: 注册 forward hook，在第 k 层 Attention 输出后注入图信息 ────────
        # 使用 hook 而非手动逐层调用，避免触碰 DeBERTa-v3 内部
        # rel_embeddings / query_states / attention_mask 格式等细节
        def _inject_hook(module, layer_input, layer_output):
            # DeBERTa 层在 query_states=None 时返回纯 tensor [B, seq, 768]
            # 在 query_states!=None 时返回 (hidden, query) tuple
            # 统一取出 hidden_states，注入后再还原原始格式
            is_tuple = isinstance(layer_output, tuple)
            h = layer_output[0] if is_tuple else layer_output   # [B, seq, 768]

            # 3a. 节点级 Cross-Attention 注入 + 残差 + LayerNorm (已包含 Tanh Gating)
            h = self.cross_attn(h, node_all, node_batch_all)

            # 3b. 图级差异向量注入 + Tanh 门控 + 残差 + LayerNorm
            # delta_h_g [B, node_hidden_dim] → [B, 1, hidden_size]
            g_proj = self.graph_global_proj(delta_h_g).unsqueeze(1)
            h = self.graph_global_norm(h + torch.tanh(self.alpha_macro) * g_proj)

            # 还原原始返回格式
            return (h,) + layer_output[1:] if is_tuple else h

        # 注册在 attention 子模块上，实现在 Self-Attention 之后，FFN 之前注入
        target_submodule = self.nli_encoder.encoder.layer[self.inject_layer_k - 1].attention
        hook = target_submodule.register_forward_hook(_inject_hook)

        # ── Step 3: DeBERTa 完整前向（内部自动处理 rel_embeddings 等）──────
        try:
            encoder_outputs = self.nli_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        finally:
            hook.remove()   # 无论是否异常，都确保 hook 被清理

        # ── Step 4: [CLS] → 分类头 ───────────────────────────────────────
        cls_h = encoder_outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]
        cls_h = self.dropout_cls(cls_h)
        logits = self.classifier(cls_h)                     # [B, 2]

        return logits
