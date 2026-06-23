import os
import json
import torch
import pandas as pd
from torch_geometric.data import Data

from utils.path_utils import log_rank0

# ---------------------------------------------------------------------------
# PairData：GMN 需要的 claim/doc 配对图格式，统一支持 NLI 模型的 token keys 批处理
# ---------------------------------------------------------------------------
class PairData(Data):
    """将 claim 图 (s) 和 doc 图 (t) 打包成一个 PyG Data 对象，供 GMN 使用。
    同时支持将 input_ids 等 token 张量也打包进 Data 中。
    """
    _TOKEN_KEYS = {'input_ids', 'attention_mask', 'token_type_ids'}

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        if key in self._TOKEN_KEYS:
            return 0  # token id 不需要偏移
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ('edge_index_s', 'edge_index_t'):
            return 1
        if key in self._TOKEN_KEYS:
            return None  # None → PyG 会 stack 成新维度（dim=0），得到 [B, seq]
        return super().__cat_dim__(key, value, *args, **kwargs)


# ---------------------------------------------------------------------------
# 图文本化
# ---------------------------------------------------------------------------
def textualize_graph(graph_str: str):
    """将 JSON 格式三元组字符串解析为 nodes_df 和 edges_df。"""
    if not graph_str or not isinstance(graph_str, str):
        return (pd.DataFrame(columns=['node_attr', 'node_id']),
                pd.DataFrame(columns=['src', 'edge_attr', 'dst']))
    try:
        triples = json.loads(graph_str) if isinstance(graph_str, str) else graph_str
    except Exception:
        triples = []

    if not triples or not isinstance(triples, list):
        return (pd.DataFrame(columns=['node_attr', 'node_id']),
                pd.DataFrame(columns=['src', 'edge_attr', 'dst']))

    nodes_dict, edges_list = {}, []
    for tri in triples:
        if not isinstance(tri, list) or len(tri) != 3:
            continue
        src, edge_attr, dst = tri
        src       = (src       or ' ').lower().strip()
        edge_attr = (edge_attr or ' ').lower().strip()
        dst       = (dst       or ' ').lower().strip()
        if src not in nodes_dict:
            nodes_dict[src] = len(nodes_dict)
        if dst not in nodes_dict:
            nodes_dict[dst] = len(nodes_dict)
        edges_list.append({'src': nodes_dict[src], 'edge_attr': edge_attr, 'dst': nodes_dict[dst]})

    nodes_df = pd.DataFrame(nodes_dict.items(), columns=['node_attr', 'node_id'])
    edges_df = pd.DataFrame(edges_list)
    return nodes_df, edges_df


# ---------------------------------------------------------------------------
# Embedding 缓存路径推算
# ---------------------------------------------------------------------------
def get_embedding_path(parquet_path: str, embed_model_path: str):
    model_name = os.path.basename(embed_model_path)
    norm_path  = os.path.normpath(parquet_path)
    parts      = norm_path.split(os.sep)
    try:
        data_idx   = parts.index('data')
        base_parts = parts[:data_idx + 1]
        sub_parts  = parts[data_idx + 1:]
    except ValueError:
        return None

    if not sub_parts:
        return None

    if len(sub_parts) >= 4 and sub_parts[0] == 'minicheck':
        generator  = sub_parts[2]
        split_name = os.path.splitext(sub_parts[3])[0]
        new_parts  = base_parts + ['embeddings', model_name, 'minicheck', generator, f'{split_name}.pt']
    elif len(sub_parts) >= 3:
        dataset_name = sub_parts[0]
        gen_name     = os.path.splitext(sub_parts[2])[0]
        new_parts    = base_parts + ['embeddings', model_name, dataset_name, f'{gen_name}.pt']
    else:
        return None

    return os.path.normpath(os.path.sep.join(new_parts))


# ---------------------------------------------------------------------------
# 加载预计算 Embeddings
# ---------------------------------------------------------------------------
def load_precomputed_embeddings(parquet_path: str, embed_model_path: str, embed_cache_path: str = None):
    """尝试加载预计算的节点/边 embeddings。
    返回 (embeddings_dict, use_precomputed, embeddings_path)
    """
    embeddings_path = None
    if embed_cache_path and os.path.exists(embed_cache_path):
        embeddings_path = embed_cache_path
    else:
        embeddings_path = get_embedding_path(parquet_path, embed_model_path)

    embeddings_dict = None
    use_precomputed = False
    if embeddings_path and os.path.exists(embeddings_path):
        try:
            log_rank0(f"[Dataset] 使用预计算 Embedding: {embeddings_path}")
            embeddings_dict = torch.load(
                embeddings_path, map_location='cpu', weights_only=False
            )
            use_precomputed = True
        except Exception as e:
            log_rank0(f"[Dataset] 读取预计算 Embedding 失败，改为在线计算: {e}")
            
    return embeddings_dict, use_precomputed, embeddings_path


# ---------------------------------------------------------------------------
# 在线生成文本嵌入（用于 fallback）
# ---------------------------------------------------------------------------
def get_text_embedding_online(texts: list, tokenizer, model, device: str) -> torch.Tensor:
    if not texts:
        return torch.zeros((0, model.config.hidden_size))
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors='pt'
    )
    inputs  = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    mask    = inputs['attention_mask'].unsqueeze(-1).float()
    emb     = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return emb.cpu()


# ---------------------------------------------------------------------------
# 构建单图和图对
# ---------------------------------------------------------------------------
def build_single_graph(
    graph_str: str,
    sample_id,
    graph_key: str,
    embeddings_dict: dict = None,
    use_precomputed: bool = False,
    text_emb_cache: dict = None,
    embed_model = None,
    embed_tokenizer = None,
    device: str = 'cuda'
):
    """构建单张图的 x, edge_index, edge_attr。"""
    nodes, edges = textualize_graph(graph_str)

    if use_precomputed and embeddings_dict and sample_id in embeddings_dict:
        feats = embeddings_dict[sample_id]
        x = feats[f'{graph_key}_x'].float()
        e = feats[f'{graph_key}_e'].float()
    elif text_emb_cache is not None:
        # 使用预编码缓存 (如 NLI 的在线缓存)
        if len(nodes) == 0:
            emb_dim = len(next(iter(text_emb_cache.values()))) if text_emb_cache else 1024
            return (
                torch.zeros((1, emb_dim)),
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, emb_dim)),
            )
        
        node_texts = nodes['node_attr'].tolist()
        x = torch.stack([
            torch.tensor(text_emb_cache[t], dtype=torch.float32) if t in text_emb_cache
            else torch.zeros(len(next(iter(text_emb_cache.values()))))
            for t in node_texts
        ])
        
        if len(edges) > 0:
            edge_texts = edges['edge_attr'].tolist()
            e = torch.stack([
                torch.tensor(text_emb_cache[t], dtype=torch.float32) if t in text_emb_cache
                else torch.zeros(x.shape[1])
                for t in edge_texts
            ])
        else:
            e = torch.zeros((0, x.shape[1]))
    else:
        # 在线计算单个样本 (LLM fallback)
        if len(nodes) == 0:
            hidden = embed_model.config.hidden_size if hasattr(embed_model, 'config') else 1024
            return (
                torch.zeros((1, hidden)),
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, hidden)),
            )
        
        if embed_model is not None and hasattr(embed_model, 'encode'):
            # SentenceTransformer
            x = torch.tensor(embed_model.encode(nodes['node_attr'].tolist(), show_progress_bar=False), dtype=torch.float32)
            e = torch.tensor(embed_model.encode(edges['edge_attr'].tolist(), show_progress_bar=False), dtype=torch.float32) if len(edges) > 0 \
                else torch.zeros((0, x.shape[1]))
        elif embed_model is not None:
            # AutoModel
            x = get_text_embedding_online(nodes['node_attr'].tolist(), embed_tokenizer, embed_model, device)
            e = get_text_embedding_online(edges['edge_attr'].tolist(), embed_tokenizer, embed_model, device) if len(edges) > 0 \
                else torch.zeros((0, x.shape[1]))
        else:
            raise ValueError("No embedder or cache provided for online embedding.")

    edge_index = (
        torch.tensor([edges['src'].tolist(), edges['dst'].tolist()], dtype=torch.long)
        if len(edges) > 0
        else torch.zeros((2, 0), dtype=torch.long)
    )
    return x, edge_index, e


def build_pair_data(
    claim_graph_str: str,
    doc_graph_str: str,
    sample_id,
    embeddings_dict: dict = None,
    use_precomputed: bool = False,
    text_emb_cache: dict = None,
    embed_model = None,
    embed_tokenizer = None,
    device: str = 'cuda'
) -> PairData:
    """构建 Claim/Doc 匹配图对 (PairData)。"""
    x_s, ei_s, ea_s = build_single_graph(
        claim_graph_str, sample_id, 'claim',
        embeddings_dict, use_precomputed, text_emb_cache,
        embed_model, embed_tokenizer, device
    )
    x_t, ei_t, ea_t = build_single_graph(
        doc_graph_str, sample_id, 'doc',
        embeddings_dict, use_precomputed, text_emb_cache,
        embed_model, embed_tokenizer, device
    )
    return PairData(
        x_s=x_s, edge_index_s=ei_s, edge_attr_s=ea_s,
        x_t=x_t, edge_index_t=ei_t, edge_attr_t=ea_t,
    )
