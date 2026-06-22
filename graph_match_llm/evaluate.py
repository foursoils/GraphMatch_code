"""
graph_match_llm - 评估脚本
============================
功能：
  - 加载已训练的检查点（GNN + Projector + Cross-Attn）以及 LoRA adapter
  - 在各数据集上批量推理，解析 Yes/No，计算 BAcc / F1 / AUC 等指标
  - 结果保存为 parquet（与其他对比实验格式一致）

用法：
  cd /root/workspace/GraphMatch_code
  python -m graph_match_llm.evaluate
  python -m graph_match_llm.evaluate --config configs/graph_match_llm.yaml --ckpt models/llm_graph/best_model.pt
  python -m graph_match_llm.evaluate --dataset minicheck  # 只跑单个数据集
"""

import os
import sys
import re
import json
import argparse

import torch
import yaml
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from graph_match_llm.dataset import LLMGraphDataset, llm_graph_collate_fn
from graph_match_llm.model   import LLMGraphModel
from utils.path_utils        import resolve_num_workers


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['llm_graph']


def resolve(base: str, rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    cleaned = rel.lstrip('.').lstrip('/').lstrip('\\')
    return os.path.normpath(os.path.join(base, cleaned))


def parse_binary_pred(text: str) -> int:
    """从生成文本中解析 Yes/No 或 1/0，返回 1/0/-1（-1 表示解析失败）。"""
    if not text:
        return -1
    text_cleaned = text.strip().lower()
    
    # 优先精确匹配单个数字 0 或 1
    if text_cleaned == '1':
        return 1
    if text_cleaned == '0':
        return 0
        
    # 精确匹配 yes/no
    if text_cleaned == 'yes':
        return 1
    if text_cleaned == 'no':
        return 0

    # 优先找 "answer is: yes/no"
    m = re.search(r'answer\s+is\s*:\s*(yes|no)', text_cleaned)
    if m:
        return 1 if m.group(1) == 'yes' else 0

    # 其次找 "answer is: 1/0"
    m = re.search(r'answer\s+is\s*:\s*(1|0)', text_cleaned)
    if m:
        return 1 if m.group(1) == '1' else 0

    # 退而求其次：找最后一个 yes/no 或 1/0
    matches = re.findall(r'\b(yes|no|1|0)\b', text_cleaned)
    if matches:
        last_match = matches[-1]
        if last_match in ('yes', '1'):
            return 1
        elif last_match in ('no', '0'):
            return 0

    return -1


# compute_metrics has been removed, using sklearn.metrics instead.


def _verify_lora_loaded(model, lora_dir: str) -> bool:
    """抽样比对 adapter 文件与模型内 LoRA 权重是否一致。"""
    from safetensors import safe_open
    from peft import PeftModel

    if not isinstance(model.llm, PeftModel):
        return False

    adapter_path = os.path.join(lora_dir, 'adapter_model.safetensors')
    if not os.path.exists(adapter_path):
        return False

    with safe_open(adapter_path, framework='pt') as f:
        saved_keys = [k for k in f.keys() if k.endswith('.lora_A.weight')]
        if not saved_keys:
            return False
        saved_key = saved_keys[0]
        saved = f.get_tensor(saved_key)

    # peft 加载后 key 可能带 .default 后缀
    suffix = saved_key.split('layers.', 1)[-1] if 'layers.' in saved_key else saved_key
    model_keys = [
        k for k in model.llm.state_dict()
        if k.endswith(suffix) or k.endswith(suffix.replace('.weight', '.default.weight'))
    ]
    if not model_keys:
        return False
    loaded = model.llm.state_dict()[model_keys[0]].cpu().float()
    return torch.allclose(loaded, saved.float(), atol=1e-5)


def load_checkpoint(model: LLMGraphModel, ckpt_path: str):
    """加载 GNN / Projector / Cross-Attn 权重，以及 LoRA adapter（如有）。"""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.gmn.load_state_dict(ckpt['gmn'])
    model.projector.load_state_dict(ckpt['projector'])
    model.cross_attn_layer.load_state_dict(ckpt['cross_attn'])
    if 'gmn_delta_cls_head' in ckpt and getattr(model, 'gmn_delta_cls_head', None) is not None:
        model.gmn_delta_cls_head.load_state_dict(ckpt['gmn_delta_cls_head'])
    if 'gmn_cls_head' in ckpt and getattr(model, 'gmn_cls_head', None) is not None:
        model.gmn_cls_head.load_state_dict(ckpt['gmn_cls_head'])
    if 'graph_global_proj' in ckpt:
        model.graph_global_proj.load_state_dict(ckpt['graph_global_proj'])
    if 'graph_global_norm' in ckpt:
        model.graph_global_norm.load_state_dict(ckpt['graph_global_norm'])
    if 'alpha_macro' in ckpt:
        model.alpha_macro.data.copy_(ckpt['alpha_macro'])
    epoch = ckpt.get('epoch', '?')
    bacc  = ckpt.get('val_bacc', '?')
    
    is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    if is_main:
        print(f"[Ckpt] 加载检查点 (Epoch={epoch}, val_BAcc={bacc})")

    # 从 adapter 目录加载 LoRA（基座 LLM 上一次性挂载，避免双重 Peft 包装）
    lora_dir = os.path.join(os.path.dirname(ckpt_path), 'lora_adapter')
    if os.path.isdir(lora_dir):
        try:
            from peft import PeftModel
            if isinstance(model.llm, PeftModel):
                raise RuntimeError(
                    "model.llm 已是 PeftModel，请用 apply_lora=False 初始化后再加载 adapter。"
                )
            model.llm = PeftModel.from_pretrained(
                model.llm, lora_dir, is_trainable=False,
            )
            if is_main:
                ok = _verify_lora_loaded(model, lora_dir)
                status = "权重校验通过" if ok else "权重校验失败"
                print(f"[Ckpt] LoRA adapter 已加载: {lora_dir} ({status})")
                if not ok:
                    print("[Warn] LoRA 权重可能未正确写入，请检查 adapter 路径与配置。")
        except Exception as e:
            if is_main:
                print(f"[Warn] LoRA adapter 加载失败: {e}")
            raise


# ---------------------------------------------------------------------------
# 单数据集评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_dataset(
    model:      LLMGraphModel,
    parquet_path: str,
    embed_path:   str,
    model_cfg:    dict,
    infer_cfg:    dict,
    output_path:  str,
    accelerator,
    test_limit:   int = 0,
) -> dict:
    ds = LLMGraphDataset(
        parquet_path=parquet_path,
        embed_model_path=embed_path,
        tokenizer=model.tokenizer,
        max_txt_len=model_cfg.get('max_txt_len', 1024),
        is_train=False,
        device=str(accelerator.device),
    )

    if test_limit > 0:
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(min(test_limit, len(ds)))))

    num_workers = resolve_num_workers(infer_cfg.get('num_workers', 2))

    loader = DataLoader(
        ds,
        batch_size=infer_cfg.get('batch_size', 4),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=llm_graph_collate_fn,
    )

    # Wrap model and loader
    model, loader = accelerator.prepare(model, loader)

    all_indices, all_preds, all_labels = [], [], []

    model.eval()
    for batch in tqdm(loader, desc=f"  推理", leave=False, disable=not accelerator.is_main_process):
        unwrapped_model = accelerator.unwrap_model(model)
        result = unwrapped_model.inference(batch)
        
        pred_ids = [parse_binary_pred(p) for p in result['pred']]
        labels = [int(l) for l in result['label']]
        indices = [int(idx) for idx in batch['index']]
        
        device = accelerator.device
        pred_tensor = torch.tensor(pred_ids, dtype=torch.long, device=device)
        label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        
        # Gather outputs across all cards
        gathered_preds = accelerator.gather_for_metrics(pred_tensor)
        gathered_labels = accelerator.gather_for_metrics(label_tensor)
        gathered_indices = accelerator.gather_for_metrics(idx_tensor)
        
        all_preds.extend(gathered_preds.cpu().tolist())
        all_labels.extend(gathered_labels.cpu().tolist())
        all_indices.extend(gathered_indices.cpu().tolist())

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Reconstruct prediction array in original order, filtering duplicates from DDP padding
        unique_results = {}
        for idx, pred, label in zip(all_indices, all_preds, all_labels):
            if idx not in unique_results:
                unique_results[idx] = (pred, label)

        underlying_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
        out_df = underlying_ds.df.copy()
        if test_limit > 0:
            out_df = out_df.iloc[:min(test_limit, len(out_df))].copy()

        reconstructed_preds = []
        reconstructed_labels = []
        for idx in range(len(out_df)):
            if idx in unique_results:
                pred, label = unique_results[idx]
                reconstructed_preds.append(pred)
                reconstructed_labels.append(label)
            else:
                reconstructed_preds.append(-1)
                reconstructed_labels.append(-1)

        out_df['pred_label'] = reconstructed_preds

        # 只保留 id, claim, doc, label 和 pred_label
        cols_to_keep = ['id', 'claim', 'doc', 'label', 'pred_label']
        cols_to_keep = [col for col in cols_to_keep if col in out_df.columns]
        out_df = out_df[cols_to_keep]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out_df.to_parquet(output_path, index=False)

        # 过滤解析失败样本后计算指标供日志打印
        valid_p = [p for p in reconstructed_preds if p != -1]
        valid_l = [l for p, l in zip(reconstructed_preds, reconstructed_labels) if p != -1]
        
        acc = accuracy_score(valid_l, valid_p) if valid_p else 0.0
        bacc = balanced_accuracy_score(valid_l, valid_p) if valid_p else 0.0
        f1 = f1_score(valid_l, valid_p, average='binary', zero_division=0) if valid_p else 0.0
        
        return {
            'Acc': acc,
            'BAcc': bacc,
            'F1': f1,
            'n_samples': len(reconstructed_preds),
            'valid_samples': len(valid_p),
        }
    else:
        return None


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  default='configs/graph_match_llm.yaml')
    parser.add_argument('--ckpt',    default=None, help='检查点路径（默认 output_dir/best_model.pt）')
    parser.add_argument('--dataset', default=None, help='只评估指定数据集（留空=全部）')
    args = parser.parse_args()

    config_path = os.path.join(_PROJ_ROOT, args.config) if not os.path.isabs(args.config) \
                  else args.config
    config    = load_config(config_path)
    data_cfg  = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    infer_cfg = config['infer']

    embed_path  = resolve(_PROJ_ROOT, model_cfg['embed_model_path'])
    output_dir  = resolve(_PROJ_ROOT, train_cfg['output_dir'])
    data_root   = resolve(_PROJ_ROOT, data_cfg['data_root'])
    out_fname   = data_cfg.get('output_filename', 'llm_graph_pred.parquet')
    test_limit  = infer_cfg.get('test_limit', 0)

    # ---- GPU 与环境变量配置 ----
    gpu_ids = infer_cfg.get('gpu_ids', None)
    if gpu_ids is not None and gpu_ids != "":
        gpus = [x.strip() for x in str(gpu_ids).split(',') if x.strip()]
        num_gpus = len(gpus)
        if num_gpus > 1:
            # 检查当前是否已经在分布式环境中运行（避免无限循环拉起）
            is_distributed = any(k in os.environ for k in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
            if not is_distributed:
                print(f"\n[Self-Launcher] 检测到多卡配置 (gpu_ids: {gpu_ids})，正在自动通过 `accelerate launch` 启动多卡评估...")
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)
                
                import subprocess
                cmd = [
                    sys.executable,
                    "-m",
                    "accelerate.commands.launch",
                    f"--num_processes={num_gpus}",
                    sys.argv[0]
                ] + sys.argv[1:]
                
                result = subprocess.run(cmd)
                sys.exit(result.returncode)
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)

    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    infer_workers = resolve_num_workers(infer_cfg.get('num_workers', 2))
    if infer_workers != infer_cfg.get('num_workers', 2) and accelerator.is_main_process:
        print(f"[DataLoader] Windows 平台：num_workers 已自动设为 0（配置值 {infer_cfg.get('num_workers')} 被忽略）")

    ckpt_path = args.ckpt or os.path.join(output_dir, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"检查点不存在: {ckpt_path}")

    # ---- 初始化模型 ----
    if accelerator.is_main_process:
        print("[Init] 初始化模型...")
    model = LLMGraphModel(config, device=accelerator.device, apply_lora=False)
    load_checkpoint(model, ckpt_path)
    model.eval()

    # ---- 数据集列表 ----
    datasets = data_cfg.get('datasets', [])
    if args.dataset:
        datasets = [args.dataset]

    # minicheck 特殊处理（test split 路径格式不同）
    minicheck_test = resolve(_PROJ_ROOT, data_cfg.get(
        'val_file', '../data/minicheck/data_with_graph/gemma_26b_tk/val.parquet'
    ))

    for ds_name in datasets:
        if accelerator.is_main_process:
            print(f"\n{'='*50}")
            print(f"数据集: {ds_name}")

        if ds_name == 'minicheck':
            parquet_path = minicheck_test
        else:
            parquet_path = os.path.join(data_root, ds_name, 'data_with_graph', 'gemma_26b_tk.parquet')

        if not os.path.exists(parquet_path):
            if accelerator.is_main_process:
                print(f"  [Skip] 文件不存在: {parquet_path}")
            continue

        output_path = os.path.join(data_root, ds_name, 'our_results', out_fname)
        if accelerator.is_main_process:
            print(f"  输入路径: {parquet_path}")
            print(f"  输出路径: {output_path}")

        metrics = evaluate_dataset(
            model=model,
            parquet_path=parquet_path,
            embed_path=embed_path,
            model_cfg=model_cfg,
            infer_cfg=infer_cfg,
            output_path=output_path,
            accelerator=accelerator,
            test_limit=test_limit,
        )

        if accelerator.is_main_process and metrics is not None:
            print(f"  [Done] 处理完成。样本数: {metrics['n_samples']} | "
                  f"Acc: {metrics['Acc']:.4f} | BAcc: {metrics['BAcc']:.4f} | F1: {metrics['F1']:.4f}")
            print(f"  结果已写出: {output_path}")

    if accelerator.is_main_process:
        print("\n[All Done] 所有数据集批量推理完毕。")


if __name__ == '__main__':
    main()
