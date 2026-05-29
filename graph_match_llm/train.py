"""
graph_match_llm - 训练脚本
============================
功能：
  - 加载 CoT 增强训练集 + 验证集
  - 初始化 LLMGraphModel（Qwen3.5-4B + LoRA + GNN + Cross-Attn 注入）
  - SFT 训练，loss 对 CoT + 答案部分计算
  - Early Stopping（监控 val BAcc），保存最优检查点

用法：
  cd /root/workspace/GraphMatch_code
  python -m graph_match_llm.train
  python -m graph_match_llm.train --config configs/graph_match_llm.yaml
"""

import os
import sys
import json
import re
import argparse
import random
import numpy as np

import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from graph_match_llm.dataset import LLMGraphDataset, llm_graph_collate_fn
from graph_match_llm.model   import LLMGraphModel


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


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_binary_pred(text: str) -> int:
    """从生成文本里提取 Yes/No 或 1/0，返回 1/0/-1（-1 表示解析失败）。"""
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


def compute_bacc(preds: list, labels: list) -> float:
    """计算 Balanced Accuracy。"""
    tp = fp = tn = fn = 0
    for p, l in zip(preds, labels):
        if p == 1 and l == 1: tp += 1
        elif p == 1 and l == 0: fp += 1
        elif p == 0 and l == 0: tn += 1
        elif p == 0 and l == 1: fn += 1
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return (recall_pos + recall_neg) / 2


# ---------------------------------------------------------------------------
# 验证（生成式，取 BAcc）
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model: LLMGraphModel, loader: DataLoader, device: torch.device, accelerator) -> tuple:
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="  [Val]", leave=False, disable=not accelerator.is_main_process):
        # 使用 unwrap_model 以免在评估时调用 inference 发生 DDP 错误
        unwrapped_model = accelerator.unwrap_model(model)
        result = unwrapped_model.inference(batch)
        
        pred_ids = [parse_binary_pred(p) for p in result['pred']]
        labels = [int(l) for l in result['label']]
        
        pred_tensor = torch.tensor(pred_ids, dtype=torch.long, device=device)
        label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        
        # 跨卡收集所有进程的预测与真实值
        gathered_preds = accelerator.gather_for_metrics(pred_tensor)
        gathered_labels = accelerator.gather_for_metrics(label_tensor)
        
        all_preds.extend(gathered_preds.cpu().tolist())
        all_labels.extend(gathered_labels.cpu().tolist())
        
    # 解析失败的样本算错（视为 -1）
    valid = [(p, l) for p, l in zip(all_preds, all_labels) if p != -1]
    parse_rate = len(valid) / max(len(all_preds), 1)
    bacc = compute_bacc([p for p, _ in valid], [l for _, l in valid]) if valid else 0.0
    model.train()
    return bacc, parse_rate


# ---------------------------------------------------------------------------
# 主训练
# ---------------------------------------------------------------------------

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/graph_match_llm.yaml')
    args = parser.parse_args()

    config_path = os.path.join(_PROJ_ROOT, args.config) if not os.path.isabs(args.config) \
                  else args.config
    config = load_config(config_path)

    data_cfg  = config['data']
    model_cfg = config['model']
    train_cfg = config['training']

    # ---- GPU 与环境变量配置 ----
    gpu_ids = train_cfg.get('gpu_ids', None)
    if gpu_ids is not None and gpu_ids != "":
        gpus = [x.strip() for x in str(gpu_ids).split(',') if x.strip()]
        num_gpus = len(gpus)
        if num_gpus > 1:
            # 检查当前是否已经在分布式环境中运行（避免无限循环拉起）
            is_distributed = any(k in os.environ for k in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
            if not is_distributed:
                print(f"\n[Self-Launcher] 检测到多卡配置 (gpu_ids: {gpu_ids})，正在自动通过 `accelerate launch` 启动多卡训练...")
                # 先设置 CUDA_VISIBLE_DEVICES，让子进程继承它
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)
                
                import subprocess
                # 构造启动命令，使用当前 Python 解释器和脚本路径
                cmd = [
                    sys.executable,
                    "-m",
                    "accelerate.commands.launch",
                    f"--num_processes={num_gpus}",
                    sys.argv[0]
                ] + sys.argv[1:]
                
                # 运行子进程并同步退出状态
                result = subprocess.run(cmd)
                sys.exit(result.returncode)
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)

    seed_everything(train_cfg.get('seed', 42))

    # ---- 路径解析 ----
    train_file       = resolve(_PROJ_ROOT, data_cfg['train_cot_file'])
    val_file         = resolve(_PROJ_ROOT, data_cfg['val_file'])
    embed_path       = resolve(_PROJ_ROOT, model_cfg['embed_model_path'])
    output_dir       = resolve(_PROJ_ROOT, train_cfg['output_dir'])
    train_embed_file = resolve(_PROJ_ROOT, data_cfg['train_embed_file']) \
                       if data_cfg.get('train_embed_file') else None
    val_embed_file   = resolve(_PROJ_ROOT, data_cfg['val_embed_file']) \
                       if data_cfg.get('val_embed_file') else None
    
    # 限制只有主进程创建文件夹
    grad_accum      = train_cfg.get('grad_accum_steps', 16)
    mixed_precision = train_cfg.get('mixed_precision', 'bf16')
    # find_unused_parameters=True：LoRA hook 方式注入导致部分参数不直接参与 loss，DDP 需要此选项
    ddp_kwargs  = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )
    
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    # ---- 模型初始化 ----
    accelerator.print("\n[1/4] 初始化模型...")
    model = LLMGraphModel(config, device=accelerator.device)
    if accelerator.is_main_process:
        model.print_trainable_params()

    device = accelerator.device
    tokenizer = model.tokenizer

    # ---- 数据集 ----
    accelerator.print("\n[2/4] 构建数据集...")
    train_target = train_cfg.get('train_target', 'cot_and_answer')

    train_ds = LLMGraphDataset(
        parquet_path=train_file,
        embed_model_path=embed_path,
        tokenizer=tokenizer,
        max_txt_len=model_cfg.get('max_txt_len', 1024),
        is_train=True,
        train_target=train_target,
        device=str(device),
        embed_cache_path=train_embed_file,
    )
    val_ds = LLMGraphDataset(
        parquet_path=val_file,
        embed_model_path=embed_path,
        tokenizer=tokenizer,
        max_txt_len=model_cfg.get('max_txt_len', 1024),
        is_train=False,
        device=str(device),
        embed_cache_path=val_embed_file,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get('batch_size', 1),
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 2),
        collate_fn=llm_graph_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get('eval_batch_size', 4),
        shuffle=False,
        num_workers=train_cfg.get('num_workers', 2),
        collate_fn=llm_graph_collate_fn,
        pin_memory=True,
    )

    # ---- 优化器（双学习率）----
    accelerator.print("\n[3/4] 初始化优化器和调度器...")
    lora_lr  = train_cfg.get('lora_lr',  2e-4)
    graph_lr = train_cfg.get('graph_lr', 5e-5)
    wd       = train_cfg.get('weight_decay', 0.05)

    # 按参数名分组
    lora_params  = []
    graph_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora_' in name:
            lora_params.append(param)
        else:
            graph_params.append(param)  # GNN、Projector、CrossAttn

    optimizer = AdamW([
        {'params': lora_params,  'lr': lora_lr,  'weight_decay': 0.0},
        {'params': graph_params, 'lr': graph_lr, 'weight_decay': wd},
    ])

    # 用 accelerator.prepare 托管所有核心对象
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # 准备完数据加载器后，在多卡下它的长度会缩减，此时再算总步数更准确
    num_epochs    = train_cfg.get('num_epochs', 5)
    total_steps   = (len(train_loader) // grad_accum) * num_epochs
    warmup_steps  = int(total_steps * train_cfg.get('warmup_ratio', 0.1))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scheduler = accelerator.prepare(scheduler)

    # ---- 训练循环 ----
    accelerator.print(f"\n[4/4] 开始训练（{num_epochs} epoch，grad_accum={grad_accum}）\n")
    patience      = train_cfg.get('patience', 3)
    best_bacc     = -1.0
    no_improve    = 0
    best_ckpt     = os.path.join(output_dir, 'best_model.pt')
    history       = []

    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_lm_loss  = 0.0
        epoch_aux_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{num_epochs}", dynamic_ncols=True, disable=not accelerator.is_main_process)
        for step, batch in enumerate(pbar, 1):
            with accelerator.accumulate(model):
                total_loss, lm_loss, aux_loss = model(batch)
                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                    )
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # 平均所有 GPU 的 loss 供展示
            if accelerator.state.num_processes > 1:
                loss_for_log = accelerator.reduce(total_loss.detach(), "mean").item()
                lm_loss_for_log = accelerator.reduce(lm_loss.detach(), "mean").item()
                aux_loss_for_log = accelerator.reduce(aux_loss.detach(), "mean").item()
            else:
                loss_for_log = total_loss.item()
                lm_loss_for_log = lm_loss.item()
                aux_loss_for_log = aux_loss.item()

            epoch_loss     += loss_for_log
            epoch_lm_loss  += lm_loss_for_log
            epoch_aux_loss += aux_loss_for_log

            if accelerator.is_main_process:
                pbar.set_postfix({
                    'lm':  f'{epoch_lm_loss/step:.3f}',
                    'aux': f'{epoch_aux_loss/step:.3f}',
                    'lr':  f'{optimizer.param_groups[0]["lr"]:.1e}',
                })

        n = len(train_loader)
        avg_loss     = epoch_loss     / n
        avg_lm_loss  = epoch_lm_loss  / n
        avg_aux_loss = epoch_aux_loss / n
        accelerator.print(f"\nEpoch {epoch:02d} | total={avg_loss:.4f}  lm={avg_lm_loss:.4f}  aux(gmn)={avg_aux_loss:.4f}")

        # ---- 验证 ----
        bacc, parse_rate = validate(model, val_loader, device, accelerator)
        accelerator.print(f"         | val_BAcc={bacc:.4f}  parse_rate={parse_rate:.2%}")
        
        if accelerator.is_main_process:
            history.append({
                'epoch':      epoch,
                'train_loss': avg_loss,
                'lm_loss':    avg_lm_loss,
                'aux_loss':   avg_aux_loss,
                'val_bacc':   bacc,
            })

            # ---- 保存最优 ----
            if bacc > best_bacc:
                best_bacc  = bacc
                no_improve = 0
                unwrapped_model = accelerator.unwrap_model(model)
                ckpt = {
                    'epoch':        epoch,
                    'val_bacc':     bacc,
                    'train_loss':   avg_loss,
                    'gmn':          unwrapped_model.gmn.state_dict(),
                    'projector':    unwrapped_model.projector.state_dict(),
                    'cross_attn':   unwrapped_model.cross_attn_layer.state_dict(),
                    'gmn_cls_head': unwrapped_model.gmn_cls_head.state_dict(),
                    'graph_to_head': unwrapped_model.graph_to_head.state_dict(),
                    'gammas':       unwrapped_model.gammas.data,
                }
                # LoRA adapter 单独保存
                try:
                    unwrapped_model.llm.save_pretrained(os.path.join(output_dir, 'lora_adapter'))
                except Exception as e:
                    print(f"  [Warn] LoRA adapter 保存失败: {e}")
                
                # 使用 accelerator.save 安全写入文件
                accelerator.save(ckpt, best_ckpt)
                print(f"  ✅ 最优模型已保存 (BAcc={bacc:.4f})")
            else:
                no_improve += 1
                print(f"  ⚠️  无改善 ({no_improve}/{patience})")
                
        # 跨进程同步 early stopping 计数
        no_improve_tensor = torch.tensor(no_improve, dtype=torch.long, device=device)
        # 用 broadcast 广播主进程的 no_improve 给其他进程，保持同步退出
        if accelerator.state.num_processes > 1:
            import torch.distributed as dist
            dist.broadcast(no_improve_tensor, src=0)
            no_improve = no_improve_tensor.item()

        if no_improve >= patience:
            accelerator.print("  Early Stopping 触发，训练结束。")
            break

    # ---- 保存训练历史 ----
    if accelerator.is_main_process:
        history_path = os.path.join(output_dir, 'train_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\n训练完成！最优 val_BAcc={best_bacc:.4f}")
        print(f"检查点: {best_ckpt}")
        print(f"训练历史: {history_path}")


if __name__ == '__main__':
    train()
