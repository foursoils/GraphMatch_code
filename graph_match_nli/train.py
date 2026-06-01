"""
NLI-Graph 融合模型训练脚本

流程:
  1. 加载 configs/nli_graph.yaml
  2. 构建 NLIGraphDataset（tokenizer 输入 + 图对）
  3. 训练 NLIGraphClassifier（GMN + DeBERTa 中间层注入）
  4. Early Stopping，按 val F1 保存最优检查点
"""
import os
import sys
import yaml
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')  # MPS 不限制上限，避免 OOM
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import NLIGraphClassifier
from dataset import NLIGraphDataset


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 验证函数
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, criterion, use_amp=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            input_ids      = batch.input_ids
            attention_mask = batch.attention_mask
            token_type_ids = batch.token_type_ids
            labels = batch.y.view(-1).long()

            with autocast('cuda', enabled=use_amp):
                logits = model(input_ids, attention_mask, token_type_ids, batch)
                # NLI 模型: class 0=entailment(支持), class 1=not_entailment(幻觉)
                # 数据集:   label 1=支持,              label 0=幻觉
                # → 损失目标需要翻转: dataset_label=1 → nli_class=0
                labels_nli = 1 - labels
                loss = criterion(logits, labels_nli)
            total_loss += loss.item() * labels.size(0)

            probs = torch.softmax(logits, dim=-1)[:, 0]   # class 0 (entailment/支持) 概率
            preds = (logits[:, 0] > logits[:, 1]).long()  # entailment 赢 → 预测为支持(1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    n = len(all_labels)
    avg_loss = total_loss / n
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
    return avg_loss, acc, f1, auc


# ─────────────────────────────────────────────────────────────────────────────
# 主训练流程
# ─────────────────────────────────────────────────────────────────────────────
def train():
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'configs', 'graph_match_nli.yaml')
    config      = load_config(config_path)['nli_graph']
    cfg_dir     = os.path.join(base_dir, 'configs')

    def resolve(p):
        return os.path.normpath(os.path.join(cfg_dir, p))

    train_path    = resolve(config['data']['train_parquet'])
    val_path      = resolve(config['data']['val_parquet'])
    nli_model_path = resolve(config['model']['nli_model_path'])
    emb_model_path = resolve(config['model']['embedding_model_path'])
    best_loss_path = resolve(config['training']['best_loss_path'])
    best_f1_path   = resolve(config['training']['best_f1_path'])
    os.makedirs(os.path.dirname(best_loss_path), exist_ok=True)
    os.makedirs(os.path.dirname(best_f1_path), exist_ok=True)

    _dev = config['model']['device']
    if _dev == 'cuda' and not torch.cuda.is_available():
        _dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    elif _dev == 'mps' and not torch.backends.mps.is_available():
        _dev = 'cpu'
    device = torch.device(_dev)
    set_seed(config['training'].get('seed', 42))
    print(f"使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(nli_model_path, use_fast=False)
    train_ds = NLIGraphDataset(
        train_path, tokenizer, emb_model_path,
        max_length=config['model']['max_length'],
        device=str(device),
        embed_cache_path=resolve(config['data']['train_embed_file']) if config['data'].get('train_embed_file') else None,
    )
    val_ds = NLIGraphDataset(
        val_path, tokenizer, emb_model_path,
        max_length=config['model']['max_length'],
        device=str(device),
        embed_cache_path=resolve(config['data']['val_embed_file']) if config['data'].get('val_embed_file') else None,
    )
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        follow_batch=['x_s', 'x_t']
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        follow_batch=['x_s', 'x_t']
    )

    # ── 模型 ──────────────────────────────────────────────────────────────
    print("[4/5] 初始化 NLI-Graph 融合模型...")
    model = NLIGraphClassifier(
        nli_model_path      = nli_model_path,
        node_input_dim      = config['model']['node_input_dim'],
        edge_input_dim      = config['model']['node_input_dim'],
        node_hidden_dim     = config['model']['node_hidden_dim'],
        num_prop_layers     = config['model']['num_prop_layers'],
        inject_layer_k      = config['model']['inject_layer_k'],
        num_heads           = config['model']['num_heads'],
        dropout             = config['model']['dropout'],
        freeze_nli_layers   = config['model'].get('freeze_nli_layers', 0),
    ).to(device).float()   # MPS 上强制 float32，避免 f16/f32 混合导致崩溃

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,} | 可训练: {trainable_params:,}")

    # ── 差异化学习率：DeBERTa 用更小 lr，GMN + 注入层用较大 lr ────────────
    deberta_params = list(model.nli_encoder.parameters())
    other_params   = [p for p in model.parameters()
                      if not any(p is q for q in deberta_params)]
    lr_base   = config['training']['learning_rate']
    lr_deberta = lr_base * config['training'].get('deberta_lr_ratio', 0.1)

    optimizer = AdamW([
        {'params': deberta_params, 'lr': lr_deberta},
        {'params': other_params,   'lr': lr_base},
    ], weight_decay=config['training'].get('weight_decay', 0.01))

    # 学习率：Linear Warmup → Cosine Decay（到末尾自然降到 ~0，掐死后期过拟合空间）
    num_epochs    = config['training']['num_epochs']
    accum_steps   = config['training'].get('accum_steps', 1)
    steps_per_ep  = (len(train_loader) + accum_steps - 1) // accum_steps
    total_steps   = num_epochs * steps_per_ep
    warmup_steps  = int(total_steps * config['training'].get('warmup_ratio', 0.1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )
    print(f"  调度器: accum_steps={accum_steps} | warmup={warmup_steps} steps / total={total_steps} steps (cosine decay)")

    # 混合精度
    use_amp = (device.type == 'cuda')
    scaler  = GradScaler('cuda', enabled=use_amp)

    # 类别权重（在 NLI 类别空间计算，与 label 翻转保持一致）
    # NLI class 0 = entailment  = 支持  (dataset label=1, 少数类 → 权重更高)
    # NLI class 1 = not_entail  = 幻觉  (dataset label=0, 多数类 → 权重更低)
    labels_arr = train_ds.df['label'].values
    pos = int(np.sum(labels_arr == 1))   # 支持样本数 → NLI class 0
    neg = int(np.sum(labels_arr == 0))   # 幻觉样本数 → NLI class 1
    total = pos + neg
    w_nli0 = total / (2 * pos) if pos > 0 else 1.0   # NLI class 0 (支持/少数) 权重
    w_nli1 = total / (2 * neg) if neg > 0 else 1.0   # NLI class 1 (幻觉/多数) 权重
    class_weights = torch.tensor([w_nli0, w_nli1], dtype=torch.float32).to(device)
    print(f"  数据分布: 支持(1)={pos}, 幻觉(0)={neg} | NLI类别权重=[{w_nli0:.3f}(ent), {w_nli1:.3f}(not_ent)]")

    label_smoothing = float(config['training'].get('label_smoothing', 0.0))
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    print(f"  Label smoothing: {label_smoothing}")

    patience        = config['training']['patience']
    monitor_metric  = config['training'].get('monitor_metric', 'val_loss')   # 'val_loss' or 'val_f1'
    assert monitor_metric in ('val_loss', 'val_f1'), f"未知 monitor_metric={monitor_metric}"
    print(f"  监控指标: {monitor_metric}（'val_loss' 表示降低更好；'val_f1' 表示升高更好）")

    # 双 best 保存路径已在配置中指定
    best_val_loss   = float('inf')
    best_val_f1     = 0.0
    patience_cnt    = 0

    # ── 训练循环 ──────────────────────────────────────────────────────────
    print(f"\n[5/5] 开始训练（共 {num_epochs} epoch，早停耐心={patience}）\n")
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        total_loss, all_preds, all_labels = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{num_epochs}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)

            input_ids      = batch.input_ids
            attention_mask = batch.attention_mask
            token_type_ids = batch.token_type_ids
            labels         = batch.y.view(-1).long()

            with autocast('cuda', enabled=use_amp):
                logits = model(input_ids, attention_mask, token_type_ids, batch)
                labels_nli = 1 - labels   # dataset label 翻转为 NLI class index
                raw_loss = criterion(logits, labels_nli)
                loss   = raw_loss / accum_steps
            
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()   # cosine 调度按 step 推进

            total_loss += raw_loss.item() * labels.size(0)
            preds = (logits[:, 0] > logits[:, 1]).long().detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            running_acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({
                'loss': f'{raw_loss.item():.4f}',
                'acc':  f'{running_acc:.4f}',
            })

        n = len(all_labels)
        train_loss = total_loss / n
        train_acc  = accuracy_score(all_labels, all_preds)
        train_f1   = f1_score(all_labels, all_preds, average='binary', zero_division=0)

        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, device, criterion, use_amp)
        cur_lr_main    = optimizer.param_groups[1]['lr']
        cur_lr_deberta = optimizer.param_groups[0]['lr']

        # 过拟合警告：Train-Val Loss 差距 + Val Loss 上升趋势
        gap_loss = val_loss - train_loss
        gap_f1   = train_f1 - val_f1
        warn = ""
        if gap_loss > 0.4 or gap_f1 > 0.10:
            warn = "  ⚠️ 过拟合迹象"

        print(
            f"  Epoch {epoch:02d} | "
            f"Train L={train_loss:.4f} Acc={train_acc:.4f} F1={train_f1:.4f} | "
            f"Val L={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f} AUC={val_auc:.4f} | "
            f"LR(main/deberta)={cur_lr_main:.2e}/{cur_lr_deberta:.2e}{warn}"
        )

        improved_loss = val_loss < best_val_loss
        improved_f1   = val_f1   > best_val_f1

        if improved_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss, 'val_f1': val_f1,
                'val_acc':  val_acc,  'val_auc': val_auc,
                'config':   config,
            }, best_loss_path)
            print(f"  📉 best_loss 已更新 (Val Loss={best_val_loss:.4f}) -> {best_loss_path}")

        if improved_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss, 'val_f1': val_f1,
                'val_acc':  val_acc,  'val_auc': val_auc,
                'config':   config,
            }, best_f1_path)
            print(f"  📈 best_f1 已更新 (Val F1={best_val_f1:.4f}) -> {best_f1_path}")

        improved = improved_loss if monitor_metric == 'val_loss' else improved_f1
        if improved:
            patience_cnt = 0
        else:
            patience_cnt += 1
            print(f"  ⏳ 早停计数: {patience_cnt}/{patience} (基于 {monitor_metric})")
            if patience_cnt >= patience:
                print(f"\n⛔ 早停触发！最佳 Val Loss={best_val_loss:.4f}, Val F1={best_val_f1:.4f}")
                break

    print(f"\n训练完成！最佳 Val Loss={best_val_loss:.4f} | 最佳 Val F1={best_val_f1:.4f}")
    print(f"  推荐使用: {best_loss_path}（按 Val Loss 选出，泛化最好）")
    print(f"  参考使用: {best_f1_path}（按 Val F1 选出，可能轻度过拟合）")


if __name__ == '__main__':
    train()
