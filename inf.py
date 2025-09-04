# ==============================================================================
# inference_ablation.py (最终版 - 绝对无任何模块调用)
# ==============================================================================

import torch
import torch.nn as nn
import os
import random
import numpy as np
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchmetrics

# --- 依赖于你项目结构的部分 ---
try:
    from datasets2 import IQA_build_loader
    from config.config import load_config
    from models_blip.blip import build_agiqa_model
except ImportError as e:
    print(f"Error importing project modules: {e}")
    exit(1)


# ==============================================================================
# 写死的配置
# ==============================================================================
class HardcodedConfig:
    """一个包含所有写死配置的类"""
    CONFIG_PATH = '/home/data1/lq/SC-AGIQA-main/config/aigciqa2023.yaml'
    CHECKPOINT_PATH = '/home/data1/lq/SC-AGIQA-main/checkpoints_2023k/blip_model_weights0.9207311868667603.pth'
    
    # 最终输出文件的路径
    FINAL_OUTPUT_PATH = "/home/data1/lq/SC-AGIQA-main/moe_analysis_results.json"
    
    SEED = 1742174487 # 对应 aigciqa2023.yaml


# ==============================================================================
# 函数定义 (这部分不变)
# ==============================================================================

def setup_seed(seed_value):
    """根据给定的种子值设置随机性"""
    print(f"Using fixed seed: {seed_value}")
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(20)
    torch.cuda.manual_seed(20)
    torch.cuda.manual_seed_all(20)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("Seed setup complete.")

def initialize_model_for_inference(config, checkpoint_path, device):
    """加载配置，初始化模型，并加载预训练权重"""
    print("\nInitializing model for inference...")
    model = build_agiqa_model(config, device)
    print("Model structure built.")

    print(f"Loading weights from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"FATAL ERROR: Checkpoint file not found at {checkpoint_path}")
        exit(1)
    state_dict = torch.load(checkpoint_path, map_location=device)

    if list(state_dict.keys())[0].startswith('module.'):
        print("DataParallel 'module.' prefix detected. Stripping prefix...")
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Warning: Could not load state_dict directly. Error: {e}. Attempting with strict=False...")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    print("Model initialized and weights loaded successfully.")
    return model

def run_inference(model, data_loader, config, device, cfg: HardcodedConfig):
    """
    执行推理的核心函数。
    **本版本绝对不接触、不调用、不修改任何子模块。**
    """
    model.eval()

    # --- 步骤 1: 正常推理 ---
    print("\nStarting inference loop...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, consis in tqdm(data_loader, desc="Inference Progress"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images, consis)
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())
    print("Inference loop finished.")

    # --- 步骤 2: 推理脚本独立保存自己的数据 ---
    # (这会覆盖掉任何由MoERegressor模块写入的内容)
    print(f"\nSaving final scores and labels to: {cfg.FINAL_OUTPUT_PATH}")
    
    final_preds_list = torch.cat(all_preds, dim=0).squeeze().tolist()
    final_labels_list = torch.cat(all_labels, dim=0).squeeze().tolist()
    
    scores_data = {'predictions': final_preds_list, 'ground_truth': final_labels_list}
    
    try:
        os.makedirs(os.path.dirname(cfg.FINAL_OUTPUT_PATH), exist_ok=True)
        with open(cfg.FINAL_OUTPUT_PATH, 'w') as f:
            json.dump(scores_data, f, indent=4)
        print(f"Successfully saved scores and labels. File at '{cfg.FINAL_OUTPUT_PATH}' now contains this data.")
    except Exception as e:
        print(f"ERROR saving scores and labels to file: {e}")

    # --- 步骤 3: 计算并打印性能指标 ---
    print("\nCalculating performance metrics...")
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    if config.DATA.PATCH_NUM > 1:
        valid_len = (all_preds.shape[0] // config.DATA.PATCH_NUM) * config.DATA.PATCH_NUM
        all_preds = all_preds[:valid_len].view(-1, config.DATA.PATCH_NUM).mean(dim=1)
        all_labels = all_labels[:valid_len].view(-1, config.DATA.PATCH_NUM).mean(dim=1)
    all_preds, all_labels = all_preds.squeeze(), all_labels.squeeze()
    
    spearman_corr = torchmetrics.functional.spearman_corrcoef(all_preds.float(), all_labels.float()).item()
    pearson_corr = torchmetrics.functional.pearson_corrcoef(all_preds.float(), all_labels.float()).item()
    
    print("\n" + "="*50)
    print("--- INFERENCE RESULTS ---")
    print(f"  Spearman's Rank Correlation Coefficient (SRCC): {spearman_corr:.4f}")
    print(f"  Pearson's Linear Correlation Coefficient (PLCC): {pearson_corr:.4f}")
    print("="*50)


# ==============================================================================
# 主执行入口
# ==============================================================================

def main():
    """主函数，使用写死的配置启动推理流程。"""
    cfg = HardcodedConfig()
    
    print("--- Starting AGIQA Inference Script (Zero-Call Version) ---")
    print(f"Config file: {cfg.CONFIG_PATH}")
    print(f"Checkpoint file: {cfg.CHECKPOINT_PATH}")
    
    setup_seed(cfg.SEED)
    
    config = load_config(cfg.CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = initialize_model_for_inference(config, cfg.CHECKPOINT_PATH, device)
    
    print("\nLoading data...")
    _, _, _, data_loader_val = IQA_build_loader(config)
    print(f"Data loaded. Validation set has {len(data_loader_val.dataset)} samples.")
    
    # 运行推理
    run_inference(model, data_loader_val, config, device, cfg)
    
    print("\n--- Script finished. ---")


if __name__ == "__main__":
    main()