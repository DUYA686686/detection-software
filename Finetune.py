import os
import datetime
import time
import torch
import torch.nn.utils.prune as prune
import argparse

import transforms
from network_files import FasterRCNN, AnchorsGenerator
from my_dataset import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils
from backbone.model_swiftness import swiftnet


def finalize_pruning(model):
    """固化模型中剪枝后的各模块，仅处理已被剪枝的层"""
    # 固化 RPN head
    if hasattr(model, "rpn") and hasattr(model.rpn, "head"):
        rpn_head = model.rpn.head
        # 检查是否已被剪枝
        if prune.is_pruned(rpn_head.conv):
            prune.remove(rpn_head.conv, 'weight')
        if prune.is_pruned(rpn_head.cls_logits):
            prune.remove(rpn_head.cls_logits, 'weight')
        if prune.is_pruned(rpn_head.bbox_pred):
            prune.remove(rpn_head.bbox_pred, 'weight')

    # 固化 ROI head
    if hasattr(model, "roi_heads"):
        roi_heads = model.roi_heads
        if hasattr(roi_heads, "box_head"):
            if prune.is_pruned(roi_heads.box_head.fc6):
                prune.remove(roi_heads.box_head.fc6, 'weight')
            if prune.is_pruned(roi_heads.box_head.fc7):
                prune.remove(roi_heads.box_head.fc7, 'weight')
        if hasattr(roi_heads, "box_predictor"):
            if prune.is_pruned(roi_heads.box_predictor.cls_score):
                prune.remove(roi_heads.box_predictor.cls_score, 'weight')
            if prune.is_pruned(roi_heads.box_predictor.bbox_pred):
                prune.remove(roi_heads.box_predictor.bbox_pred, 'weight')

    print("剪枝模块已固化（仅处理已剪枝层）")
    return model


def finetune(parser_data):
    print("使用的 finetune checkpoint:", parser_data.finetune_checkpoint)
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    model = torch.load(parser_data.finetune_checkpoint, map_location='cpu')
    model.to(device)
    # 定义数据预处理和数据加载器（这里只示例训练和验证集）
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }
    VOC_root = parser_data.data_path
    train_dataset = VOCDataSet(VOC_root, "2012", data_transform["train"], "train.txt")
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=parser_data.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,  # 根据机器情况调整
        collate_fn=train_dataset.collate_fn
    )
    val_dataset = VOCDataSet(VOC_root, "2012", data_transform["val"], "val.txt")
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=val_dataset.collate_fn
    )

    # 定义优化器和学习率调度器
    params = [p for p in model.parameters() if p.requires_grad]
    finetune_lr = 0.005 / 5  # finetune 时进一步降低学习率（例如降低 5 倍）
    optimizer = torch.optim.Adam(params, lr=finetune_lr, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.33)
    scaler = torch.cuda.amp.GradScaler() if parser_data.amp else None

    # 开始 finetune 训练（finetune_epochs 表示微调轮次，可根据需要延长）
    train_loss = []
    val_map = []
    time_start = time.time()
    for epoch in range(parser_data.start_epoch, parser_data.finetune_epochs):
        mean_loss, lr = utils.train_one_epoch(
            model, optimizer, train_data_loader,
            device=device, epoch=epoch,
            print_freq=50, warmup=True,
            scaler=scaler
        )
        train_loss.append(mean_loss.item())
        lr_scheduler.step()

        # evaluate on validation set
        coco_info = utils.evaluate(model, val_data_loader, device=device)
        val_map.append(coco_info[1])
        print("Epoch {}: Loss = {:.4f}, mAP = {:.4f}".format(epoch, mean_loss.item(), coco_info[1]))

    time_end = time.time()
    print("Finetune 总耗时: {:.2f} 秒".format(time_end - time_start))

    # 在 finetune 完成后，固化剪枝（移除 weight_orig 等额外参数）
    model = finalize_pruning(model)

    # 保存固化后的最终模型，仅包含模型权重，便于推理对比
    final_model_path = os.path.join(parser_data.output_dir, "final_pruned_model.pth")
    torch.save(model, final_model_path)
    print("固化后的最终模型已保存至:", final_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Faster R-CNN with Pruning Finalization")
    parser.add_argument('--device', default='cuda:0', help='训练设备')
    parser.add_argument('--data-path', default='./', help='数据集根目录')
    parser.add_argument('--num-classes', default=6, type=int, help='检测目标类别数（不含背景）')
    parser.add_argument('--output_dir', default='./finetune_output0.9', help='模型保存目录')
    # 将 finetune-checkpoint 参数默认设置为 "model-11.pth"，如果没有指定则使用该默认值
    parser.add_argument('--finetune-checkpoint', type=str, default="IR7-EC_prune0.9/model-1.pth",
                        help='用于 finetune 的 checkpoint 文件路径')
    parser.add_argument('--batch_size', default=4, type=int, help='训练时的 batch size')
    parser.add_argument('--finetune_epochs', default=25, type=int, help='finetune 总轮次')
    parser.add_argument('--start_epoch', default=0, type=int, help='finetune 开始 epoch')
    parser.add_argument("--amp", action="store_true", help="是否使用混合精度训练")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    finetune(args)
