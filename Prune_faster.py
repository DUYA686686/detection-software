import os
import datetime
import time
import torch
import torch.nn as nn
import transforms
from Finetune import finalize_pruning
from network_files import FasterRCNN, AnchorsGenerator
from my_dataset import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils
from backbone.model_v3 import mobilenet_v3_large, mobilenet_v3_small, mobilenet_v3_mini
from backbone.model_swiftness import swiftnet
from backbone.model_cracknet2 import cracknet
from backbone.ir15_ir6 import ir15, ir10, ir9, ir8, ir7, ir6
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor


def create_model(num_classes):
    #######################################################################################################################
    # 使用 IR7-EC 结构
    backbone = swiftnet(num_classes)
    # 使用 torchvision 的 create_feature_extractor 提取指定层（下采样16倍层）
    backbone = create_feature_extractor(backbone, return_nodes={"cbam": "0"})
    backbone.out_channels = 96
    #######################################################################################################################
    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=[7, 7],
                                                    sampling_ratio=2)
    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    # 载入预训练模型权重
    weights_dict = torch.load("backbone/faster R-IR7-EC.pth", map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    return model


def apply_pruning(model, prune_amount=0.5):
    """
    对 Faster R-CNN 模型进行结构化剪枝，物理移除低重要性通道，并调整模型结构
    """

    def _prune_conv_layer(layer, prune_amount, in_channels=None):
        """剪枝卷积层（按输出通道剪枝）并重建新层"""
        if not isinstance(layer, nn.Conv2d):
            return layer
        original_in_channels = in_channels if in_channels is not None else layer.in_channels
        # 对每个输出通道计算 L1 范数
        importance = torch.norm(layer.weight.data, p=1, dim=(1, 2, 3))
        num_kept = int(layer.out_channels * (1 - prune_amount))
        _, kept_indices = torch.topk(importance, k=num_kept, largest=True)
        new_conv = nn.Conv2d(
            in_channels=original_in_channels,
            out_channels=num_kept,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            bias=(layer.bias is not None)
        )
        new_conv.weight.data = layer.weight.data[kept_indices].clone()
        if layer.bias is not None:
            new_conv.bias.data = layer.bias.data[kept_indices].clone()
        return new_conv

    def _prune_linear_layer(layer, prune_amount, in_features=None):
        """
        剪枝全连接层（按输出神经元剪枝）并重建新层，
        若指定 in_features 则覆盖原值
        """
        if not isinstance(layer, nn.Linear):
            return layer
        importance = torch.norm(layer.weight.data, p=1, dim=1)
        num_kept = int(layer.out_features * (1 - prune_amount))
        _, kept_indices = torch.topk(importance, k=num_kept, largest=True)
        new_fc = nn.Linear(
            in_features=layer.in_features if in_features is None else in_features,
            out_features=num_kept,
            bias=(layer.bias is not None)
        )
        # 注意：这里只剪枝输出维度（行方向）
        new_fc.weight.data = layer.weight.data[kept_indices].clone()
        if layer.bias is not None:
            new_fc.bias.data = layer.bias.data[kept_indices].clone()
        return new_fc

    def prune_conv_layer_input(layer, new_in_channels):
        """
        对卷积层进行输入通道剪枝：
        调整输入通道数为 new_in_channels，同时保持输出通道不变
        """
        if not isinstance(layer, nn.Conv2d):
            return layer
        importance = torch.norm(layer.weight.data, p=1, dim=(0, 2, 3))
        _, kept_indices = torch.topk(importance, k=new_in_channels, largest=True)
        new_conv = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            bias=(layer.bias is not None)
        )
        new_conv.weight.data = layer.weight.data[:, kept_indices].clone()
        if layer.bias is not None:
            new_conv.bias.data = layer.bias.data.clone()
        return new_conv

    def prune_linear_layer_input(layer, new_in_features):
        """
        对全连接层进行输入剪枝：
        将 in_features 调整为 new_in_features，同时保持输出神经元数不变
        """
        if not isinstance(layer, nn.Linear):
            return layer
        importance = torch.norm(layer.weight.data, p=1, dim=0)  # 针对每个输入特征计算重要性
        _, kept_indices = torch.topk(importance, k=new_in_features, largest=True)
        new_linear = nn.Linear(
            in_features=new_in_features,
            out_features=layer.out_features,
            bias=(layer.bias is not None)
        )
        new_linear.weight.data = layer.weight.data[:, kept_indices].clone()
        if layer.bias is not None:
            new_linear.bias.data = layer.bias.data.clone()
        return new_linear

    # ---- 剪枝 RPN Head ----
    if hasattr(model, "rpn") and hasattr(model.rpn, "head"):
        rpn_head = model.rpn.head
        # 1. 对 rpn_head.conv 按输出通道剪枝
        rpn_head.conv = _prune_conv_layer(rpn_head.conv, prune_amount)
        new_conv_out_channels = rpn_head.conv.out_channels

        # 2. 对 cls_logits 层按输入通道剪枝，调整为 new_conv_out_channels
        rpn_head.cls_logits = prune_conv_layer_input(rpn_head.cls_logits, new_conv_out_channels)
        # 3. 对 bbox_pred 层做同样处理
        rpn_head.bbox_pred = prune_conv_layer_input(rpn_head.bbox_pred, new_conv_out_channels)

        # ---- 剪枝 ROI Head ----
        if hasattr(model, "roi_heads"):
            roi_heads = model.roi_heads
            if hasattr(roi_heads, "box_head"):
                # 获取当前模型所在的设备
                device = next(model.parameters()).device

                # 剪枝 fc6（按输出神经元剪枝）
                roi_heads.box_head.fc6 = _prune_linear_layer(roi_heads.box_head.fc6, prune_amount).to(device)
                new_fc6_out_features = roi_heads.box_head.fc6.out_features

                # 剪枝 fc7：输入调整为fc6的输出，输出再剪枝
                roi_heads.box_head.fc7 = prune_linear_layer_input(roi_heads.box_head.fc7,
                                                                  new_in_features=new_fc6_out_features).to(device)
                roi_heads.box_head.fc7 = _prune_linear_layer(roi_heads.box_head.fc7, prune_amount,
                                                             in_features=new_fc6_out_features).to(device)
                new_fc7_out_features = roi_heads.box_head.fc7.out_features

                # 同步调整box_predictor的输入维度，并确保新层在GPU上
                if hasattr(roi_heads, "box_predictor"):
                    num_classes = roi_heads.box_predictor.cls_score.out_features
                    # 重建cls_score层并迁移到GPU
                    roi_heads.box_predictor.cls_score = nn.Linear(
                        new_fc7_out_features, num_classes
                    ).to(device)
                    # 重建bbox_pred层并迁移到GPU
                    roi_heads.box_predictor.bbox_pred = nn.Linear(
                        new_fc7_out_features, num_classes * 4
                    ).to(device)

    print(f"剪枝完成，剪枝比例：{prune_amount * 100}%")
    return model


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }
    VOC_root = parser_data.data_path
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))
    train_dataset = VOCDataSet(VOC_root, "2012", data_transform["train"], "train.txt")
    train_sampler = None
    if parser_data.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        group_ids = create_aspect_ratio_groups(train_dataset, k=parser_data.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, parser_data.batch_size)
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using %g dataloader workers' % nw)
    if train_sampler:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    val_dataset = VOCDataSet(VOC_root, "2012", data_transform["val"], "val.txt")
    val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=nw,
                                                      collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=parser_data.num_classes + 1)
    model.to(device)
    # --- 对模型中 RPN head 和 ROI head 进行剪枝 ---
    # prune_ratio = 0.7  # 剪枝比例，可根据需要调整
    model = apply_pruning(model, prune_amount=parser_data.prune_ratio)
    # 验证 RPN 各层维度
    print("RPN卷积层输出通道:", model.rpn.head.conv.out_channels)
    print("cls_logits输入通道:", model.rpn.head.cls_logits.in_channels)
    print("cls_logits权重形状:", model.rpn.head.cls_logits.weight.shape)
    print("bbox_pred输入通道:", model.rpn.head.bbox_pred.in_channels)
    print("bbox_pred权重形状:", model.rpn.head.bbox_pred.weight.shape)
    pruned_model_path = os.path.join(parser_data.output_dir, "pruned_model.pth")
    torch.save(model, pruned_model_path)
    print("剪枝后的模型已保存至:", pruned_model_path)
    params = [p for p in model.parameters() if p.requires_grad]
    finetune_lr = 0.005 / 3
    optimizer = torch.optim.Adam(params, lr=finetune_lr, weight_decay=0.0005)
    scaler = torch.cuda.amp.GradScaler() if parser_data.amp else None
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.33)
    if parser_data.resume != "":
        checkpoint = torch.load(parser_data.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        parser_data.start_epoch = checkpoint['epoch'] + 1
        if parser_data.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(parser_data.start_epoch))
    train_loss = []
    learning_rate = []
    val_map = []
    time_start = time.time()
    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True,
                                              scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)
        lr_scheduler.step()
        coco_info = utils.evaluate(model, val_data_set_loader, device=device)
        with open(results_file, "a") as f:
            result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")
        val_map.append(coco_info[1])
        finalize_pruning(model)
        torch.save(model, os.path.join(parser_data.output_dir, "model-{}.pth".format(epoch)))
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)
    with open("train_loss.txt", "a") as tl:
        tl.write(str(train_loss) + "\n")
    with open("learning_rate.txt", "a") as lr_file:
        lr_file.write(str(learning_rate) + "\n")
    with open("val_map.txt", "a") as v_map:
        v_map.write(str(val_map) + "\n")
    time_end = time.time()
    time_sum = time_end - time_start
    print(time_sum)
    with open('time.txt', 'w') as tt:
        tt.write(str(time_sum))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--data-path', default='./', help='dataset')
    parser.add_argument('--num-classes', default=6, type=int, help='num_classes')
    parser.add_argument('--output-dir', default='./IR7-EC_prune0.9', help='path where to save')
    parser.add_argument('--prune_ratio', default=0.9, type=int, help='剪枝比例')

    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)
