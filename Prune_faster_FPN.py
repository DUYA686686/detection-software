import os
import datetime
import time
import torch

import transforms
from network_files import FasterRCNN, AnchorsGenerator
from my_dataset import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils
from backbone import BackboneWithFPN, LastLevelMaxPool
from backbone.model_swiftness import swiftnet
import torch.nn as nn


def create_model(num_classes):
    import torchvision
    from torchvision.models.feature_extraction import create_feature_extractor
    #######################################################################################
    # fpn backbone #
    backbone = swiftnet(num_classes)
    # print(backbone)
    return_layers = {"features.2": "0",  # stride 8
                     "features.4": "1",  # stride 16
                     "features.6": "2"}  # stride 32

    # 提供给fpn的每个特征层channel
    in_channels_list = [24, 40, 96]
    new_backbone = create_feature_extractor(backbone, return_layers)
    # 添加特征层输出验证
    # img = torch.randn(1, 3, 224, 224)
    # outputs = new_backbone(img)
    # for k, v in outputs.items():
    #     print(f"{k} shape: {v.shape}")
    #######################################################################################
    backbone_with_fpn = BackboneWithFPN(new_backbone,
                                        return_layers=return_layers,
                                        in_channels_list=in_channels_list,
                                        out_channels=128,  # 调整为更小的值，避免维度冲突
                                        extra_blocks=LastLevelMaxPool(),
                                        re_getter=False)

    anchor_sizes = ((64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorsGenerator(sizes=anchor_sizes,
                                        aspect_ratios=aspect_ratios)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2'],  # 在哪些特征层上进行RoIAlign pooling
                                                    output_size=[7, 7],  # RoIAlign pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone_with_fpn,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    # 载入预训练模型权重
    weights_dict = torch.load("FPN_save_weights/model-29.pth", map_location='cpu')
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

    # ---- 剪枝 RPN Head（适配FPN的多层RPN）----
    if hasattr(model.rpn, "heads"):
        # 遍历FPN中每个特征层的RPN Head
        for rpn_head in model.rpn.heads:
            # 1. 剪枝RPN的共享卷积层
            rpn_head.conv = _prune_conv_layer(rpn_head.conv, prune_amount)
            new_conv_out = rpn_head.conv.out_channels

            # 2. 调整分类和回归层的输入通道
            rpn_head.cls_logits = prune_conv_layer_input(rpn_head.cls_logits, new_conv_out)
            rpn_head.bbox_pred = prune_conv_layer_input(rpn_head.bbox_pred, new_conv_out)

    # ---- 剪枝 ROI Head（适配FPN的池化与全连接）----
    if hasattr(model.roi_heads, "box_head"):
        box_head = model.roi_heads.box_head
        device = next(model.parameters()).device
        # 新增导入语句
        from network_files.faster_rcnn_framework import TwoMLPHead  # 根据实际模块路径调整
        # 剪枝全连接层（假设使用TwoMLPHead结构）
        if isinstance(box_head, TwoMLPHead):
            # 剪枝fc6
            box_head.fc6 = _prune_linear_layer(box_head.fc6, prune_amount).to(device)
            new_fc6_out = box_head.fc6.out_features

            # 剪枝fc7的输入并再次剪枝输出
            box_head.fc7 = prune_linear_layer_input(box_head.fc7, new_fc6_out).to(device)
            box_head.fc7 = _prune_linear_layer(box_head.fc7, prune_amount).to(device)
            new_fc7_out = box_head.fc7.out_features

            # 调整box_predictor的输入维度
            if hasattr(model.roi_heads, "box_predictor"):
                num_classes = model.roi_heads.box_predictor.cls_score.out_features
                model.roi_heads.box_predictor.cls_score = nn.Linear(new_fc7_out, num_classes).to(device)
                model.roi_heads.box_predictor.bbox_pred = nn.Linear(new_fc7_out, num_classes * 4).to(device)

    print(f"剪枝完成，剪枝比例：{prune_amount * 100}%")
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = args.data_path
    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # load train data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    train_dataset = VOCDataSet(VOC_root, "2012", data_transform["train"], "train.txt")
    train_sampler = None

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减小训练时所需GPU显存，默认使用
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图像高宽比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # 每个batch图片从同一高宽比例区间中取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    if train_sampler:
        # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
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

    # load validation data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_dataset = VOCDataSet(VOC_root, "2012", data_transform["val"], "val.txt")
    val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=nw,
                                                      collate_fn=val_dataset.collate_fn)

    # create model num_classes equal background + 20 classes
    model = create_model(num_classes=args.num_classes + 1)
    # print(model)
    # ms = open('model.txt', 'w')
    # ms.write(str(model))
    # ms.close()
    # print(model)
    model.to(device)
    model = apply_pruning(model, prune_amount=args.prune_ratio)  # 新增剪枝调用
    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005,
    #                             momentum=0.9, weight_decay=0.0005)

    #
    optimizer = torch.optim.Adam(params, lr=0.005, weight_decay=0.0005)
    #
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.33)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []
    time_start = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 10 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True,
                                              scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        coco_info = utils.evaluate(model, val_data_set_loader, device=device)

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP
        from Finetune import finalize_pruning
        finalize_pruning(model)
        torch.save(model, os.path.join(args.output_dir, "model-{}.pth".format(epoch)))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)
    ######################输出train loss，learninging rate, val map##################################
    tl = open("train_loss.txt", "a")
    tl.write(str(train_loss) + "\n")
    tl.close()

    lr = open("learning_rate.txt", "a")
    lr.write(str(learning_rate) + "\n")
    lr.close()

    v_map = open("val_map.txt", "a")
    v_map.write(str(val_map) + "\n")
    v_map.close()
    ######################输出train loss，learninging rate, val map##################################
    # time
    time_end = time.time()
    time_sum = time_end - time_start
    print(time_sum)
    tt = open('time.txt', 'w')
    tt.write(str(time_sum))
    tt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--prune_ratio', default=0.7, type=int, help='剪枝比例')
    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录(VOCdevkit)
    parser.add_argument('--data-path', default='./', help='dataset')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=6, type=int, help='num_classes')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./FPN_prune_save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    # 训练的batch size
    parser.add_argument('--batch_size', default=8, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
