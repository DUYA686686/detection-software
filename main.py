import cv2

from login import *
from Interface import Ui_MainWindow as Ui_MainWindow_inter
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog
# from PyQt5 import QtWidgets
from PyQt5 import QtCore, QtGui
import os
import time
import torch

import transforms
from network_files import FasterRCNN, AnchorsGenerator
from my_dataset import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils
from backbone.model_swiftness import swiftnet
import datetime
import matplotlib.pyplot as plt

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMessageBox, QFileDialog, QMainWindow
from child import Ui_Dialog as view
import os
import cv2


class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 无边框
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.ui.pushButton.clicked.connect(self.change_widget2)
        self.ui.pushButton_2.clicked.connect(self.change_widget3)
        self.ui.widget_3.hide()
        self.ui.pushButton_3.clicked.connect(self.go_to_inter)
        # self.show()

    def change_widget3(self):
        self.ui.widget_2.hide()
        self.ui.widget_3.show()

    def change_widget2(self):
        self.ui.widget_3.hide()
        self.ui.widget_2.show()

    def go_to_inter(self):
        account = self.ui.lineEdit.text()
        password = self.ui.lineEdit_2.text()
        if account == 'root' and password == '123':
            interface = InterfaceWindow()
            interface.show()
            self.close()
        else:
            pass

    # #  拖动
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.isMaximized() == False:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))  # 更改鼠标图案

    def mouseMoveEvent(self, mouse_event):
        if QtCore.Qt.LeftButton and self.m_flag:
            self.move(mouse_event.globalPos() - self.m_Position)  # 更改窗口位置
            mouse_event.accept()

    def mouseReleaseEvent(self, mouse_event):
        self.m_flag = False
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))


# 首页界面

class InterfaceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow_inter()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.pushButton_training.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.pushButton_detection.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        self.ui.pushButton_3.clicked.connect(self.resize_win)
        #
        self.view = Select()
        self.ui.pushButton_10.clicked.connect(self.gotoview)
        self.ui.pushButton_gotoweight.clicked.connect(self.gotofile)
        # training mode interface
        # 开始训练、开始验证
        self.ui.pushButton_6.clicked.connect(self.start_train)
        self.ui.pushButton_8.clicked.connect(self.start_val)
        # input
        self.ui.pushButton_data.clicked.connect(self.input_train_data)
        self.ui.pushButton_output.clicked.connect(self.output_weight_path)
        self.ui.pushButton_11.clicked.connect(self.textbrowser_clear_coco)
        self.ui.pushButton_map.clicked.connect(self.show_map)
        self.ui.pushButton_lrl.clicked.connect(self.show_lrl)

        # detection mode interface
        self.ui.pushButton_output_3.clicked.connect(self.load_weight_2)
        self.ui.pushButton_detect.clicked.connect(self.start_detect)
        self.ui.pushButton_data_2.clicked.connect(self.input_detect_data)
        self.ui.pushButton_output_2.clicked.connect(self.outputdir_path)
        # self.show()

    ####################################################################
    def gotoview(self):
        self.view.show()
    def resize_win(self):
        if self.isMaximized():
            self.showNormal()
            self.ui.pushButton_3.setIcon(QtGui.QIcon("./icons/maximize.png"))
        else:
            self.showMaximized()
            self.ui.pushButton_3.setIcon(QtGui.QIcon("./icons/MINIMIZE.png"))

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.isMaximized() == False:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))  # 更改鼠标图案

    def mouseMoveEvent(self, mouse_event):
        if QtCore.Qt.LeftButton and self.m_flag:
            self.move(mouse_event.globalPos() - self.m_Position)  # 更改窗口位置
            mouse_event.accept()

    def mouseReleaseEvent(self, mouse_event):
        self.m_flag = False
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

    def gotofile(self):
        folder = r'.'
        os.startfile(folder)
    ##############################################################

    def input_train_data(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, caption="Open file", directory="./")
        self.ui.lineEdit_datapath.setText(path)

    def output_weight_path(self):
        path2 = QtWidgets.QFileDialog.getExistingDirectory(self, caption="Save weights", directory="./")
        self.ui.lineEdit_outputdir.setText(path2)

    def input_detect_data(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, caption="Open file", directory="./")
        self.ui.lineEdit_datapath_2.setText(path)

    def outputdir_path(self):
        path2 = QtWidgets.QFileDialog.getExistingDirectory(self, caption="Detected path", directory="./")
        self.ui.lineEdit_outputdir_2.setText(path2)

    def load_weight_2(self):
        weight_path = QtWidgets.QFileDialog.getOpenFileName(self, 'choose weight', './')
        weight_path = weight_path[0]
        self.ui.lineEdit_outputdir_3.setText(weight_path)

    def textbrowser_clear_coco(self):
        self.ui.textBrowser.clear()

    def plot_loss_and_lr(self, train_loss, learning_rate):
        try:
            x = list(range(len(train_loss)))
            fig, ax1 = plt.subplots(1, 1)
            ax1.plot(x, train_loss, 'r', label='loss')
            ax1.set_xlabel("step")
            ax1.set_ylabel("loss")
            ax1.set_title("Train Loss and lr")
            plt.legend(loc='best')

            ax2 = ax1.twinx()
            ax2.plot(x, learning_rate, label='lr')
            ax2.set_ylabel("learning rate")
            ax2.set_xlim(0, len(train_loss))  # 设置横坐标整数间隔
            plt.legend(loc='best')
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
            fig.subplots_adjust(right=0.8)  # 防止出现保存图片显示不全的情况
            fig.savefig('./loss_and_lr.png')
            plt.close()
            print("successful save loss curve! ")
        except Exception as e:
            print(e)

    def plot_map(self, mAP):
        try:
            x = list(range(len(mAP)))
            plt.plot(x, mAP, label='mAp')
            plt.xlabel('epoch')
            plt.ylabel('mAP')
            plt.title('Eval mAP')
            plt.xlim(0, len(mAP))
            plt.legend(loc='best')

            plt.savefig('./mAP.png')
            plt.close()
            print("successful save mAP curve!")
        except Exception as e:
            print(e)

    def show_map(self):
        imgName_cv2 = cv2.imread('./mAP.png')
        imgName_cv2 = cv2.resize(imgName_cv2, dsize=(400, 300))
        im0 = cv2.cvtColor(imgName_cv2, cv2.COLOR_RGB2BGR)
        # 这里使用cv2把这张图片读取进来，也可以用QtGui.QPixmap方式。然后由于cv2读取的跟等下显示的RGB顺序不一样，所以需要转换一下顺序
        showImage = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1], QtGui.QImage.Format_RGB888)
        self.ui.label_map.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def show_lrl(self):
        imgName_cv2 = cv2.imread('./loss_and_lr.png')
        imgName_cv2 = cv2.resize(imgName_cv2, dsize=(400, 300))
        im0 = cv2.cvtColor(imgName_cv2, cv2.COLOR_RGB2BGR)
        # 这里使用cv2把这张图片读取进来，也可以用QtGui.QPixmap方式。然后由于cv2读取的跟等下显示的RGB顺序不一样，所以需要转换一下顺序
        showImage = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1], QtGui.QImage.Format_RGB888)
        self.ui.label_lrl.setPixmap(QtGui.QPixmap.fromImage(showImage))

    #######################################################################
    def create_model(self, num_classes):
        import torchvision
        from torchvision.models.feature_extraction import create_feature_extractor
        #######################################################################################################################
        # # IR7-EC
        backbone = swiftnet(num_classes)
        # print(backbone)
        backbone = create_feature_extractor(backbone, return_nodes={"cbam": "0"})  # 下采样16倍的层做为接口
        # out = backbone(torch.rand(1, 3, 224, 224))
        # print(out["0"].shape)
        backbone.out_channels = 96
        # # cracknet
        # backbone = cracknet(num_classes)
        # # print(backbone)
        # backbone = create_feature_extractor(backbone, return_nodes={"fea3": "0"})  # 下采样16倍的层做为接口
        # # out = backbone(torch.rand(1, 3, 224, 224))
        # # print(out["0"].shape)
        # backbone.out_channels = 96
        # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256),),
        #                                     aspect_ratios=((0.5, 1.0, 2.0),))
        anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                            aspect_ratios=((0.5, 1.0, 2.0),))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行RoIAlign pooling
                                                        output_size=[7, 7],  # RoIAlign pooling输出特征矩阵尺寸
                                                        sampling_ratio=2)  # 采样率

        model = FasterRCNN(backbone=backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)
        return model

    def train(self, parser_data):
        device = torch.device(parser_data.device)
        # ui_print.print_(self, text="Using {} device training.".format(device.type))
        print("Using {} device training.".format(device.type))
        #
        # self.ui.textBrowser.append("Using {} device training.".format(device.type))
        # QtWidgets.QApplication.processEvents()  # 加上这个功能，不然有卡顿
        #
        # 用来保存coco_info的文件
        results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        data_transform = {
            "train": transforms.Compose([transforms.ToTensor(),
                                         transforms.RandomHorizontalFlip(0.5)]),
            "val": transforms.Compose([transforms.ToTensor()])
        }

        VOC_root = parser_data.data_path
        # check voc root
        if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
            raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

        # load train data set
        # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
        train_dataset = VOCDataSet(VOC_root, "2012", data_transform["train"], "train.txt")
        train_sampler = None

        # 是否按图片相似高宽比采样图片组成batch
        # 使用的话能够减小训练时所需GPU显存，默认使用
        if parser_data.aspect_ratio_group_factor >= 0:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
            # 统计所有图像高宽比例在bins区间中的位置索引
            group_ids = create_aspect_ratio_groups(train_dataset, k=parser_data.aspect_ratio_group_factor)
            # 每个batch图片从同一高宽比例区间中取
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, parser_data.batch_size)

        # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
        batch_size = parser_data.batch_size
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
        model = self.create_model(num_classes=parser_data.num_classes + 1)
        # print(model)

        model.to(device)

        # define optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        # optimizer = torch.optim.SGD(params, lr=0.005,
        #                             momentum=0.9, weight_decay=0.0005)

        #
        optimizer = torch.optim.Adam(params, lr=0.005, weight_decay=0.0005)
        #
        scaler = torch.cuda.amp.GradScaler() if parser_data.amp else None

        # learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.33)

        # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
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

        for epoch in range(parser_data.start_epoch, parser_data.epochs):
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
                result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            val_map.append(coco_info[1])  # pascal mAP

            # save weights
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            if parser_data.amp:
                save_files["scaler"] = scaler.state_dict()
            torch.save(save_files, parser_data.output_dir + "/model-{}.pth".format(epoch))

        # plot loss and lr curve
        if len(train_loss) != 0 and len(learning_rate) != 0:
            self.plot_loss_and_lr(train_loss, learning_rate)

        # plot mAP curve
        if len(val_map) != 0:
            self.plot_map(val_map)

        ######################输出train loss，learninging rate, val map##################################
        # tl = open("train_loss.txt", "a")
        # tl.write(str(train_loss) + "\n")
        # tl.close()
        #
        # lr = open("learning_rate.txt", "a")
        # lr.write(str(learning_rate) + "\n")
        # lr.close()
        #
        # v_map = open("val_map.txt", "a")
        # v_map.write(str(val_map) + "\n")
        # v_map.close()
        ######################输出train loss，learninging rate, val map##################################

    def start_train(self):
        import argparse
        #####################################
        # parser = argparse.ArgumentParser(
        #     description=__doc__)
        # # 训练设备类型
        # parser.add_argument('--device', default='cuda:0', help='device')
        # # 训练数据集的根目录(VOCdevkit)
        # parser.add_argument('--data-path', default='./', help='dataset')
        # # 检测目标类别数(不包含背景)
        # parser.add_argument('--num-classes', default=6, type=int, help='num_classes')
        # # 文件保存地址
        # parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
        # # 若需要接着上次训练，则指定上次训练保存权重文件地址
        # parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
        # # 指定接着从哪个epoch数开始训练
        # parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
        # # 训练的总epoch数
        # parser.add_argument('--epochs', default=20, type=int, metavar='N',
        #                     help='number of total epochs to run')
        # # 训练的batch size
        # parser.add_argument('--batch_size', default=4, type=int, metavar='N',
        #                     help='batch size when training.')
        # parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
        # # 是否使用混合精度训练(需要GPU支持混合精度)
        # parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")
        #
        # args = parser.parse_args()
        # print(args)
        ############################################################################################
        device = 'cuda:0' if self.ui.checkBox.isChecked() else 'cpu'
        data_path = self.ui.lineEdit_datapath.text()
        num_classes = self.ui.spinBox_numclass.text()
        output_dir = self.ui.lineEdit_outputdir.text()
        epoch = self.ui.spinBox_epoch.text()
        batch_size = self.ui.spinBox_batchsize.text()
        #
        parser = argparse.ArgumentParser(
            description=__doc__)

        # 训练设备类型
        parser.add_argument('--device', default=device, help='device')
        # 训练数据集的根目录(VOCdevkit)
        parser.add_argument('--data-path', default=data_path, help='dataset')  # ./'
        # 检测目标类别数(不包含背景)
        parser.add_argument('--num-classes', default=num_classes, type=int, help='num_classes')
        # 文件保存地址
        parser.add_argument('--output-dir', default=output_dir, help='path where to save')
        # 若需要接着上次训练，则指定上次训练保存权重文件地址
        parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
        # 指定接着从哪个epoch数开始训练
        parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
        # 训练的总epoch数
        parser.add_argument('--epochs', default=epoch, type=int, metavar='N',
                            help='number of total epochs to run')
        # 训练的batch size
        parser.add_argument('--batch_size', default=batch_size, type=int, metavar='N',
                            help='batch size when training.')
        parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
        # 是否使用混合精度训练(需要GPU支持混合精度)
        parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

        args = parser.parse_args()
        print(args)
        ############################################################################################################
        # 检查保存权重文件夹是否存在，不存在则创建
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        self.train(args)

    def validate(self, parser_data):
        import json
        from train_utils import get_coco_api_from_dataset, CocoEvaluator
        from tqdm import tqdm
        from summary import summarize
        device = torch.device(parser_data.device)
        print("Using {} device training.".format(device.type))
        self.ui.textBrowser.append("Using {} device training.".format(device.type))
        QtWidgets.QApplication.processEvents()  # 加上这个功能，不然有卡顿

        data_transform = {
            "val": transforms.Compose([transforms.ToTensor()])
        }

        # read class_indict
        label_json_path = './pascal_voc_classes.json'
        assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
        json_file = open(label_json_path, 'r')

        class_dict = json.load(json_file)
        json_file.close()
        category_index = {v: k for k, v in class_dict.items()}

        VOC_root = parser_data.data_path
        # check voc root
        if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
            raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

        # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
        batch_size = parser_data.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        # print('Using %g dataloader workers' % nw)

        # load validation data set
        val_dataset = VOCDataSet(VOC_root, "2012", data_transform["val"], "val.txt")
        val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                         batch_size=1,
                                                         shuffle=False,
                                                         num_workers=nw,
                                                         pin_memory=True,
                                                         collate_fn=val_dataset.collate_fn)

        # create model num_classes equal background + 20 classes
        # 注意，这里的norm_layer要和训练脚本中保持一致
        # backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
        # model = FasterRCNN(backbone=backbone, num_classes=parser_data.num_classes + 1)
        model = self.create_model(num_classes=parser_data.num_classes + 1)
        # print(model)

        model.to(device)
        # 载入你自己训练好的模型权重
        weights_path = parser_data.weights
        assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
        # print(model)

        model.to(device)

        # evaluate on the test dataset
        coco = get_coco_api_from_dataset(val_dataset)
        iou_types = ["bbox"]
        coco_evaluator = CocoEvaluator(coco, iou_types)  # coco_gt为coco
        cpu_device = torch.device("cpu")

        model.eval()
        # time_start = time.time()
        with torch.no_grad():
            for image, targets in tqdm(val_dataset_loader, desc="validation..."):
                # 将图片传入指定设备device
                image = list(img.to(device) for img in image)
                # time_start = time.time()
                # inference
                outputs = model(image)

                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                # time_end = time.time()
                # model_time = time_end - time_start
                # print(1 / model_time)
                res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
                coco_evaluator.update(res)

        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        # #
        # pr_array = coco_evaluator.summarize.eval['precision'][0, :, 0, 0, 2]
        # x = np.arange(0.0, 1.01, 0.01)
        # plt.xlabel('recall')
        # plt.ylabel('precision')
        # plt.xlim(0, 1.0)
        # plt.ylim(0, 1.01)
        #
        # plt.plot(x, pr_array, 'b-', label='IOU=0.5')
        #
        # plt.legend(loc='lower left')
        # plt.show()
        # #

        coco_eval = coco_evaluator.coco_eval["bbox"]
        # calculate COCO info for all classes

        coco_stats, print_coco = summarize(coco_eval)

        # calculate voc info for every classes(IoU=0.5)
        voc_map_info_list = []
        for i in range(len(category_index)):
            stats, _ = summarize(coco_eval, catId=i)
            voc_map_info_list.append(" {:15}: {}".format(category_index[i + 1], stats[1]))

        print_voc = "\n".join(voc_map_info_list)
        print(print_voc)

        # 将验证结果保存至txt文件中
        with open("record_mAP.txt", "w") as f:
            record_lines = ["COCO results:",
                            print_coco,
                            "",
                            "mAP(IoU=0.5) for each category:",
                            print_voc]
            f.write("\n".join(record_lines))
            self.ui.textBrowser.append("\n".join(record_lines))
            QtWidgets.QApplication.processEvents()  # 加上这个功能，不然有卡顿

    def start_val(self):
        import argparse

        parser = argparse.ArgumentParser(
            description=__doc__)

        # # 使用设备类型
        # parser.add_argument('--device', default='cuda', help='device')
        #
        # # 检测目标类别数
        # parser.add_argument('--num-classes', type=int, default='6', help='number of classes')
        #
        # # 数据集的根目录(VOCdevkit)
        # parser.add_argument('--data-path', default='./', help='dataset root')
        #
        # # 训练好的权重文件
        # parser.add_argument('--weights', default='./backbone/Final_stair.pth', type=str, help='training weights')
        #
        # # batch size
        # parser.add_argument('--batch_size', default=4, type=int, metavar='N',
        #                     help='batch size when validation.')

        ############################################################################################
        device = 'cuda:0' if self.ui.checkBox_2.isChecked() else 'cpu'
        data_path = self.ui.lineEdit_datapath_2.text()
        num_classes = self.ui.spinBox_numclass_2.text()
        batch_size = self.ui.spinBox_batchsize_2.text()
        weights = self.ui.lineEdit_outputdir_3.text()
        # # #
        # 使用设备类型
        parser.add_argument('--device', default=device, help='device')

        # 检测目标类别数
        parser.add_argument('--num-classes', type=int, default=num_classes, help='number of classes')

        # 数据集的根目录(VOCdevkit)
        parser.add_argument('--data-path', default=data_path, help='dataset root')

        # 训练好的权重文件
        parser.add_argument('--weights', default=weights, type=str, help='training weights')

        # batch size
        parser.add_argument('--batch_size', default=batch_size, type=int, metavar='N',
                            help='batch size when validation.')
        ##########################################################################################

        args = parser.parse_args()

        self.validate(args)

    def detect(self, parser_data):
        from draw_box_utils import draw_box
        import json
        from PIL import Image
        # import torchvision
        from torchvision import transforms as trans
        # get devices
        device = torch.device(parser_data.device)
        print("Using {} device training.".format(device.type))

        # create model
        model = self.create_model(num_classes=parser_data.num_classes + 1)

        # load train weights
        train_weights = parser_data.weights
        assert os.path.exists(train_weights), "not found {} file.".format(train_weights)
        model.load_state_dict(torch.load(train_weights, map_location=device)['model'])
        model.to(device)

        # read class_indict
        label_json_path = './pascal_voc_classes.json'
        assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
        json_file = open(label_json_path, 'r')
        class_dict = json.load(json_file)
        json_file.close()
        category_index = {v: k for k, v in class_dict.items()}

        # load image
        # path = "./Detect2"  # 图片所在的文件夹路径
        # save_detect_path = './detected'
        path = parser_data.data_path
        save_detect_path = parser_data.outputdir

        if not os.path.exists(save_detect_path):
            os.makedirs(save_detect_path)
        #
        for maindir, subdir, file_name_list in os.walk(path):
            print(file_name_list)
            for file_name in file_name_list:
                image = os.path.join(maindir, file_name)  # 获取每张图片的路径
                original_img = Image.open(image)
                original_img = original_img.resize((int(self.ui.lineEdit_resize.text()), int(self.ui.lineEdit_resize_2.text())), Image.ANTIALIAS)
                # from pil image to tensor, do not normalize image
                data_transform = trans.Compose([trans.ToTensor()])
                img = data_transform(original_img)
                # expand batch dimension
                img = torch.unsqueeze(img, dim=0)

                model.eval()  # 进入验证模式
                with torch.no_grad():
                    # init
                    img_height, img_width = img.shape[-2:]
                    init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                    model(init_img)

                    # t_start = time_synchronized()
                    predictions = model(img.to(device))[0]
                    # t_end = time_synchronized()
                    # print("inference+NMS time: {}".format(t_end - t_start))

                    predict_boxes = predictions["boxes"].to("cpu").numpy()
                    predict_classes = predictions["labels"].to("cpu").numpy()
                    predict_scores = predictions["scores"].to("cpu").numpy()

                    if len(predict_boxes) == 0:
                        print("没有检测到任何目标!")

                    draw_box(original_img,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index,
                             thresh=0.5,
                             line_thickness=3)
                    # plt.imshow(original_img)
                    # plt.show()
                    # 保存预测的图片结果
                    image_out = os.path.join(save_detect_path, file_name)
                    original_img.save(image_out)

    def start_detect(self):
        import argparse

        parser = argparse.ArgumentParser(
            description=__doc__)
        # # #
        device = 'cuda:0' if self.ui.checkBox_2.isChecked() else 'cpu'
        data_path = self.ui.lineEdit_datapath_2.text()
        num_classes = self.ui.spinBox_numclass_2.text()
        # batch_size = self.ui.spinBox_batchsize_2.text()
        weights = self.ui.lineEdit_outputdir_3.text()
        outputdir = self.ui.lineEdit_outputdir_2.text()
        # # #
        # 使用设备类型
        parser.add_argument('--device', default=device, help='device')

        # 检测目标类别数
        parser.add_argument('--num-classes', type=int, default=num_classes, help='number of classes')

        # 测试集的目录
        parser.add_argument('--data-path', default=data_path, help='dataset root')

        # 输出detection结果图
        parser.add_argument('--outputdir', default=outputdir, help='output path of images after detection')

        # 训练好的权重文件
        parser.add_argument('--weights', default=weights, type=str, help='training weights')

        # batch size
        # parser.add_argument('--batch_size', default=batch_size, type=int, metavar='N',
        #                     help='batch size when validation.')
        ##########################################################################################

        args = parser.parse_args()

        self.detect(args)


######################################################################################
class Select(QDialog):

    def __init__(self):
        super().__init__()
        # 使用ui文件导入定义界面类
        self.path_pic_next1 = None
        self.ui = view()
        # 初始化界面
        self.ui.setupUi(self)
        # 链接按钮与函数
        self.ui.button_52.clicked.connect(self.select_pic)
        self.ui.button_53.clicked.connect(self.next_pic)
        self.ui.button_51.clicked.connect(self.previous_pic)
        # 用于检查是否选择了文件的变量，防止未选择文件就点击“下一张或“上一张”而崩溃
        self.variable_of_chick = 0
        # self.show()

    def select_pic(self):
        # “选择文件”函数
        self.variable_of_chick = 1  # 已经选择文件，置1
        self.path_pic_main = QFileDialog.getOpenFileName(self, "View detection results", ".", "Images (*.jpg)")[0]  # 选择jpg图片
        if self.path_pic_main:  # 判断路径是否为空，防止因中途取消而崩溃
            position_of_key = self.path_pic_main.rfind("/")  # 从后往前寻找关键字'/'的位置
            self.path_of_pic_folder = self.path_pic_main[0:position_of_key]  # 截取图片所在文件夹的路径
            self.name_of_main_pic = self.path_pic_main[position_of_key + 1:len(self.path_pic_main)]  # 截取图片名
            self.names_all_pics = os.listdir(self.path_of_pic_folder)  # 获取图片所在文件夹中所有文件的文件名
            self.index_main = self.names_all_pics.index(self.name_of_main_pic)  # 获取主图的索引
            if self.index_main == len(self.names_all_pics) - 2:  # 选择的图片是倒数第二张
                self.situation_1()
            elif self.index_main == len(self.names_all_pics) - 1:  # 选择的图片是倒数第一张
                self.situation_2()
            elif self.index_main == 0:  # 选择的图片是正数第一张
                self.situation_3()
            elif self.index_main == 1:  # 选择的图片是正数第二张
                self.situation_4()
            else:
                self.situation_normal()  # 正常情况

        return None

    def next_pic(self):
        # “下一张”函数
        if self.variable_of_chick == 1:  # 检查是否选择过文件
            print("next")
            self.ui.button_51.setEnabled(True)  # 激活“上一张”的按钮，用于在其在正数第一张时被禁用后的激活
            self.index_main = self.index_main + 1  # 将主图的索引加一
            self.path_pic_main = self.path_of_pic_folder + '/' + self.names_all_pics[self.index_main]  # 合成主图路径
            if self.index_main == len(self.names_all_pics) - 2:  # 下一张的图片是倒数第二张
                self.situation_1()
            elif self.index_main == len(self.names_all_pics) - 1:  # 下一张的图片是倒数第一张
                self.situation_2()
                QMessageBox.about(self, '警告', '这已经是最后一张照片了！')
            elif self.index_main == 1:  # 下一张的图片是正数第二张
                self.situation_4()
            else:  # 正常情况
                self.situation_normal()
        else:
            QMessageBox.about(self, '警告', '未指定图片')
            return None

    def previous_pic(self):
        if self.variable_of_chick == 1:  # 检查是否选择过文件
            print("previous")
            self.ui.button_53.setEnabled(True)  # 激活“下一张”的按钮，用于在其在倒数第一张时被禁用后的激活
            self.index_main = self.index_main - 1  # 将主图的索引减一
            self.path_pic_main = self.path_of_pic_folder + '/' + self.names_all_pics[self.index_main]  # 合成主图路径
            if self.index_main == len(self.names_all_pics) - 2:  # 下一张的图片是倒数第二张
                self.situation_1()
            elif self.index_main == 1:  # 下一张的图片是正数第二张
                self.situation_4()
            elif self.index_main == 0:  # 下一张的图片是正数第一张
                self.situation_3()
                QMessageBox.about(self, '警告', '这已经是第一张照片了！')
            else:  # 正常情况
                self.situation_normal()
        else:
            QMessageBox.about(self, '警告', '未指定图片')
            return None

    def situation_1(self):
        # 这种情况只需要四张图片的路径，主图路径在主函数合成，其他三张图的路径在本函数进行合成
        self.path_pic_previous2 = self.path_of_pic_folder + '/' + self.names_all_pics[self.index_main - 2]
        self.path_pic_previous1 = self.path_of_pic_folder + '/' + self.names_all_pics[self.index_main - 1]
        self.path_pic_next1 = self.path_of_pic_folder + '/' + self.names_all_pics[self.index_main + 1]
        self.pic_previous2 = self.fill(self.path_pic_previous2, self.ui.label_51.width() / self.ui.label_51.height())
        self.ui.label_51.setPixmap(QPixmap(self.pic_previous2))
        self.ui.label_56.setText(self.names_all_pics[self.index_main - 2])

        self.pic_previous1 = self.fill(self.path_pic_previous1, self.ui.label_52.width() / self.ui.label_52.height())
        self.ui.label_52.setPixmap(QPixmap(self.pic_previous1))
        self.ui.label_57.setText(self.names_all_pics[self.index_main - 1])

        self.pic_main = self.fill(self.path_pic_main, self.ui.label_53.width() / self.ui.label_53.height())
        self.ui.label_53.setPixmap(QPixmap.fromImage(self.pic_main))
        self.ui.label_58.setText(self.names_all_pics[self.index_main])

        self.pic_next1 = self.fill(self.path_pic_next1, self.ui.label_54.width() / self.ui.label_54.height())
        self.ui.label_54.setPixmap(QPixmap(self.pic_next1))
        self.ui.label_59.setText(self.names_all_pics[self.index_main + 1])
        self.ui.label_55.setPixmap(QPixmap(""))  # 给lab传入空图片
        self.ui.label_60.setText("")  # 给lab传入空字符
        # 以下函数同理

    def situation_2(self):
        self.path_pic_previous2 = self.path_of_pic_folder + '/' + self.names_all_pics[self.index_main - 2]
        self.path_pic_previous1 = self.path_of_pic_folder + '/' + self.names_all_pics[self.index_main - 1]
        self.pic_previous2 = self.fill(self.path_pic_previous2, self.ui.label_51.width() / self.ui.label_51.height())
        self.ui.label_51.setPixmap(QPixmap(self.pic_previous2))
        self.ui.label_56.setText(self.names_all_pics[self.index_main - 2])
        self.pic_previous1 = self.fill(self.path_pic_previous1, self.ui.label_52.width() / self.ui.label_52.height())
        self.ui.label_52.setPixmap(QPixmap(self.pic_previous1))
        self.ui.label_57.setText(self.names_all_pics[self.index_main - 1])
        self.pic_main = self.fill(self.path_pic_main, self.ui.label_53.width() / self.ui.label_53.height())
        self.ui.label_53.setPixmap(QPixmap.fromImage(self.pic_main))
        self.ui.label_58.setText(self.names_all_pics[self.index_main])
        self.ui.label_54.setPixmap(QPixmap(""))
        self.ui.label_59.setText("")
        self.ui.label_55.setPixmap(QPixmap(""))
        self.ui.label_60.setText("")
        self.ui.button_53.setDisabled(True)  # 游览到最后一张时时禁用下一张按钮

    def situation_3(self):
        self.path_pic_next1 = self.path_of_pic_folder + '/' + self.names_all_pics[self.index_main + 1]
        self.path_pic_next2 = self.path_of_pic_folder + '/' + self.names_all_pics[self.index_main + 2]
        self.ui.label_51.setPixmap(QPixmap(""))
        self.ui.label_56.setText("")
        self.ui.label_52.setPixmap(QPixmap(""))
        self.ui.label_57.setText("")
        self.pic_main = self.fill(self.path_pic_main, self.ui.label_53.width() / self.ui.label_53.height())
        self.ui.label_53.setPixmap(QPixmap.fromImage(self.pic_main))
        self.ui.label_58.setText(self.names_all_pics[self.index_main])
        self.pic_next1 = self.fill(self.path_pic_next1, self.ui.label_54.width() / self.ui.label_54.height())
        self.ui.label_54.setPixmap(QPixmap(self.pic_next1))
        self.ui.label_59.setText(self.names_all_pics[self.index_main + 1])
        self.pic_next2 = self.fill(self.path_pic_next2, self.ui.label_55.width() / self.ui.label_55.height())
        self.ui.label_55.setPixmap(QPixmap(self.pic_next2))
        self.ui.label_60.setText(self.names_all_pics[self.index_main + 2])
        self.ui.button_51.setDisabled(True)  # 游览到第一张时时禁用上一张按钮

    def situation_4(self):
        self.path_pic_previous1 = self.path_of_pic_folder + '/' + self.names_all_pics[self.index_main - 1]
        self.path_pic_next1 = self.path_of_pic_folder + '/' + self.names_all_pics[self.index_main + 1]
        self.path_pic_next2 = self.path_of_pic_folder + '/' + self.names_all_pics[self.index_main + 2]
        self.ui.label_51.setPixmap(QPixmap(""))
        self.ui.label_56.setText("")

        self.pic_previous1 = self.fill(self.path_pic_previous1, self.ui.label_52.width() / self.ui.label_52.height())
        self.ui.label_52.setPixmap(QPixmap(self.pic_previous1))
        self.ui.label_57.setText(self.names_all_pics[self.index_main - 1])

        self.pic_main = self.fill(self.path_pic_main, self.ui.label_53.width() / self.ui.label_53.height())
        self.ui.label_53.setPixmap(QPixmap.fromImage(self.pic_main))
        self.ui.label_58.setText(self.names_all_pics[self.index_main])

        self.pic_next1 = self.fill(self.path_pic_next1, self.ui.label_54.width() / self.ui.label_54.height())
        self.ui.label_54.setPixmap(QPixmap(self.pic_next1))
        self.ui.label_59.setText(self.names_all_pics[self.index_main + 1])

        self.pic_next2 = self.fill(self.path_pic_next2, self.ui.label_55.width() / self.ui.label_55.height())
        self.ui.label_55.setPixmap(QPixmap(self.pic_next2))
        self.ui.label_60.setText(self.names_all_pics[self.index_main + 2])

    def situation_normal(self):
        self.path_pic_previous2 = self.path_of_pic_folder + '/' + self.names_all_pics[self.index_main - 2]
        self.path_pic_previous1 = self.path_of_pic_folder + '/' + self.names_all_pics[self.index_main - 1]
        self.path_pic_next1 = self.path_of_pic_folder + '/' + self.names_all_pics[self.index_main + 1]
        self.path_pic_next2 = self.path_of_pic_folder + '/' + self.names_all_pics[self.index_main + 2]
        # 填充label53的边缘

        self.pic_previous2 = self.fill(self.path_pic_previous2, self.ui.label_51.width() / self.ui.label_51.height())
        self.ui.label_51.setPixmap(QPixmap(self.pic_previous2))
        self.ui.label_56.setText(self.names_all_pics[self.index_main - 2])

        self.pic_previous1 = self.fill(self.path_pic_previous1, self.ui.label_52.width() / self.ui.label_52.height())
        self.ui.label_52.setPixmap(QPixmap(self.pic_previous1))
        self.ui.label_57.setText(self.names_all_pics[self.index_main - 1])

        self.pic_main = self.fill(self.path_pic_main, self.ui.label_53.width() / self.ui.label_53.height())
        self.ui.label_53.setPixmap(QPixmap.fromImage(self.pic_main))
        self.ui.label_58.setText(self.names_all_pics[self.index_main])

        self.pic_next1 = self.fill(self.path_pic_next1, self.ui.label_54.width() / self.ui.label_54.height())
        self.ui.label_54.setPixmap(QPixmap(self.pic_next1))
        self.ui.label_59.setText(self.names_all_pics[self.index_main + 1])

        self.pic_next2 = self.fill(self.path_pic_next2, self.ui.label_55.width() / self.ui.label_55.height())
        self.ui.label_55.setPixmap(QPixmap(self.pic_next2))
        self.ui.label_60.setText(self.names_all_pics[self.index_main + 2])

    def fill(self, path, ratio):
        # 边缘填充函数
        pic = cv2.imread(path)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        k1 = pic.shape[1] / pic.shape[0]  # 图片的长宽比
        k2 = ratio  # label的长宽比
        if k1 >= k2:
            # 假如图片的长宽比大于等于label的长宽比，填充宽度
            a = int(((pic.shape[0] * k1 / k2) - pic.shape[0]) / 2)  # 计算填充量并均分
            pic_mian = cv2.copyMakeBorder(pic, a, a, 0, 0, cv2.BORDER_CONSTANT,
                                          value=[240, 240, 240])  # 填充颜色rgb[240, 240, 240]
            image_height, image_width, image_depth = pic_mian.shape
            saved_img_show = QImage(pic_mian.data, image_width, image_height, image_width * image_depth,
                                    QImage.Format_RGB888)
        else:
            # 假如图片的长宽比小于label的长宽比，填充长度
            b = int(((pic.shape[1] * k2 / k1) - pic.shape[1]) / 2)  # 计算填充量并均分
            pic_mian = cv2.copyMakeBorder(pic, 0, 0, b, b, cv2.BORDER_CONSTANT, value=[240, 240, 240])
            image_height, image_width, image_depth = pic_mian.shape
            saved_img_show = QImage(pic_mian.data, image_width, image_height, image_width * image_depth,
                                    QImage.Format_RGB888)
        return saved_img_show


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = LoginWindow()
    # win_2 = InterfaceWindow()
    # win_3 = Select()
    #
    # btn = win_2.ui.pushButton_10
    # btn.clicked.connect(win_3.exec)
    win.show()
    sys.exit(app.exec_())
