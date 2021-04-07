import os
import numpy as np
import argparse
import torch
import torchvision.transforms as trns
from models.resnet34 import Resnet34Center
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--test_root', type=str, default='data/test')
parser.add_argument('--ckpnt', type=str, default='ckpnt_center/checkpoint-400.pth')
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--num_class', type=int, default=685)

args = parser.parse_args()

np.random.seed(8777)

def main():
    test_root = args.test_root
    embedding_dim = args.embedding_dim
    num_class = args.num_class
    ckpnt = args.ckpnt

    root = os.getcwd()

    with open(os.path.join(root, test_root, 'closed_set', 'labels.txt')) as f:
        labels = f.readlines()

    closed_set_label = []
    for label in labels:
        label = int(label)
        closed_set_label.append(label)


    with open(os.path.join(root, test_root, 'open_set', 'labels.txt')) as f:
        labels = f.readlines()

    open_set_label = []
    for label in labels:
        label = int(label)
        open_set_label.append(label)


    test_transform = trns.Compose([
        trns.Resize((256, 256)),
        #trns.RandomCrop((224, 224)),
        #trns.RandomHorizontalFlip(),
        trns.ToTensor(),
        trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = Resnet34Center(
        num_classes=num_class,
        embedding_dimension=embedding_dim,
        pretrained=False
    )
    model.load_state_dict(torch.load(os.path.join(root, ckpnt)))
    model.cuda()

    closed_set_pred = []
    for i in tqdm(range(len(closed_set_label))):
        img_path_1 = os.path.join(root, test_root, 'closed_set', 'test_pairs', 'test_pair_' + str(i) + '_1.jpg')
        img_path_2 = os.path.join(root, test_root, 'closed_set', 'test_pairs', 'test_pair_' + str(i) + '_2.jpg')
        img_1 = Image.open(img_path_1).convert('RGB')
        img_2 = Image.open(img_path_2).convert('RGB')
        img_1 = test_transform(img_1).unsqueeze(0).cuda()
        img_2 = test_transform(img_2).unsqueeze(0).cuda()

        model.eval()

        with torch.no_grad():
            feature_1 = model(img_1)
            feature_2 = model(img_2)

        np_feat_1 = feature_1.detach().cpu().numpy()
        np_feat_2 = feature_2.detach().cpu().numpy()
        dis = np.sum(np.square(np_feat_1 - np_feat_2)) / embedding_dim

        threshold = 1.4
        if dis < threshold:
            closed_set_pred.append(1)
        else:
            closed_set_pred.append(0)

    closed_set_pred = np.array(closed_set_pred)
    closed_set_label = np.array(closed_set_label)

    acc = (closed_set_pred == closed_set_label).mean()
    print('closed_set accurancy',acc)

    open_set_pred = []
    for i in tqdm(range(len(open_set_label))):
        img_path_1 = os.path.join(root, test_root, 'open_set', 'test_pairs', 'test_pair_' + str(i) + '_1.jpg')
        img_path_2 = os.path.join(root, test_root, 'open_set', 'test_pairs', 'test_pair_' + str(i) + '_2.jpg')
        img_1 = Image.open(img_path_1).convert('RGB')
        img_2 = Image.open(img_path_2).convert('RGB')
        img_1 = test_transform(img_1).unsqueeze(0).cuda()
        img_2 = test_transform(img_2).unsqueeze(0).cuda()

        model.eval()

        with torch.no_grad():
            feature_1 = model(img_1)
            feature_2 = model(img_2)

        np_feat_1 = feature_1.detach().cpu().numpy()
        np_feat_2 = feature_2.detach().cpu().numpy()
        dis = np.sum(np.square(np_feat_1 - np_feat_2)) / embedding_dim

        threshold = 1.1
        if dis < threshold:
            open_set_pred.append(1)
        else:
            open_set_pred.append(0)

    open_set_pred = np.array(open_set_pred)
    open_set_label = np.array(open_set_label)

    acc = (open_set_pred == open_set_label).mean()
    print('open_set accurancy',acc)


if __name__ == '__main__':
    main()



