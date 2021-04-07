import os
import time
import numpy as np
import argparse
from dataloader import APDDataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as trns
from torch.utils.tensorboard import SummaryWriter
from models.resnet34 import Resnet34Center
from ArcMarginModel import ArcMarginModel
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN

writer = SummaryWriter()

parser = argparse.ArgumentParser()

parser.add_argument('--train_root', type=str, default='./data/train')
parser.add_argument('--checkpoint_dir', type=str, default='./ckpnt')
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--num_class', type=int, default=685)
parser.add_argument('--learning_rate', type=float, default=0.5)

parser.add_argument('--test_root', type=str, default='data/test')
parser.add_argument('--ckpnt', type=str, default='trained_model/checkpoint.pth')

parser.add_argument('--mode', type=str, default='train')

args = parser.parse_args()

np.random.seed(8777)

train_root = args.train_root
epochs = args.epochs
batch_size = args.batch_size
embedding_dim = args.embedding_dim
num_class = args.num_class
learning_rate = args.learning_rate
checkpoint_dir = args.checkpoint_dir
test_root = args.test_root
ckpnt = args.ckpnt
mode = args.mode

root = os.getcwd()

def train():

    train_transform = trns.Compose([
        trns.Resize((256, 256)),
        #trns.RandomCrop((224, 224)),
        #trns.RandomHorizontalFlip(),
        trns.ToTensor(),
        trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = APDDataset(root=train_root, transform=train_transform)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=1)

    model = Resnet34Center(
        num_classes=num_class,
        embedding_dimension=embedding_dim,
        pretrained=False
    )
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=[150, 300],
        gamma=0.1
    )
    criterion = torch.nn.CrossEntropyLoss().cuda()
    metric_fc = ArcMarginModel(num_class, embedding_dim).cuda()

    train_start_time = time.time()
    for epoch in range(epochs):

        if (epoch+1) % 20 == 0 and epoch > 0:
            file_name = 'checkpoint-%d.pth' % (epoch + 1)
            checkpoint_path = os.path.join(checkpoint_dir, file_name)
            print('Save the network at %s' % checkpoint_path)
            torch.save(model.state_dict(), checkpoint_path)

        train_loss = []
        train_accs = []

        model.train()
        scheduler.step()
        for param_group in optimizer.param_groups:
            print('The learning rate is ', param_group['lr'])

        for batch_index, (data, labels) in enumerate(train_loader):
            
            data, labels = data.cuda(), labels.cuda()

            feature = model(data)
            output = metric_fc(feature, labels)
            loss = criterion(output.cuda(), labels.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=-1) == labels).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        print(f"[ Train | {epoch + 1:03d}/{epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        writer.add_scalar('average train loss', train_loss, epoch)
        writer.add_scalar('average train acc', train_acc, epoch)

    writer.close()
    train_end_time = time.time()
    total_time_elapsed = train_end_time - train_start_time
    print("\nTraining finished: total time elapsed: {:.2f} hours.".format(total_time_elapsed/3600))

def forward_closed():

    with open(os.path.join(root, test_root, 'closed_set', 'labels.txt')) as f:
        labels = f.readlines()

    closed_set_label = []
    for label in labels:
        label = int(label)
        closed_set_label.append(label)


    test_transform = trns.Compose([
        trns.Resize((256, 256)),
        #trns.RandomCrop((224, 224)),
        #trns.RandomHorizontalFlip(),
        #trns.ToTensor(),
        trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = Resnet34Center(
        num_classes=num_class,
        embedding_dimension=embedding_dim,
        pretrained=False
    )
    model.load_state_dict(torch.load(os.path.join(root, ckpnt)))
    model.cuda()

    mtcnn = MTCNN(image_size=256, margin=0)    

    closed_set_pred = []
    for i in tqdm(range(len(closed_set_label))):
        img_path_1 = os.path.join(root, test_root, 'closed_set', 'test_pairs', 'test_pair_' + str(i) + '_1.jpg')
        img_path_2 = os.path.join(root, test_root, 'closed_set', 'test_pairs', 'test_pair_' + str(i) + '_2.jpg')
        img_1 = Image.open(img_path_1).convert('RGB')
        img_2 = Image.open(img_path_2).convert('RGB')
        
        #img_1 = test_transform(img_1)
        #img_2 = test_transform(img_2)

        # face detection
        img_1 = mtcnn(img_1)    
        img_2 = mtcnn(img_2)    
        
        img_1 = img_1.unsqueeze(0).cuda()
        img_2 = img_2.unsqueeze(0).cuda()

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

def forward_open():

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
    if mode == 'train':
        train()
    elif mode == 'closed':
        forward_closed()
    elif mode == 'open':
        forward_open()


