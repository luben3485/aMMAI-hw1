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

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

writer = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument('--train_root', type=str, default='./data/train')
parser.add_argument('--checkpoint_dir', type=str, default='./ckpnt_softmax_0407')
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--num_class', type=int, default=685)
parser.add_argument('--learning_rate', type=float, default=0.5)

args = parser.parse_args()

np.random.seed(8777)

def main():
    train_root = args.train_root
    epochs = args.epochs
    batch_size = args.batch_size
    embedding_dim = args.embedding_dim
    num_class = args.num_class
    learning_rate = args.learning_rate
    checkpoint_dir = args.checkpoint_dir

    train_transform = trns.Compose([
        trns.Resize((256, 256)),
        #trns.RandomCrop((224, 224)),
        #trns.RandomHorizontalFlip(),
        trns.ToTensor(),
        trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = APDDataset(root=train_root, transform=train_transform)
    train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True, num_workers=1)

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

            embedding, logits = model.forward_training(data)
            loss = criterion(logits.cuda(), labels.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(dim=-1) == labels).float().mean()

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


if __name__ == '__main__':
    main()



