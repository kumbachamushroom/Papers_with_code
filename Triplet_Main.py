from __future__ import print_function
import torch
import torch.nn as nn
from Triplet_DataLoader import TripletLoader
from utlis import get_conv2d_output_shape, get_conv1d_output_shape
import os
import numpy as np
import wandb
from Triplet_Net import VGGVox, TripletNet
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from visdom import Visdom
from torch.autograd import Variable
from Triplet_DataLoader import Spectrogram_Loader
from losses import batch_hard_triplet_loss
import numpy as np
from Triplet_DataLoader import Triplet_Time_Loader
import time


class AverageMeter:
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.valu = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val *n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(dista, distb):
    margin = 0.2
    pred = (distb - dista - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name],
                                    name=split_name)

def test(test_loader, tnet, criterion, epoch):
    losses = AverageMeter()
    accs = AverageMeter()


    # switch evaluation mode
    tnet.eval()
    for batch_idx, (data1, data2, data3) in enumerate(test_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        #compute output
        dista, distb, embedding_x, embedding_y, embedding_z = tnet(data1,data2,data3)
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        #test_loss = criterion(dista, distb, target).item()
        test_loss = criterion(embedding_x, embedding_y, embedding_z).item()
        #measure the accuracy and record the loss
        acc = accuracy(dista, distb)
        accs.update(acc, data1.size(0))
        losses.update(test_loss,data1.size(0))

    print('\nTest set: average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            losses.avg, 100. * accs.avg))
    #plotter.plot('acc', 'test', epoch, accs.avg)
    #plotter.plot('loss','test', epoch, losses.avg)
    wandb.log({"Test accuracy":accs.avg,"Test Loss":losses.avg})
    return accs.avg

def test_batch(test_loader, tnet, epoch):
    tnet.eval()
    losses = AverageMeter()

    i = 0
    total_samples = 0
    correct_samples = 0
    positive_dist = 0
    negative_dist = 0
    with torch.no_grad():
        for batch in iter(test_loader):
            #i = i + 1

            spectrograms, int_labels, string_labels = batch
            int_labels = int_labels.cuda()
            # labels = labels.view(labels.size(0), -1).cuda()
            spectrograms = spectrograms.cuda()
            embeddings = tnet(spectrograms)
            loss, correct, total = batch_hard_triplet_loss(int_labels, embeddings, margin=2,device='cuda', squared=True)
            losses.update(loss,1)
            correct_samples = correct_samples + correct
            total_samples = total_samples + total
            #negative_dist = negative_dist + hardest_neg.sum()
            #positive_dist = positive_dist + hardest_pos.sum()


        print('Epoch {} Test Accuracy: {:.4f} Test_Loss: {:.4f} \t'.format(epoch, (
                    correct_samples / total_samples),losses.avg))
        wandb.log({"Test Accuracy": correct_samples / total_samples, "Test Loss": losses.avg})
        return


def train_batch(train_loader, tnet, optimizer, epoch):
    tnet.train()
    losses = AverageMeter()
    i = 0
    total_samples = 0
    correct_samples = 0
    positive_dist = 0
    negative_dist = 0
    for batch in iter(train_loader):
        i = i + 1

        spectrograms, int_labels, string_labels = batch
        int_labels = int_labels.cuda()
        #labels = labels.view(labels.size(0), -1).cuda()
        spectrograms = spectrograms.cuda()
        embeddings = tnet(spectrograms)

        loss, correct, total = batch_hard_triplet_loss(int_labels, embeddings, margin=2, device='cuda', squared=True)
        correct_samples = correct_samples + correct
        total_samples = total_samples + total
        #negative_dist = negative_dist + hardest_neg.sum()
        #positive_dist = positive_dist + hardest_pos.sum()
        losses.update(loss,1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss, 1)
        print('Train Epoch: {} [{}/{}]\t'
              'Loss: {:.4f} \t'
              .format(
            epoch, i, len(iter(train_loader)),
            loss))
    print('Epoch {} Accuracy: {:.4f} Loss: {:.4f} \t'.format(epoch,(correct_samples/total_samples),losses.avg))
    wandb.log({"Train Accuracy":correct_samples/total_samples, "Train Loss":losses.avg})




def train(train_loader, tnet, criterion, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()

    #switch to train mode
    tnet.train()

    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        #compute the output
        dista, distb, embedded_x, embedded_y, embedded_z = tnet(data1, data2, data3)
        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)

        loss_triplet = criterion(embedded_x, embedded_y, embedded_z)
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        #loss = loss_triplet + 0.0001 * loss_embedd
        #print("THE LOSS TRIPLET IS: ",loss_triplet)
        #measure accuracy and record loss
        acc = accuracy(dista,distb)
        losses.update(loss_triplet.item(), data1.size(0))
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.item()/3, data1.size(0))

        #compute gradient and do optimizer step
        optimizer.zero_grad()
        loss_triplet.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                losses.val, losses.avg,
                       100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg), end='\r', flush=True)
        # log avg values to somewhere
        #plotter.plot('acc', 'train', epoch, accs.avg)
        #plotter.plot('loss', 'train', epoch, losses.avg)
        #plotter.plot('emb_norms', 'train', epoch, emb_norms.avg)
        wandb.log({"Train accuracy": accs.avg, "Train loss":losses.avg, "Train Embedding Norms":emb_norms.avg})


def main():
    #Train settings
    wandb.init(project="vgg_triplet")
    global args, best_acc
    parser = argparse.ArgumentParser(description='VGG Triplet-Loss Speaker Embedding')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs',type=int, default=50, metavar='N',
                        help='number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--no-cuda', action='store_true',default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval',type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training score')
    parser.add_argument('--margin', type=float, default=2, metavar='M',
                        help='margin for triplet loss (default: 0.2)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--name', default='TripletNet_RMSprop', type=str,
                        help='name of experiment')
    parser.add_argument('--base-path',type=str,
                        default='/home/lucas/PycharmProjects/Papers_with_code/data/AMI/amicorpus_individual/Extracted_Speech',
                        help='string to triplets')
    parser.add_argument('--ap-file',default='anchor_pairs.txt',type=str,
                        help='name of file with anchor-positive pairs')
    parser.add_argument('--s-file',default='trimmed_sample_list.txt',type=str,
                        help='name of sample list')
    parser.add_argument('--save-path',default='/home/lucas/PycharmProjects/Papers_with_code/data/models/VGG_Triplet'
                        ,type=str, help='path to save models to')
    parser.add_argument('--save', type=bool, default=True,
                        help='save model?')
    parser.add_argument('--load', type=bool, default=False,
                        help='load model from latest checkpoint')
    args = parser.parse_args()

    wandb.run.name = args.name
    wandb.run.save()
    wandb.config.update(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args.no_cuda)
    print(torch.cuda.is_available())
    if args.cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.seed)
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

    #train_loader = torch.utils.data.DataLoader(TripletLoader(base_path=args.base_path,anchor_positive_pairs=args.ap_file,sample_list=args.s_file,train=True),
    #                                           batch_size=args.batch_size,shuffle=True,**kwargs)
    #test_loader = torch.utils.data.DataLoader(TripletLoader(base_path=args.base_path,anchor_positive_pairs=args.ap_file,sample_list=args.s_file,train=False),
    #                                          batch_size=args.test_batch_size,shuffle=True,**kwargs)

    #single_train_loader = torch.utils.data.DataLoader(Spectrogram_Loader(base_path=args.base_path, anchor_positive_pairs=args.ap_file, sample_list=args.s_file, train=True), batch_size=args.batch_size, shuffle=True, **kwargs)
    #single_test_loader = torch.utils.data.DataLoader(Spectrogram_Loader(base_path=args.base_path, anchor_positive_pairs=args.ap_file, sample_list=args.s_file, train=False), batch_size=args.test_batch_size, shuffle=True, **kwargs)

    train_time_loader = torch.utils.data.DataLoader(Triplet_Time_Loader(path=os.path.join(args.base_path,args.s_file), train=True), batch_size=args.batch_size, shuffle=True, **kwargs)
    test_time_loader = torch.utils.data.DataLoader(Triplet_Time_Loader(path=os.path.join(args.base_path,args.s_file), train=False), batch_size=args.test_batch_size, shuffle=True, **kwargs)


    #global plotter
    #plotter = VisdomLinePlotter(env_name=args.name)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=7)
            self.conv2 = nn.Conv2d(16,16,kernel_size=7)
            self.bn_1 = nn.BatchNorm2d(16)
            self.conv3 = nn.Conv2d(16, 32, kernel_size=7)
            self.conv4 = nn.Conv2d(32, 32, kernel_size=7)
            self.bn_2 = nn.BatchNorm2d(32)
            self.conv2_drop = nn.Dropout2d(p=0.2)
            self.fc1 = nn.Linear(448, 256)
            self.fc2 = nn.Linear(256, 256)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(F.max_pool2d(self.bn_1(self.conv2(x)), 7))
            x = F.relu(self.conv3(x))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.bn_2(self.conv4(x))), 7))
            #print("SIZE  ",x.size())
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            return self.fc2(x)
    model = VGGVox()
    if args.cuda:
        model.to(device)
    if args.load:
        model.load_state_dict(torch.load(args.save_path))
        print("Model loaded from state dict")
    #tnet = TripletNet(model)
    #if args.cuda:
    #    tnet.to(device)
    wandb.watch(model)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    #criterion = torch.nn.MarginRankingLoss(margin = args.margin)
    criterion = nn.TripletMarginLoss(margin=args.margin, p=2)

    #optimizer = optim.Adam(tnet.parameters(),lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.RMSprop(model.parameters(),lr=args.lr ,alpha=0.8, momentum=args.momentum)
    #n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    #print('  + NUmber of params: {}'.format(n_parameters))

    for epoch in range(1, args.epochs+1):
        start_time = time.time()
        train_batch(train_time_loader,model,optimizer,epoch)
        test_batch(test_time_loader,model,epoch)
        duration = time.time() - start_time
        print("Done training epoch {} in {:.4f}".format(epoch, duration))

    #for epoch in range(1, args.epochs + 1):
    #    test_batch(single_train_loader, model, epoch)

    if args.save:
        torch.save(model.state_dict(),args.save_path)
        print("Model Saved")



if __name__ == '__main__':
    main()
