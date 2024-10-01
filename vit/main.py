import torch
import random
import argparse
import time
#import utils
#import config
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
#import torch.distributed as dist
#import torch.multiprocessing as mp
from metrics import accuracy
import model
from model import ViT
from model_long import ViTL
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from scheduler import WarmupCosineSchedule
from torchvision import datasets, transforms
from dataloader import ImageFolder
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='vision transformer')
parser.add_argument('--data', type=str, default='imagenet', metavar='N',
                    help='data')
parser.add_argument('--data_path', type=str, default='./imagenet', metavar='N',
                    help='data')
parser.add_argument('--model', type=str, default='vit', metavar='N',
                    help='model')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',   
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0.3, metavar='M',
                    help='adam weight_decay (default: 0.5)')
parser.add_argument('--t_max', type=float, default=80000, metavar='M',
                    help='cosine annealing steps')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--alpha', type=int, default=0.9, metavar='N',
                    help='alpha')
parser.add_argument('--workers', type=int, default=4, metavar='N',
                    help='num_workers')
parser.add_argument('--gpu',type=str,default='0',
                    help = 'gpu')
parser.add_argument('--mode',type=str,default='None',
                    help = 'train/val mode')
parser.add_argument('--resume',type=bool,default=False)
parser.add_argument('--model_type',type=str,default=False)

args = parser.parse_args()

def main() :
    #data load.
    st = time.time()
    train_dir = os.path.join(args.data_path,'train')
    test_dir = os.path.join(args.data_path,'val')

    trainset = ImageFolder(
        train_dir,
        transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])]))
    valset = ImageFolder(
        test_dir,
        transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])]))


    print('data load',time.time()-st)

    args.batch_size = int(args.batch_size)
    args.workers = args.workers
    train_sampler = torch.utils.data.RandomSampler(trainset)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,sampler=train_sampler)
    val_loader = DataLoader(valset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=False)

    DEPTH=12

    NUM_CLASSES=1000
    if args.model_type == 'long':
        print("\nLong Connections On \n")
        model = ViTL(in_channels = 3,
                patch_size = 16,
                emb_size = 768,
                img_size = 224,
                depth = DEPTH,
                n_classes = NUM_CLASSES,
                )
    else:
        print("\n Vanilla ViT \n")
        model = ViT(in_channels = 3,
                patch_size = 16,
                emb_size = 768,
                img_size = 224,
                depth = DEPTH,
                n_classes = NUM_CLASSES,
                )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of trainable parameters: {total_params}')

    #cdir = 'checkpoints_long'
    #ckpt = cdir + '/checkpoint_epoch_3.pth.tar'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == 'train':

        if args.resume == True:
            print("Resume Training...")
            checkpoint = torch.load(ckpt)
            # model
            model = model.to(device)
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer
            optimizer = optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
            optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler
            scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=0.0000001)
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
            print("Resume from epoch, ", start_epoch)
        else:
            print("Start from Scratch...")
            start_epoch = 0
            optimizer = optim.AdamW(model.parameters(), lr=args.lr,weight_decay = args.weight_decay)
            #learning rate warmup of 80k steps
            t_total = ( len(train_loader.dataset) / args.batch_size ) * args.epochs
            t_epoch = len(train_loader.dataset) / args.batch_size
            print(f"total_step : {t_total}")
            print(f"per_epoch_step : {t_epoch}")
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps= 300000, t_total=args.t_max)

        model = model.to(device)
        criterion = nn.CrossEntropyLoss().to(device)

        #clip_norm
        max_norm = 1

        #train
        scaler = torch.cuda.amp.GradScaler()

        best_accuracy = -1.0
        for epoch in range(start_epoch, args.epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", unit="batch", bar_format="{l_bar}{r_bar}", dynamic_ncols=True)
            for batch_idx, (data, target) in enumerate(train_loader):
                st = time.time()
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    output, _ = model(data)
                    loss = criterion(output,target)
                scaler.scale(loss).backward()
                '''
                nan_in_gradients=False
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            nan_in_gradients = True
                            print(f"NaN detected in gradients of {name}")
                        #grad_norm = param.grad.norm()
                        #print(f"Gradient norm for {name}: {grad_norm.item()}")
                '''
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  #iter
                train_loss = loss.data

                ############
                pbar.set_postfix({
                    'Loss': f'{train_loss:.6f}',
                    'Progress': f'{batch_idx * len(data)}/{int(len(train_loader.dataset))} ({100. * batch_idx / len(train_loader):.0f}%)',
                    'LR': f'{scheduler.get_last_lr()[0]:.6f}'
                })
                ############
            if True: # esle save every x epoch
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                },filename=f'checkpoints_long/checkpoint_{epoch}.pth.tar')
            model.eval()
            correct = 0
            total_acc1 = 0
            total_acc5 = 0
            total_acc1_list = [0 for _ in range(DEPTH+1)]
            total_acc5_list = [0 for _ in range(DEPTH+1)]
            step=0
            st = time.time()
            for batch_idx,(data, target) in enumerate(val_loader) :
                with torch.no_grad() :
                    data, target = data.to(device), target.to(device)
                    output, intermid = model(data, eval_=True)
                val_loss = criterion(output,target)
                val_loss = val_loss.data

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                total_acc1 += acc1[0]
                total_acc5 += acc5[0]
                #########
                for (idx, inter) in enumerate(intermid):
                    acc1, acc5 = accuracy(inter, target, topk=(1, 5))
                    total_acc1_list[idx] = total_acc1_list[idx] + acc1[0]
                    total_acc5_list[idx] = total_acc5_list[idx] + acc5[0]
                #########
                step+=1

            epoch_acc1 = total_acc1/step
            epoch_acc5 = total_acc5/step
            ########
            epoch_acc1_list = []
            epoch_acc5_list = []
            for i in total_acc1_list:
                epoch_acc1_list.append(i/step)
            for i in total_acc5_list:
                epoch_acc5_list.append(i/step)
            ########
            print("")
            print('\nval set: top1: {}, top5 : {} '.format(epoch_acc1, epoch_acc5))
            print("Top1-Accuracy")
            print(", ".join("Layer %d: %2.5f" % (idx+1, acc) for idx, acc in enumerate(epoch_acc1_list)))
            print("Top5-Accuracy")
            print(", ".join("Layer %d: %2.5f" % (idx+1, acc) for idx, acc in enumerate(epoch_acc5_list)))
            print("")

            ###### Save Best Checkpoint ######
            if epoch_acc1 > best_accuracy:
                best_accuracy = epoch_acc1
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                    },filename=f'checkpoints_long/checkpoint_best_model.pth.tar')

    if args.mode == 'val' :
        print("Validating...")
        checkpoint = torch.load(ckpt)
        # model
        model = model.to(device)
        #model_storage_size(model)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

        criterion = nn.CrossEntropyLoss().to(device)

        model.eval()
        correct = 0
        total_acc1 = 0
        total_acc5 = 0
        total_acc1_list = [0 for _ in range(DEPTH+1)]
        total_acc5_list = [0 for _ in range(DEPTH+1)]
        #inference_time_list = []
        step=0
        st = time.time()
        for batch_idx,(data, target) in enumerate(val_loader):
            with torch.no_grad() :
                data, target = data.to(device), target.to(device)
                #start_time = time.time()
                output, intermid = model(data, eval_=True)
                #end_time = time.time()
                #inference_time = end_time - start_time
                #inference_time_list.append(inference_time)
            val_loss = criterion(output,target)
            val_loss = val_loss.data

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            total_acc1 += acc1[0]
            total_acc5 += acc5[0]
            #########
            for (idx, inter) in enumerate(intermid):
                acc1, acc5 = accuracy(inter, target, topk=(1, 5))
                total_acc1_list[idx] = total_acc1_list[idx] + acc1[0]
                total_acc5_list[idx] = total_acc5_list[idx] + acc5[0]
            #########
            step+=1
            batch_counter += 1

        epoch_acc1 = total_acc1/step
        epoch_acc5 = total_acc5/step
        ########
        epoch_acc1_list = []
        epoch_acc5_list = []
        for i in total_acc1_list:
            epoch_acc1_list.append(i/step)
        for i in total_acc5_list:
            epoch_acc5_list.append(i/step)
        ########

        print('\nval set: top1: {}, top5 : {} '.format(epoch_acc1, epoch_acc5))
        print("Top1-Accuracy")
        print(", ".join("Layer %d: %2.5f" % (idx+1, acc) for idx, acc in enumerate(epoch_acc1_list)))
        print("Top5-Accuracy")
        print(", ".join("Layer %d: %2.5f" % (idx+1, acc) for idx, acc in enumerate(epoch_acc5_list)))

        #print("Avg. Inference Time: ", sum(inference_time_list) / len(inference_time_list))


import psutil
def get_cpu_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss  # in bytes

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def model_storage_size(model, dtype=torch.float32):
    """
    Calculate the total storage size of the model parameters in MB.
    """
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Dtype: {param.dtype}")
        break

    total_params = 0
    bytes_per_param = torch.finfo(dtype).bits / 8
    print("Bytes per parameter: ", bytes_per_param)

    for param in model.parameters():
        total_params += param.numel()

    total_storage = total_params * bytes_per_param
    total_storage_MB = total_storage / (1024 ** 2)
    print("Total size in MB: ", total_storage_MB)

if __name__ == '__main__' :
   main()