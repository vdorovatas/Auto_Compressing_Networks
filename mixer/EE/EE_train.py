import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warmup_scheduler
import numpy as np
import time
import json
from analyzer import RepresentationAnalyzer, print_comparison_results
from utils import rand_bbox

class Trainer(object):
    def __init__(self, model, device, args):
        self.device = device
        self.clip_grad = 1.0
        self.cutmix_beta = 1.0
        self.cutmix_prob = 0.5
        self.model = model
        self.args = args
        self.epochs = args.epochs
        print('NUM_EPOCHS: ', self.epochs)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=0.000001)
        WARMUP = 10
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1.,
                         total_epoch=WARMUP, after_scheduler=self.base_scheduler)

        self.scaler = torch.amp.GradScaler()

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.num_steps = 0
        self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.

    def _train_one_step(self, batch):
        self.model.train()
        img, label = batch
        self.num_steps += 1
        img, label = img.to(self.device), label.to(self.device)

        self.optimizer.zero_grad() 
        L = 16  # total number of layers
        weights_for_layers = [2*(l+1)/(L*(L+1)) for l in range(L)]
        if self.cutmix_beta > 0 and r < self.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(self.cutmix_beta, self.cutmix_beta)
            rand_index = torch.randperm(img.size(0)).to(self.device)
            target_a = label
            target_b = label[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            # compute output
            with torch.amp.autocast(device_type='cuda'):
                out, intermid = self.model(img)
                ###  EE ###
                intermid_loss = []
                for idx, i in enumerate(intermid):
                    l = self.criterion(i, target_a) * lam + self.criterion(i, target_b) * (1. - lam)
                    wei = weights_for_layers[idx]
                    intermid_loss.append(wei*l)
                loss = sum(intermid_loss) 
       
        else:
            # compute output
            with torch.amp.autocast(device_type='cuda'):
                out, intermid = self.model(img)
                ### EE ###
                intermid_loss = []
                for idx, i in enumerate(intermid):
                    l = self.criterion(i, label)
                    wei = weights_for_layers[idx]
                    intermid_loss.append(wei*l)
                loss = sum(intermid_loss) 
          
        
        self.scaler.scale(loss).backward()
        
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
    
        acc = out.argmax(dim=-1).eq(label).sum(-1)/img.size(0)
        self.train_loss += loss * img.size(0)


    @torch.no_grad
    def _test_one_step(self, batch, T=0.5):
        self.model.eval()
        img, label = batch
        img, label = img.to(self.device), label.to(self.device)

        with torch.no_grad():
            out, intermid = self.model(img, eval_=True)
            loss = self.criterion(out, label)

        self.epoch_loss += loss * img.size(0)
        self.epoch_corr += out.argmax(dim=-1).eq(label).sum(-1)

        pred = out 
        for j, i in enumerate(intermid):
            e = entropy_from_logits(i)
            self.entropy_count[j] += e
            if e <= T: 
                pred = i  
                self.branch_count[j] += 1
                break

        self.epoch_corr_list += pred.argmax(dim=-1).eq(label).sum(-1)

    def fit(self, train_dl, test_dl):
        train_loss = []
        val_loss = []
        best_accuracy = -1.0
        total_steps = len(train_dl) * self.epochs
        for epoch in range(1, self.epochs+1):
            print("Epoch: ", epoch)
            print("LR: ", self.scheduler.get_last_lr()[0])
            self.train_loss = 0.0
            num_imgs = 0.

            self.layer_grad_norm_bp_epoch = [[] for _ in range(16)]
            for step, batch in enumerate(train_dl):
                self._train_one_step(batch)
                num_imgs += batch[0].size(0)
            
            self.train_loss /= num_imgs
            print("Train loss: ", self.train_loss.item())
           
            if epoch > 280 and epoch % 10 == 0: # else save every x epoch
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()
                },filename=(f'EE2_checkpoints/acn_{epoch}.pth.tar' if self.args.arch == 'acn' else f'EE2_checkpoints/res_{epoch}.pth.tar')) 
            
            self.scheduler.step()
             
            num_imgs = 0.
            self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.
            self.epoch_corr_list = 0.
            self.branch_count = 17*[0.]
            self.entropy_count = 17*[0.]
            for batch in test_dl:
                self._test_one_step(batch)
                num_imgs += batch[0].size(0)
            self.epoch_loss /= num_imgs
            self.epoch_acc = self.epoch_corr / num_imgs
            self.epoch_acc_list = self.epoch_corr_list / num_imgs  

            print("Val loss: ", self.epoch_loss.item())
            print("Full inference Val acc: ", self.epoch_acc.item())
            print("EE Acc.:", self.epoch_acc_list.item())
            self.branch_count.append(num_imgs - sum(self.branch_count))
            print('Speedup: ', self.branch_count)
            print('Avg. Entropy: ', [e / num_imgs for e in self.entropy_count])
            print('')
            
            train_loss.append(self.train_loss.item())
            val_loss.append(self.epoch_loss.item())

        return train_loss, val_loss


    def eval(self, test_dl):
        T = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for t in T:
            num_imgs = 0.
            self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.
            self.epoch_corr_list = 0.
            self.branch_count = 17*[0.]
            self.entropy_count = 17*[0.]
            for batch in test_dl:
                self._test_one_step(batch, t)
                num_imgs += batch[0].size(0)
            self.epoch_loss /= num_imgs
            self.epoch_acc = self.epoch_corr / num_imgs
            self.epoch_acc_list = self.epoch_corr_list / num_imgs

            print('#####')
            print('Entroppy threshold: ', t)
            print('#####')
            print("Full inference acc. (no EE): ", self.epoch_acc.item())
            print("EE Acc.:", self.epoch_acc_list.item())
            self.branch_count.append(num_imgs - sum(self.branch_count))
            print('Speedup: ', self.branch_count)
            print('')

        return train_loss, val_loss

    def eval_analyze(self, test_dl):
        analyzer = RepresentationAnalyzer()
        num_imgs = 0.
        self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.
        self.epoch_corr_list = 0.
        self.branch_count = 17*[0.]
        self.entropy_count = 17*[0.]
        self.model.eval()
        for batch in test_dl:
            img, labels = batch
            img = img.to(self.device)
            with torch.no_grad():
                out, intermid = self.model(img, eval_=True)
    
            acn_features = resnet_features = out
            img, labels = batch        
            cka_score = analyzer.cka(acn_features, resnet_features)
            isotropy_acn = analyzer.isotropy_measure(acn_features)
            isotropy_resnet = analyzer.isotropy_measure(resnet_features)
            nc_acn = analyzer.neural_collapse_metrics(acn_features, labels)
            nc_resnet = analyzer.neural_collapse_metrics(resnet_features, labels)
            print_comparison_results(cka_score, isotropy_acn, isotropy_resnet, nc_acn, nc_resnet)
            break
        return train_loss, val_loss


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)             

def entropy_from_logits(logits):
    # logits: tensor shape [num_classes] or [1, num_classes]
    probs = F.softmax(logits, dim=-1)  # compute probabilities
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
    return entropy
