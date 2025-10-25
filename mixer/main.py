import torch
from dataloader import get_dataloaders
from dataloader import get_filtered_dataloaders
from train import Trainer
import sys 
import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='res', choices=['res', 'acn', 'concat', 'denseformer'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=300, help='Num Epochs')
    args = parser.parse_args()
    NUM_CLASSES=10
    #train_dl, test_dl = get_filtered_dataloaders(NUM_CLASSES)
    train_dl, test_dl = get_dataloaders()
    if args.arch == 'res':
        print('Res-Mixer')
        from res_mixer import MLPMixer
        model = MLPMixer(
            in_channels=3,
            img_size=32,
            hidden_size=128, #128
            patch_size=4,
            hidden_c=512,
            hidden_s=64,
            num_layers=16, 
            num_classes=NUM_CLASSES,
            drop_p=0.,
            off_act=False,
            is_cls_token=True,
            vanilla=True
            )
    elif  args.arch == 'acn':
        print('AC-Mixer')
        from ac_mixer import MLPMixer
        model = MLPMixer(
            in_channels=3,
            img_size=32,
            hidden_size=128, #128
            patch_size=4,
            hidden_c=512,
            hidden_s=64,
            num_layers=16,
            num_classes=NUM_CLASSES,
            drop_p=0.,
            off_act=False,
            is_cls_token=True,
            vanilla=False
            )
    elif  args.arch == 'concat':
        print('Concat-Mixer')
        from concat_mixer import MLPMixer
        model = MLPMixer(
            in_channels=3,
            img_size=32,
            hidden_size=128, #128
            patch_size=4,
            hidden_c=512,
            hidden_s=64,
            num_layers=10,
            num_classes=NUM_CLASSES,
            drop_p=0.,
            off_act=False,
            is_cls_token=True,
            vanilla=False
            )
    elif  args.arch == 'denseformer':
        print('Denseformer-Mixer')
        from denseformer_mixer import MLPMixer
        model = MLPMixer(
            in_channels=3,
            img_size=32,
            hidden_size=128, #128
            patch_size=4,
            hidden_c=512,
            hidden_s=64,
            num_layers=16,
            num_classes=NUM_CLASSES,
            drop_p=0.,
            off_act=False,
            is_cls_token=True,
            vanilla=False
            )
    else:
        print('Arch error!')
        sys.exit(1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    model.to(device)
     
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of trainable parameters: {total_params}')
    trainer = Trainer(model, device, args)
    train_loss, test_loss = trainer.fit(train_dl, test_dl)
