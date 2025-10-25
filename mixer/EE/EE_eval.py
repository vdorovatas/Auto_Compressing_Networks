import torch
from dataloader import get_dataloaders
from dataloader import get_filtered_dataloaders
from EE_train import Trainer
import sys 
import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='res', choices=['res', 'acn', 'hybrid', 'clamped_hybrid'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=300, help='Num Epochs')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    args = parser.parse_args()
    NUM_CLASSES=10
    #train_dl, test_dl = get_filtered_dataloaders(NUM_CLASSES)
    train_dl, test_dl = get_dataloaders()
    if args.arch == 'res':
        print('Res-Mixer')
        from EE_res_mixer import MLPMixer
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
    elif  args.arch == 'hybrid' or args.arch == 'clamped_hybrid':
        if args.arch == 'hybrid': print('Hybrid-Mixer')
        else: print('clamped  Hybrid-Mixer')
        from hybrid_mixer import MLPMixer
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
    
    ####
    #checkpoint = torch.load(args.checkpoint, weights_only=False)
    #model.load_state_dict(checkpoint['state_dict'], strict=True)
    ####
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    model.to(device)
    ########
    from thop import profile
    from thop import clever_format
    model.to('cpu')
    dummy_input = torch.randn(1, 3, 32, 32)  # adjust shape
    flops, params = profile(model, inputs=(dummy_input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"GFLOPs: {flops}")
    sys.exit()
    ########
    trainer = Trainer(model, device, args)
    trainer.eval(test_dl)
