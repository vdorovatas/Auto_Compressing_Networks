## Training on ImageNet:

0. clone repo
1. create environment 
2. pip install requirements.txt
3. download imagenet (classic repo structure): wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar
4. train: python3 main.py --lr=... --batch_size=... --weight_decay=... --t_max=... --mode='train' --model_type [vanilla, long] 
5. val: python3 main.py  --batch_size=... --weight_decay=... --model='val' --model_type [vanilla, long]
