cd ..

wandb disabled

for ckpt in ./save/rn50_50ep/checkpoint.pth; do

    echo $ckpt
    python -m torch.distributed.run --nproc_per_node=1 classification/train.py \
        --model 'resnet50' \
        --batch-size 128 \
        --test-only \
        --resume $ckpt

done