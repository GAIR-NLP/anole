# Example Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py
torchrun --nproc_per_node=8 train_image_head.py

# Convert pytorch_model.bin to consolidated.pth
python bin_to_pth.py