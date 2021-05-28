run_name = 'CELEBA'

# GPU - CHECK BEFORE EACH RUN (Nvidia DGX-1)
pt_gpu_num = 3
npt_gpu_num = 1

# Test Loop Size
min_size = 1000
max_size = 50000

# Encoder Settings
encoder = 'models/celeba_encoder_trained.pt'
num_channels = (96,64,32,32)
num_res = 64
num_kernels = 5
num_hidden = 512

# LR
pt_lr = 10
npt_lr = 0.3

img_shape = (64, 64)
img_channels_shape = (1,64,64)

