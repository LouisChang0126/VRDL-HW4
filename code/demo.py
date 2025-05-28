import argparse
import subprocess
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from net.model_PromptIR import PromptIR
from net.model_AdaIR import AdaIR
from utils.dataset_utils import TestSpecificDataset
from utils.image_io import save_image_tensor
import lightning.pytorch as pl
import torch.nn.functional as F
import torch.nn as nn
import os
from PIL import Image
import zipfile

def apply_tta(model, input_):
    """
    Apply Test-Time Augmentation (TTA) with multiple transformations:
    original, horizontal flip, vertical flip, 90°/180°/270° rotations, diagonal flip.
    """
    b, c, h, w = input_.shape
    outputs = []

    # Original input
    output = model(input_)
    outputs.append(output)

    # Horizontal flip
    input_hflip = torch.flip(input_, dims=[3])
    output_hflip = model(input_hflip)
    outputs.append(torch.flip(output_hflip, dims=[3]))

    # Vertical flip
    input_vflip = torch.flip(input_, dims=[2])
    output_vflip = model(input_vflip)
    outputs.append(torch.flip(output_vflip, dims=[2]))

    # 90° rotation (transpose + horizontal flip)
    input_rot90 = torch.transpose(input_, 2, 3)
    input_rot90 = torch.flip(input_rot90, dims=[3])
    output_rot90 = model(input_rot90)
    output_rot90 = torch.flip(output_rot90, dims=[3])
    output_rot90 = torch.transpose(output_rot90, 2, 3)
    outputs.append(output_rot90)

    # 180° rotation (horizontal + vertical flip)
    input_rot180 = torch.flip(input_, dims=[2, 3])
    output_rot180 = model(input_rot180)
    outputs.append(torch.flip(output_rot180, dims=[2, 3]))

    # 270° rotation (horizontal flip + transpose)
    input_rot270 = torch.flip(input_, dims=[3])
    input_rot270 = torch.transpose(input_rot270, 2, 3)
    output_rot270 = model(input_rot270)
    output_rot270 = torch.transpose(output_rot270, 2, 3)
    output_rot270 = torch.flip(output_rot270, dims=[3])
    outputs.append(output_rot270)

    # Diagonal flip (transpose)
    input_diag = torch.transpose(input_, 2, 3)
    output_diag = model(input_diag)
    outputs.append(torch.transpose(output_diag, 2, 3))

    # Anti-diagonal flip (transpose + horizontal flip + vertical flip)
    input_anti_diag = torch.transpose(input_, 2, 3)
    input_anti_diag = torch.flip(input_anti_diag, dims=[3])  # Horizontal flip
    input_anti_diag = torch.flip(input_anti_diag, dims=[2])  # Vertical flip
    output_anti_diag = model(input_anti_diag)
    output_anti_diag = torch.flip(output_anti_diag, dims=[2])  # Reverse vertical flip
    output_anti_diag = torch.flip(output_anti_diag, dims=[3])  # Reverse horizontal flip
    output_anti_diag = torch.transpose(output_anti_diag, 2, 3)  # Reverse transpose
    outputs.append(output_anti_diag)

    # Average the outputs
    final_output = torch.mean(torch.stack(outputs, dim=0), dim=0)
    final_output = torch.clamp(final_output, 0, 1)
    return final_output

class AdaIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = AdaIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self,x):
        return self.net(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--test_path', type=str, default="test/degraded/", help='save path of test images, can be directory or an image')
    parser.add_argument('--output_path', type=str, default="output/demo/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="train_ckpt/val-psnr-epoch=148-val_psnr=29.55.ckpt", help='checkpoint save path')
    parser.add_argument('--tta', type=bool, default=True, help='Enable Test-Time Augmentation')
    opt = parser.parse_args()

    ckpt_path = opt.ckpt_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Construct the output dir
    subprocess.check_output(['mkdir', '-p', opt.output_path])

    np.random.seed(0)
    torch.manual_seed(0)

    # Make network
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.cuda)
    net = AdaIRModel.load_from_checkpoint(ckpt_path).to(device)
    net.eval()

    test_set = TestSpecificDataset(opt)
    testloader = DataLoader(test_set, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    print('Start testing...')
    with torch.no_grad():
        for ([clean_name], degrad_patch) in tqdm(testloader):
            degrad_patch = degrad_patch.to(device)

            if opt.tta:
                print("Using Test-Time Augmentation")
                restored = apply_tta(net, degrad_patch)
            else:
                restored = net(degrad_patch)
                    
            save_image_tensor(restored, opt.output_path + clean_name[0] + '.png')

    # Save to .npz and zip
    folder_path = opt.output_path
    output_npz = 'pred.npz'
    images_dict = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            image = Image.open(file_path).convert('RGB')
            img_array = np.array(image)
            img_array = np.transpose(img_array, (2, 0, 1))
            images_dict[filename] = img_array

    np.savez(output_npz, **images_dict)

    zipfile_name = f'{opt.ckpt_name.split(".")[0]}.zip'
    with zipfile.ZipFile(zipfile_name, 'w') as zipf:
        zipf.write(output_npz)

    print(f"Saved {len(images_dict)} images to {zipfile_name}")