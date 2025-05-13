import h5py
import PIL.Image as Image
import numpy as np
import os
import glob
import torch
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torchvision import transforms
from model import CANNet
import matplotlib.pyplot as plt
import cv2


transform_msi = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_sar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.359, 0.359, 0.359], std=[0.210, 0.210, 0.210])
])

# Image Paths
img_folder_msi = "vessel_dataset_for_CAN/test_data/images"
img_folder_sar = "vessel_dataset_for_CAN/test_data/images_sar"

img_paths = sorted(glob.glob(os.path.join(img_folder_msi, '*.tif')))

# Load CANNet model
model = CANNet()
model = model.cuda()
checkpoint = torch.load('vessel_dataset_for_CAN/multi_modal_model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

pred = []
gt = []

for img_path in img_paths:
    img_msi = Image.open(img_path).convert('RGB')
    img_msi = transform_msi(img_msi)  # (3, H, W)

    sar_path = img_path.replace(img_folder_msi, img_folder_sar)
    img_sar = Image.open(sar_path).convert('RGB')
    img_sar = transform_sar(img_sar)  # (3, H, W)
    img_sar = img_sar[:1, :, :]  #  (1, H, W)

    img = torch.cat([img_msi, img_sar], dim=0)  # (4, H, W)
    img = img.unsqueeze(0).cuda()  # (1, 4, H, W)
    
    _, _, h, w = img.shape
    h_d, w_d = h // 2, w // 2

    img_1 = Variable(img[:, :, :h_d, :w_d])
    img_2 = Variable(img[:, :, :h_d, w_d:])
    img_3 = Variable(img[:, :, h_d:, :w_d])
    img_4 = Variable(img[:, :, h_d:, w_d:])

    density_1 = model(img_1).data.cpu().numpy()
    density_2 = model(img_2).data.cpu().numpy()
    density_3 = model(img_3).data.cpu().numpy()
    density_4 = model(img_4).data.cpu().numpy()


    gt_path = img_path.replace('.tif', '.h5').replace('images', 'ground_truth_h5')
    with h5py.File(gt_path, 'r') as gt_file:
        groundtruth = np.asarray(gt_file['density'])

    
    # Save pred density maps
    H, W = img.shape[2:4]
    h_out, w_out = density_1[0, 0].shape

    density_full = np.zeros((2 * h_out, 2 * w_out), dtype=np.float32)
    density_full[:h_out, :w_out] = density_1[0, 0]
    density_full[:h_out, w_out:] = density_2[0, 0]
    density_full[h_out:, :w_out] = density_3[0, 0]
    density_full[h_out:, w_out:] = density_4[0, 0]

    # Upsample back to the original image size
    density_resized = cv2.resize(density_full, (W, H), interpolation=cv2.INTER_CUBIC)

    # Save pred density maps as .png
    save_dir = "./density_pred_results"
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.basename(img_path).replace(".tif", ".png")
    save_path = os.path.join(save_dir, filename)
    plt.imsave(save_path, density_resized, cmap='jet')
    
    # Compute predicted count
    pred_sum = density_1.sum() + density_2.sum() + density_3.sum() + density_4.sum()
    pred.append(pred_sum)
    gt.append(np.sum(groundtruth))

mae = mean_absolute_error(pred, gt)
mse = mean_squared_error(pred, gt)
rmse = np.sqrt(mean_squared_error(pred, gt))

print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
