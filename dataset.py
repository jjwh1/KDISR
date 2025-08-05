import torch
from torch.utils.data import Dataset
import cv2
import os

class InpaintDataset(Dataset):
    def __init__(self, image_dir, mask_dir, gt_dir,large_mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.gt_dir = gt_dir
        self.large_mask_dir = large_mask_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.bmp'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)
        gt_path = os.path.join(self.gt_dir, filename)
        large_mask_path = os.path.join(self.large_mask_dir, filename)

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path)
        large_mask = cv2.imread(large_mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Could not read mask at {mask_path}")
        if gt is None:
            raise FileNotFoundError(f"Could not read ground truth at {gt_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        mask = mask / 255.0
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB) / 255.0
        large_mask = large_mask / 255.0

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        gt = torch.from_numpy(gt).permute(2, 0, 1).float()
        large_mask = torch.from_numpy(large_mask).unsqueeze(0).float()
        input_tensor = torch.cat([image, mask], dim=0)

        return input_tensor, gt, mask, filename, image, large_mask

