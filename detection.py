import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import Adam
import albumentations as A
import albumentations.pytorch.transforms as apt

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import PIL.Image
import numpy as np
import random
import glob
import copy


def visualize(image, coords):
    image_with_dots = copy.deepcopy(image)
    for i in range(14):
        pt_x, pt_y = int(coords[i][0]), int(coords[i][1])
        image_with_dots[pt_y-1:pt_y+1, pt_x-1:pt_x+1] = (255, 0, 0)
    plt.figure()
    plt.imshow(image_with_dots)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RESIZE_HEIGHT = 160
RESIZE_WIDTH = 160


def check_keypoints(keypoints, image_shape):
    '''
    keypoints: list [[x1, y1], ...]
    image_shape: np.array (c x h x w) or (h, w, c)
    '''
    if image_shape[0] == 3:
        h, w = image_shape[1:]
    else:
        h, w = image_shape[:2]
    for pair in keypoints:
        x, y = pair[0], pair[1]
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
    return True


def remake_gt(gt):
    gt_new = {}
    for img_file in gt:
        coords = gt[img_file]
        new_coords = []
        for i in range(14):
            new_coords.append([coords[2 * i], coords[2 * i + 1]])
        gt_new[img_file] = new_coords
    return gt_new


class ImageDataset(Dataset):
    def __init__(
        self,
        mode,
        root_dir,
        transform,
        gt=None,
        train_fraction=0.9,
        split_seed=42
    ):
        # We can't store all the images in memory at the same time,
        # because sometimes we have to work with very large datasets
        # so we will only store data paths.

        rng = random.Random(split_seed)
        # Make sure that train and validation splits
        # use the same (random) order of samples
        img_paths = sorted(glob.glob(f"{root_dir}/*"))
        split = int(train_fraction * len(img_paths))
        rng.shuffle(img_paths)

        if mode == "train":
            img_paths = img_paths[:split]
        elif mode == "val":
            img_paths = img_paths[split:]
        elif mode == 'test':
            img_paths = img_paths
        else:
            raise RuntimeError(f"Invalid mode: {mode!r}")

        self.mode = mode
        self.img_paths = img_paths
        self.transform = transform
        
        if mode != 'test':
            self.gt = gt

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_name = img_path[-9:]
        image = np.array(PIL.Image.open(img_path).convert('RGB'))

        if self.mode == 'train':
            keypoints = self.gt[img_name]
            
            if not check_keypoints(keypoints, image.shape):
                img_path = self.img_paths[0]
                img_name = img_path[-9:]
                image = np.array(PIL.Image.open(img_path).convert('RGB'))
                keypoints = self.gt[img_name]
            
            #visualize(image, keypoints)
            while True:
                transformed = self.transform(image=image, keypoints=keypoints)
                if check_keypoints(transformed['keypoints'], transformed['image'].shape):
                    #visualize(transformed['image'], transformed['keypoints'])
                    #assert False
                    keypoints_flat = torch.tensor(np.array(transformed['keypoints']).flatten())
                    return transformed['image'], keypoints_flat
        elif self.mode == 'val':
            keypoints = self.gt[img_name]
            keypoints_flat = torch.tensor(np.array(keypoints).flatten())
            return self.transform(image=image)['image'], keypoints_flat, torch.tensor(image.shape)
        else:
            return self.transform(image=image)['image'], torch.tensor(image.shape), img_name


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.lin = nn.Linear(in_channels, in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        att_weights = self.lin(self.flatten(self.aap(self.bn1(x))))
        return self.bn2(x * att_weights[..., None, None])


class MyBlock(nn.Module):
    def __init__(self, in_channels, same_dim=True):
        super().__init__()
        
        if same_dim:
            out_channels = in_channels
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
            self.scale = nn.Identity()
        else:
            out_channels = 2 * in_channels
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.scale = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_sep2 = nn.Sequential(                   # separable convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding='same')
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.bn2(self.conv_sep2(self.relu1(self.bn1(self.conv1(x)))))
        return self.relu2(out + self.scale(x))


class MyBlockFast(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels // 4
        out_channels = in_channels
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding='same')
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding='same')
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)) + x)
        return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_in = nn.Conv2d(3, 66, kernel_size=5, stride=1, padding='same')
        self.bn_in = nn.BatchNorm2d(66)
        self.relu_in = nn.ReLU(inplace=True)
        self.maxpool_in = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.mb11 = MyBlock(66)
        self.mb12 = MyBlock(66)
        
        self.mb21 = MyBlock(66, False)
        self.mb22 = MyBlock(132)
        self.mb23 = MyBlock(132)
        
        self.ab1 = AttentionBlock(132)
        
        self.mb31 = MyBlock(132, False)
        self.mb32 = MyBlock(264)
        self.mb33 = MyBlock(264)
        self.mb34 = MyBlock(264)
        
        self.ab2 = AttentionBlock(264)
        
        self.mb41 = MyBlock(264, False)
        self.mb42 = MyBlock(528)
        self.mb43 = MyBlock(528)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        self.lin1 = nn.Linear(528, 100)
        self.relu1 = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(100, 28)

    def forward(self, x):
        out = self.maxpool_in(self.relu_in(self.bn_in(self.conv_in(x))))
        out = self.mb12(self.mb11(out))
        out = self.mb23(self.mb22(self.mb21(out)))
        out = self.ab1(out)
        out = self.mb34(self.mb33(self.mb32(self.mb31(out))))
        out = self.ab2(out)
        out = self.mb43(self.mb42(self.mb41(out)))
        out = self.flatten(self.avgpool(out))
        return self.lin2(self.relu1(self.lin1(out)))


def validate_detector(model, img_dir, gt):
    '''
    gt: dict image_filename -> 28 coords x1, y1, ..., x14, y14
    img_dir: directory with images
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    gt_new = remake_gt(gt)

    common_transforms = [
        A.Resize(height=RESIZE_HEIGHT, width=RESIZE_WIDTH, p=1),
        A.ToFloat(max_value=255),
        A.Normalize(max_pixel_value=1.0, mean=IMAGENET_MEAN, std=IMAGENET_STD),
        apt.ToTensorV2(),
    ]
    val_transforms = A.Compose(common_transforms)
    val_dataset = ImageDataset(mode='val', root_dir=img_dir, transform=val_transforms, gt=gt_new)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False)

    val_mse_list = []
    model.eval()
    with torch.no_grad():
        for img_batch, coords_batch, shapes_batch in val_dataloader:
            img_batch = img_batch.to(device)
            coords_batch = coords_batch.to(device)
            shapes_batch = shapes_batch.to(device)
            
            coords_pred = model(img_batch)
            coords_pred[:, ::2] *= (100 / RESIZE_WIDTH)
            coords_pred[:, 1::2] *= (100 / RESIZE_HEIGHT)
            
            coords_batch[:, ::2] *= (100 / shapes_batch[:, 1].reshape(-1, 1))
            coords_batch[:, 1::2] *= (100 / shapes_batch[:, 0].reshape(-1, 1))
            mse = F.mse_loss(input=coords_pred, target=coords_batch.to(device))
            val_mse_list.append(mse.item())

    return np.mean(val_mse_list)


def train_detector(train_gt, train_img_dir, fast_train):
    '''
    train_gt: dict image_filename -> 28 coords x1, y1, ..., x14, y14
    train_img_dir: directory with images
    fast_train: train mode
    '''

    gt_new = remake_gt(train_gt)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    augmentations = [
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5, border_mode=0),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.CLAHE(clip_limit=3, tile_grid_size=(20, 20), p=0.2),
    ]
    common_transforms = [
        A.Resize(height=RESIZE_HEIGHT, width=RESIZE_WIDTH, p=1),
        A.ToFloat(max_value=255),
        A.Normalize(max_pixel_value=1.0, mean=IMAGENET_MEAN, std=IMAGENET_STD),
        apt.ToTensorV2(),
    ]
    
    if not fast_train:
        batch_size = 64
    else:
        batch_size = 2
    
    train_transforms = A.Compose(augmentations + common_transforms, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    train_dataset = ImageDataset(mode='train', root_dir=train_img_dir, transform=train_transforms, gt=gt_new)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = ModelNew().to(device)
        
    optimizer = Adam(model.parameters())
    
    if fast_train:
        num_epochs = 1
        max_batch_count = 2
    else:
        num_epochs = 120
        max_batch_count = 1e10
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    model.train()
    train_mse_list = []
    val_every = 5
    val_mse_list = [0] * num_epochs
    for epoch in tqdm(range(num_epochs), desc=f'Training'):
        print(f'Epoch {epoch}')
        epoch_train_mse_list = []
        
        batch_counter = 0
        for img_batch, coords_batch in train_dataloader:
            img_batch = img_batch.to(device)
            coords_batch = coords_batch.to(device).float()

            coords_pred = model(img_batch).float()
            optimizer.zero_grad()
            loss = F.mse_loss(input=coords_pred, target=coords_batch)
            loss.backward()
            optimizer.step()
            epoch_train_mse_list.append(loss.item())
            
            batch_counter += 1
            if batch_counter == max_batch_count:
                break

        scheduler.step()
        train_mse_list.append(np.mean(epoch_train_mse_list))
        print('Train mse', train_mse_list[-1])

        if epoch % val_every == 0:
            val_mse = validate_detector(model, train_img_dir, train_gt)
            val_mse_list[epoch] = val_mse
            print('Val mse', val_mse)
            if not fast_train:
                torch.save(model.state_dict(), 'facepoints_model_new_new.pt')
    
    if not fast_train:
        plt.scatter(np.arange(num_epochs), train_mse_list)
        plt.scatter(np.arange(num_epochs), val_mse_list)

    return model


def detect(model_filename, test_img_dir):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = ModelNew().to(device)
    sd = torch.load(model_filename, map_location=device, weights_only=True)
    model.load_state_dict(sd)

    common_transforms = [
        A.Resize(height=RESIZE_HEIGHT, width=RESIZE_WIDTH, p=1),
        A.ToFloat(max_value=255),
        A.Normalize(max_pixel_value=1.0, mean=IMAGENET_MEAN, std=IMAGENET_STD),
        apt.ToTensorV2(),
    ]
    test_transforms = A.Compose(common_transforms)
    test_dataset = ImageDataset(mode='test', root_dir=test_img_dir, transform=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)
    
    facepoints = {}    # dict: img_filename -> [x1, y1, ..., x14, y14]
    model.eval()
    with torch.no_grad():
        for image_batch, shapes_batch, image_names in tqdm(test_dataloader, desc='Test', total=len(test_dataloader)):
            shapes_batch = shapes_batch.to(device)
            image_batch = image_batch.to(device)
            
            coords_pred = model(image_batch)
            coords_pred[:, ::2] *= (shapes_batch[:, 1].reshape(-1, 1) / RESIZE_WIDTH)
            coords_pred[:, 1::2] *= (shapes_batch[:, 0].reshape(-1, 1) / RESIZE_HEIGHT)
            for i in range(len(coords_pred)):
                facepoints[image_names[i]] = coords_pred[i].tolist()

    return facepoints
