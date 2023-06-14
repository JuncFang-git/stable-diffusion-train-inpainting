'''
Author: Juncfang
Date: 2023-06-09 15:06:33
LastEditTime: 2023-06-13 19:47:41
LastEditors: Juncfang
Description: change from 'https://github.com/lorenzo-stacchio/Stable-Diffusion-Inpaint/blob/main/ldm/data/inpainting_dataset.py'
FilePath: /stable-diffusion/personal_workspace/inpainting_dataset.py
 
'''


import os
import random
import copy
import numpy as np
import PIL
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from einops import rearrange


class InpaintingBase(Dataset):
    def __init__(self,
                 data_root:str,
                 split:str,
                 mask_type:str='random',
                 size:int=512,
                 interpolation="bicubic",
                 ):
        """_summary_

        Args:
            data_root (str): Perfect directory structure like below. When mask folder is empty, 
            a random created mask will be used. Otherwise, make sure all file at image/mask folder is strict corresponding.
            
            |-data_root
            |-|-train
            |-|-|-image
            |-|-|-mask
            |-|-|-txt
            |-|-valid
            |-|-|-image
            |-|-|-mask
            |-|-|-txt
            
            split (str): ['train', 'valid']
            mask_type (str): ['random', 'reference', 'hybrid']. Defaults to "random".
            size (int): image/mask size. Defaults to 512.
            interpolation (str, optional): _description_. Defaults to "bicubic".

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        if split not in ['train', 'valid']:
            raise ValueError("InpaintingBase need a 'split' named in ['train', 'valid'] !")
        if mask_type not in ['random', 'reference', 'hybrid']:
            raise ValueError("InpaintingBase need a 'mask_type' in ['train', 'valid'] !")
        
        self.data_folder = os.path.join(data_root, split)
        self.image_folder = os.path.join(self.data_folder, 'image')
        self.mask_folder = os.path.join(self.data_folder, 'mask')
        self.txt_folder = os.path.join(self.data_folder, 'txt')
        self.image_paths = os.listdir(self.image_folder)
        self.mask_paths = os.listdir(self.mask_folder) if os.path.isdir(self.mask_folder) else []
        self.txt_paths = os.listdir(self.txt_folder)
        self.image_paths.sort()
        self.mask_paths.sort()
        self.txt_paths.sort()
        
        self._length_image = len(self.image_paths)
        self._length_mask = len(self.mask_paths)
        self._length_txt = len(self.txt_paths)
        
        self.mask_type = mask_type
        
        if (self._length_mask != self._length_image  or 
            self._length_mask != self._length_txt or 
            self._length_image != self._length_txt):
            if self._length_mask == 0 and self._length_image == self._length_txt:
                print(f"WARNING: Can not got any mask file at {self.mask_folder}. Using random mask by force!")
                self.mask_type = "random"
            else:
                raise ValueError(f"Got {self._length_image} images, {self._length_mask} masks, {self._length_txt} txts, when {split}. \
                    Please check folder of {self.data_folder} to make sure all files paired !!")
        
        self.size = size

        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

        self.labels = {
            "file_path_": [os.path.join(self.image_folder, l) for l in self.image_paths],
            "file_path_txt_": [os.path.join(self.txt_folder, l) for l in self.txt_paths],
        }
        if self._length_mask == 0:
            mask_dict = {"file_path_mask_": ['' for _ in range(self._length_image)]}
        else:
            mask_dict = {"file_path_mask_": [os.path.join(self.mask_folder, l) for l in self.mask_paths]}
        self.labels.update(mask_dict)

    def __len__(self):
        return self._length_image

    # generate random masks
    # Reference to 'https://github.com/huggingface/diffusers/blob/main/examples/research_projects/dreambooth_inpaint/train_dreambooth_inpaint.py'
    def get_random_mask(self, im_shape, ratio=1, mask_full_image=False):
        mask = Image.new("L", im_shape, 0)
        draw = ImageDraw.Draw(mask)
        size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
        # use this to always mask the whole image
        if mask_full_image:
            size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
        limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
        center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
        draw_type = random.randint(0, 1)
        if draw_type == 0 or mask_full_image:
            draw.rectangle(
                (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
                fill=255,
            )
        else:
            draw.ellipse(
                (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
                fill=255,
            )
        
        
        mask = np.array(mask).astype(np.float32)/255.0
        mask = mask[:,:,None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)
        
        return mask
    
    def get_reference_mask(self, mask_path, resize_to):
        mask = Image.open(mask_path).convert("L")
        if mask.size[0]!=resize_to or mask.size[1]!=resize_to:
            mask = mask.resize((resize_to,resize_to), resample=self.interpolation)
        mask = np.array(mask).astype(np.float32)/255.0

        mask = mask[:,:,None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)
        
        return mask
        
    def get_hybrid_mask(self, mask_path, resize_to):
        ref_mask = Image.open(mask_path).convert("L")
        if ref_mask.size[0]!=resize_to or ref_mask.size[1]!=resize_to:
            ref_mask = ref_mask.resize((resize_to,resize_to), resample=self.interpolation)
        w, h = ref_mask.size
        
        mask_arry = np.array(ref_mask)
        center_list = np.where(mask_arry == 255)
        rand_index = random.randint(0, len(center_list[0]))
        center = (center_list[1][rand_index], center_list[0][rand_index]) # x, y - h,w
        
        limits = (min(center[0], w - center[0]), min(center[1], h - center[1]))
        hole_size = (random.randint(0, limits[0]), random.randint(0, limits[1]))
        
        draw_type = random.randint(0, 1)
        mask = Image.fromarray(copy.deepcopy(mask_arry))
        draw = ImageDraw.Draw(mask)
        if draw_type == 0:
            draw.rectangle(
                (center[0] - hole_size[0], center[1] - hole_size[1], center[0] + hole_size[0], center[1] + hole_size[1]),
                fill=255,
            )
        else:
            draw.ellipse(
                (center[0] - hole_size[0], center[1] - hole_size[1], center[0] + hole_size[0], center[1] + hole_size[1]),
                fill=255,
            )
        
        mask = np.array(mask).astype(np.float32)/255.0
        mask = mask[:,:,None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)
        
        return mask
    

    def _transform_and_normalize(self, image_path, mask_path, txt_path, mask_type, resize_to):
        # assert
        if mask_path:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_name = os.path.splitext(os.path.basename(mask_path))[0]
            txt_name = os.path.splitext(os.path.basename(txt_path))[0]
            if not (image_name == mask_name and 
                    image_name == txt_name and 
                    mask_name == txt_name):
                raise ValueError(f"Got unpaired name. image:{image_name}, mask:{mask_name}, txt:{txt_name}")
        else:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            txt_name = os.path.splitext(os.path.basename(txt_path))[0]
            if not image_name == txt_name:
                raise ValueError(f"Got unpaired name. image:{image_name}, txt:{txt_name}")

        # get images
        image = Image.open(image_path).convert("RGB")
        
        if image.size[0]!=resize_to or image.size[1]!=resize_to:
            image = image.resize((resize_to, resize_to), resample=self.interpolation)
        image = np.array(image).astype(np.float32)/255.0 # [0,255] -> [0,1]
        image = torch.from_numpy(image)
        
        # get masks
        if mask_type == "random":
            mask = self.get_random_mask((resize_to, resize_to), 1, False)
        elif mask_type == "reference":
            mask = self.get_reference_mask(mask_path, resize_to)
        elif mask_type == "hybrid":
            mask = self.get_hybrid_mask(mask_path, resize_to)

        # get masked image
        masked_image = (1-mask)*image
        
        # get txt
        with open(txt_path, 'r') as file:
            txt = file.read()

        batch = {"image": image, "mask": mask, "masked_image": masked_image, "caption":txt}

        for k in batch:
            if k == "caption":
                continue
            batch[k] = batch[k] * 2.0 - 1.0 # [0, 1] -> [-1, 1]
            # batch[k] = rearrange(batch[k], 'c h w -> h w c')
        return batch

    def __getitem__(self, i):
    
        example = dict((k, self.labels[k][i]) for k in self.labels)

        add_dict = self._transform_and_normalize(
            example["file_path_"], 
            example["file_path_mask_"],
            example["file_path_txt_"],
            self.mask_type,
            resize_to=self.size)
        
        example.update(add_dict)

        return example


class InpaintingTrain(InpaintingBase):
    def __init__(self, data_root, mask_type, size, interpolation="bicubic", **kwargs):
        super().__init__(data_root, "train", mask_type, size, interpolation, **kwargs)


class InpaintingValidation(InpaintingBase):
    def __init__(self, data_root, mask_type, size, interpolation="bicubic", **kwargs):
        super().__init__(data_root, "valid", mask_type, size, interpolation, **kwargs)


if __name__=="__main__":
    size = 512
    # mask_type="random"
    # mask_type="reference"
    mask_type="hybrid"
    data_root = "/home/juncfang/code/stable-diffusion/personal_workspace/experiments/t0"
    
    
    de_transform =  transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/255, 1/255 ,1/255 ]),
                    ])
    
    de_transform_mask =  transforms.Compose([ transforms.Normalize(mean = [ 0. ],
                                                     std = [ 1/255]),
                    ])
    ip_train = InpaintingTrain(data_root=data_root, mask_type=mask_type,  size=size)
    ip_train_loader = DataLoader(ip_train, batch_size=1, num_workers=4, pin_memory=True, shuffle=True)

    for idx, batch in enumerate(ip_train_loader):
        im_keys = ['image', 'masked_image', 'mask']
        for k in im_keys:
            # print(batch[k].shape)
            image_de = batch[k]
            image_de = (image_de + 1)/2
            image_de = rearrange(image_de, 'b h w c ->b c h w')
            if k=="mask":
                image_de = de_transform_mask(image_de)
            else:
                image_de = de_transform(image_de)
            
            rgb_img = (image_de).type(torch.uint8).squeeze(0)
            
            img = transforms.ToPILImage()(rgb_img)  
            # print(img.size)
            img.save("%s_test.jpg" % k)
