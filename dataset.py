from __future__ import print_function

import numpy as np
import torch
from torchvision import datasets, transforms
import random as RND
from random import *
import os
import csv
from torch.utils.data import Dataset
from PIL import Image
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union

class BatchImageFolderInstance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, root, transform=None, crop_size=400, target_transform=None,
                 two_crop=False, jigsaw_transform=None):
        super(BatchImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop
        self.crop_size = crop_size
        self.jigsaw_transform = jigsaw_transform
        self.use_jigsaw = (jigsaw_transform is not None)
        self.num = self.__len__() 
        self.class_to_idx_dict = {}  
        self.find_class_to_index()
        self.transform_crop = transforms.transforms.RandomCrop(self.crop_size)

    def find_class_to_index(self):
        for i in range(len(self.classes)):
            self.class_to_idx_dict[i] = []
        for index in range(len(self.targets)):    
            self.class_to_idx_dict[self.targets[index]].append(index)
        # min_length = 10000    
        # for key in self.class_to_idx_dict:
        #     # print(len(self.class_to_idx_dict[key]))
        #     if min_length>len(self.class_to_idx_dict[key]):
        #         min_length = len(self.class_to_idx_dict[key])
        # # print("min_length: ", str(min_length))       

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """

        path, target = self.imgs[index]

        fb_size = 18
        
        ## check if size of class < 20
        while len(self.class_to_idx_dict[target])<=2*fb_size:
            target = target+1
            print('changing target class due to lack of samples')
            if target>=len(self.classes):
                target = 0

        class_idx = target
        temp_idx1 = randint(0,len(self.class_to_idx_dict[class_idx])-fb_size-fb_size//2)
        temp_idx2 = randint(min(temp_idx1+fb_size//2,len(self.class_to_idx_dict[class_idx])-fb_size), len(self.class_to_idx_dict[class_idx])-fb_size)

        first_segment = self.class_to_idx_dict[class_idx][temp_idx1:temp_idx1+fb_size]
        second_segment = self.class_to_idx_dict[class_idx][temp_idx2:temp_idx2+fb_size]

        for i in range(fb_size):
            path, target = self.imgs[first_segment[i]]
            # print(path)
            image = self.loader(path)
            if self.transform is not None:
                img = self.transform(image)
                if self.two_crop:
                    # print('two_crop')
                    path2, target2 = self.imgs[second_segment[i]]
                    # print(path2)
                    image2 = self.loader(path2)
                    img2 = self.transform(image2)
                    current_img = torch.cat([img, img2], dim=0).unsqueeze(0)  # 6, H, W
                    if i==0:
                        final_image = current_img
                    else:
                        final_image = torch.cat((final_image, current_img),0) 
                    # print(final_image.shape)

        image_chunk = self.transform_crop(final_image) 

        return image_chunk, self.targets[first_segment[0]]


class VideoFolderInstance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, root, transform=None, target_transform=None,
                 two_crop=False, jigsaw_transform=None):
        super(VideoFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop
        self.jigsaw_transform = jigsaw_transform
        self.use_jigsaw = (jigsaw_transform is not None)
        self.num = self.__len__()
        self.min_side=512.0
        self.crop_transform(crop_size=512, crop_type='center')

    def crop_transform(self, crop_size='512', crop_type='center'):
        if crop_type == 'center':
            self.transform_crop = transforms.transforms.CenterCrop(crop_size)
        elif crop_type == 'random':
            self.transform_crop = transforms.transforms.RandomCrop(crop_size)

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        path, target = self.imgs[index]
        image = self.loader(path)

        # h_ = image.size[0]//2
        # w_ = image.size[1]//2

        # resize_transform = transforms.Resize(size=[w_,h_])
        # # image
        if self.transform is not None:
            # image = resize_transform(image)
            img = self.transform(image)
            image1 = img
            this_frame_min_side = min(image1.shape[1], image1.shape[2])
            ratio = self.min_side/this_frame_min_side
            new_1st_dim = int(ratio*image1.shape[1])
            new_2nd_dim = int(ratio*image1.shape[2])
            resize_transform = transforms.Resize([new_1st_dim, new_2nd_dim])
            img = resize_transform(img)
            if self.two_crop:
                # print('two_crop')
                if np.random.uniform()<0.5:
                    # img2 = self.transform(image)
                    # print('i flipped')
                    img2 = torch.fliplr(img)
                else:
                    img2 = img
                img = torch.cat([img, img2], dim=0)
        else:
            img = image

        img = self.transform_crop(img)    

        # # jigsaw
        if self.use_jigsaw:
            jigsaw_image = self.jigsaw_transform(image)

        if self.use_jigsaw:
            return img, index, jigsaw_image
        else:
            return img, index

class MOS_dataloader(datasets.ImageFolder):
    def __init__(self, root, transform=None, mode='train', crop_size=None):
        super(MOS_dataloader, self).__init__(root, transform)
        self.transform = transform
        self.crop_size = crop_size

        self.MOS_map = dict()
        self.video_dict = dict()
        self.root = '/work/08804/smishra/ls6/DS_test_data/'
        self.path = []
        self.transform_crop = transforms.RandomCrop(self.crop_size)   

        with open('/work/08804/smishra/ls6/PyContrast/pycontrast/ShareChat_metadata.csv', newline='\n') as csvfile:
            reader = csv.DictReader(csvfile)
            index = 0
            for row in reader:
                self.video_dict[index] = row['video_ID'] 
                self.MOS_map[row['video_ID']] = row['mos']
                self.path.append(os.path.join(os.path.join(self.root, row['video_ID'][:-4]), row['video_ID'][:-4]))
                index+=1

        ## 80-20 division
        if mode == 'train':
            self.imgs = self.imgs[:114483]
        elif mode == 'test':
            self.imgs = self.imgs[114483:]

        self.num = self.__len__()

    def __len__(self) -> int:
        return len(self.imgs)


    def __getitem__(self, index):
        idx_path, video_num = self.imgs[index]

        idx_path = idx_path.split('/')[-2]
        # path = self.path[index]
        target = np.asarray([np.float(self.MOS_map[idx_path+'.mp4'])/100])

        path = os.path.join(os.path.join(self.root, idx_path), idx_path)

        fb_size = 18

        file_list = os.listdir(path)
        file_list.sort()

        temp_idx = randint(0,len(file_list)-fb_size)

        for i in range(temp_idx, temp_idx+fb_size):
            current_image_path = os.path.join(path, file_list[i])
            # print(current_image_path)
            image = self.loader(current_image_path)
            if self.transform is not None:
                img = self.transform(image)

            if i == temp_idx:
                image_chunk = img.unsqueeze(0)
            else:
                image_chunk = torch.cat((image_chunk, img.unsqueeze(0)),0)

        image_chunk = self.transform_crop(image_chunk)       

        return image_chunk, target

## Test MOS_dataloader
def test_dataloader():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
            # transforms.RandomCrop(400,),# scale=(0.5))
            transforms.ToTensor(),
            normalize
        ])
    loader = MOS_dataloader(root = '/work/08804/smishra/ls6/DS_test_data/', transform=train_transform, crop_size=400)
    train_loader = torch.utils.data.DataLoader(
        loader, batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True)

    for idx, data in enumerate(loader):
        print('here')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)

class VideoFolderInstance2(Dataset):
    def __init__(self, root, transform=None, crop_size=400, crop_type='center', target_transform=None,
                 two_crop=True):
        super().__init__()

        self.root = root
        self.data = os.listdir(root)
        self.two_crop = two_crop
        self.num = self.__len__()
        self.fb_size = 10    ## chunk size
        self.transform = transform
        self.target_transform = target_transform
        self.crop_transform(crop_size, crop_type)
        loader: Callable[[str], Any] = default_loader
        self.loader = loader
        self.min_side = 514.0

        ## Define target transforms when MOS available
        if target_transform is not None:
            self.MOS_map = dict()
            with open('/work/08804/smishra/ls6/PyContrast/pycontrast/ShareChat_metadata.csv', newline='\n') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    score = np.array(row['mos'])
                    self.MOS_map[row['video_ID'][:-4]] = torch.from_numpy(score.astype(np.float32))

        # self.min_2nd_dim = 10000000
    
    def crop_transform(self, crop_size='512', crop_type='center'):
        if crop_type == 'center':
            self.transform_crop = transforms.transforms.CenterCrop(crop_size)
        elif crop_type == 'random':
            self.transform_crop = transforms.transforms.RandomCrop(crop_size)

    def __len__(self):
        return len(self.data)    

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        path = self.data[index]

        # print("This batch index = ", index)
        # video_path = os.path.join(os.path.join(self.root, path), path)
        video_path = os.path.join(self.root, path)
        filelist = os.listdir(video_path)
        filelist.sort()

        start_idx_1 = randint(0,len(filelist) - self.fb_size)
        start_idx_2 = randint(0,len(filelist) - self.fb_size)

        for i in range(self.fb_size):
            image_path1 = os.path.join(video_path, filelist[i+start_idx_1])
            image1 = self.loader(image_path1)
            if self.transform is not None:
                image1 = self.transform(image1)
                if i==0:
                    this_frame_min_side = min(image1.shape[1], image1.shape[2])
                    ratio = self.min_side/this_frame_min_side
                    new_1st_dim = int(ratio*image1.shape[1])
                    new_2nd_dim = int(ratio*image1.shape[2])
                    resize_transform = transforms.Resize([new_1st_dim, new_2nd_dim])
                    # padding = (
                    #     int((self.min_side - new_2nd_dim)/2), 
                    #     int((self.min_side - new_1st_dim)/2), 
                    #     int((self.min_side - new_2nd_dim)/2),
                    #     int((self.min_side - new_1st_dim)/2), 
                    # )
                    # pad_transform = transforms.Pad(padding, fill=0, padding_mode='constant')
                    
                    ## Run once with the below commands uncommented to find out min dimension other than 512
                    # max_dim = max(new_1st_dim, new_2nd_dim)

                    # if max_dim < self.min_2nd_dim:
                    #     self.min_2nd_dim = max_dim

                if self.two_crop:
                    image_path2 = os.path.join(video_path, filelist[i+start_idx_2])
                    image2 = self.loader(image_path2)
                    image2 = self.transform(image2)
                    current_concatenated_img = torch.cat([image1, image2], dim=0).unsqueeze(0)  # 6, H, W
                    if i==0:
                        final_image = current_concatenated_img
                    else:
                        final_image = torch.cat((final_image, current_concatenated_img),0)
                else:
                    if i==0:
                        final_image = image1.unsqueeze(0)
                    else:
                        final_image = torch.cat((final_image, image1.unsqueeze(0)),0)

        image_chunk = resize_transform(final_image)    
        # image_chunk = pad_transform(image_chunk)        
        image_chunk = self.transform_crop(image_chunk)

        if self.target_transform is not None:
            index = self.MOS_map[path]
            index = self.target_transform.normalize(index)
        else:
            index = torch.from_numpy(np.array(index))    

        return image_chunk, index.unsqueeze(dim=0)

# def test_dataloader_VideoFolderInstance2():
#     mean = [0.5204, 0.4527, 0.4395]
#     std = [0.2828, 0.2745, 0.2687]
#     # mean = [0.485, 0.456, 0.406]
#     # std = [0.229, 0.224, 0.225]
#     normalize = transforms.Normalize(mean=mean, std=std)

#     train_transform = transforms.Compose([
#             # transforms.RandomCrop(400,),# scale=(0.5))
#             transforms.ToTensor(),
#             normalize
#         ])

#     class target_normalize():
#         def __init__(self, minimum=26.1955, maximum=65.6616):
#             ## minimum and maximum MOS captured from Human Study
#             self.min = minimum
#             self.max = maximum

#         def normalize(self, x):
#             out = (x - self.min)/(self.max - self.min)
#             return out
#     # /work/08804/smishra/ls6/LIVE-ShareChat_Data/train
#     # /work/08804/smishra/ls6/DS_test_data/
#     # loader = VideoFolderInstance2(root='/work/08804/smishra/ls6/DS_test_data/', transform=train_transform, crop_size=512, crop_type='center', two_crop=False, target_transform=target_normalize())
#     loader = VideoFolderInstance2(root='/work/08804/smishra/ls6/LIVE-ShareChat_Data/train', transform=train_transform, crop_size=512, crop_type='center', two_crop=False, target_transform=None)
#     train_loader = torch.utils.data.DataLoader(
#         loader, batch_size=1, shuffle=True,
#         num_workers=0, pin_memory=True)

#     for idx, data in enumerate(train_loader):
#         print(idx)
#         if idx == 0:
#             out = data[0]
#         else:
#             out = torch.cat((out, data[0]), dim=1)

#     import pdb;pdb.set_trace()
    
# test_dataloader_VideoFolderInstance2()

class VideoFolderInstance2_MOS(Dataset):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, root, transform=None, crop_size=400, crop_type='center', target_transform=None,
                 two_crop=True, mode='train'):
        super().__init__()

        self.root = root
        self.data = os.listdir(root)
        self.two_crop = two_crop
        self.fb_size = 10    ## chunk size
        self.transform = transform
        self.target_transform = target_transform
        self.crop_transform(crop_size, crop_type)
        loader: Callable[[str], Any] = default_loader
        self.loader = loader
        self.min_side = 514.0
        self.mode = mode

        self.train_list_len = [2, 5, 20, 49, 87, 72, 43, 17, 4, 2]
        self.test_list_len = [0, 6, 20, 49, 87, 72, 43, 18, 4, 0]

        self.bins = dict()
        self.n_bins = 10
        for i in range(self.n_bins):
            self.bins[i] = []

        ## Define target transforms when MOS available
        if target_transform is not None:
            self.MOS_map = dict()
            with open('/work/08804/smishra/ls6/PyContrast/pycontrast/ShareChat_metadata.csv', newline='\n') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    score = np.array(row['mos'])
                    self.MOS_map[row['video_ID'][:-4]] = self.target_transform.normalize(torch.from_numpy(score.astype(np.float32))
                    )
                    for i in range(self.n_bins): 
                        if 100*self.MOS_map[row['video_ID'][:-4]]<=(i+1)*10:
                            self.bins[i].append(row['video_ID'][:-4])
                            break;
        
        # for i in range(self.n_bins):
        #     random.shuffle(self.bins[i])

        if mode=='train':
            for i in range(self.n_bins):
                self.bins[i] = self.bins[i][0:self.train_list_len[i]]    
        if mode=='test':
            for i in range(self.n_bins):
                self.bins[i] = self.bins[i][self.train_list_len[i]:self.train_list_len[i]+self.test_list_len[i]]  

        self.num = self.__len__()
    
    def crop_transform(self, crop_size='512', crop_type='center'):
        if crop_type == 'center':
            self.transform_crop = transforms.transforms.CenterCrop(crop_size)
        elif crop_type == 'random':
            self.transform_crop = transforms.transforms.RandomCrop(crop_size)

    def __len__(self):
        if self.mode=='train':
            return 10    
        elif self.mode=='test':
            return 8

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        if self.mode=='train':
            # path = RND.choice(self.bins[index])                ## TAKES 20X MORE TIME
            rand_vid_idx = randint(0,len(self.bins[index])-1)
            path = self.bins[index][rand_vid_idx]
        elif self.mode=='test':
            # path = RND.choice(self.bins[index+1])              ## TAKES 20X MORE TIME
            rand_vid_idx = randint(0,len(self.bins[index+1])-1)
            path = self.bins[index+1][rand_vid_idx]

        # print("This batch index = ", index)
        # video_path = os.path.join(os.path.join(self.root, path), path)
        video_path = os.path.join(os.path.join(self.root, path), path)
        filelist = os.listdir(video_path)
        filelist.sort()

        start_idx_1 = randint(0,len(filelist) - self.fb_size)
        start_idx_2 = randint(0,len(filelist) - self.fb_size)

        for i in range(self.fb_size):
            image_path1 = os.path.join(video_path, filelist[i+start_idx_1])
            image1 = self.loader(image_path1)
            if self.transform is not None:
                image1 = self.transform(image1)
                if i==0:
                    this_frame_min_side = min(image1.shape[1], image1.shape[2])
                    ratio = self.min_side/this_frame_min_side
                    new_1st_dim = int(ratio*image1.shape[1])
                    new_2nd_dim = int(ratio*image1.shape[2])
                    resize_transform = transforms.Resize([new_1st_dim, new_2nd_dim])
                    # padding = (
                    #     int((self.min_side - new_2nd_dim)/2), 
                    #     int((self.min_side - new_1st_dim)/2), 
                    #     int((self.min_side - new_2nd_dim)/2),
                    #     int((self.min_side - new_1st_dim)/2), 
                    # )
                    # pad_transform = transforms.Pad(padding, fill=0, padding_mode='constant')
                    
                    ## Run once with the below commands uncommented to find out min dimension other than 512
                    # max_dim = max(new_1st_dim, new_2nd_dim)

                    # if max_dim < self.min_2nd_dim:
                    #     self.min_2nd_dim = max_dim

                if self.two_crop:
                    image_path2 = os.path.join(video_path, filelist[i+start_idx_2])
                    image2 = self.loader(image_path2)
                    image2 = self.transform(image2)
                    current_concatenated_img = torch.cat([image1, image2], dim=0).unsqueeze(0)  # 6, H, W
                    if i==0:
                        final_image = current_concatenated_img
                    else:
                        final_image = torch.cat((final_image, current_concatenated_img),0)
                else:
                    if i==0:
                        final_image = image1.unsqueeze(0)
                    else:
                        final_image = torch.cat((final_image, image1.unsqueeze(0)),0)

        image_chunk = resize_transform(final_image)    
        # image_chunk = pad_transform(image_chunk)        
        image_chunk = self.transform_crop(image_chunk)

        if self.target_transform is not None:
            index = self.MOS_map[path]
            # index = self.target_transform.normalize(index)
        else:
            index = torch.from_numpy(np.array(index))    

        return image_chunk, index.unsqueeze(dim=0)

def test_dataloader_VideoFolderInstance2():
    mean = [0.5204, 0.4527, 0.4395]
    std = [0.2828, 0.2745, 0.2687]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
            # transforms.RandomCrop(400,),# scale=(0.5))
            transforms.ToTensor(),
            normalize
        ])

    class target_normalize():
        def __init__(self, minimum=26.1955, maximum=65.6616):
            ## minimum and maximum MOS captured from Human Study
            self.min = minimum
            self.max = maximum

        def normalize(self, x):
            out = (x - self.min)/(self.max - self.min)
            return out
    # /work/08804/smishra/ls6/LIVE-ShareChat_Data/train
    # /work/08804/smishra/ls6/DS_test_data/
    loader = VideoFolderInstance2(root='/work/08804/smishra/ls6/LIVE-ShareChat_MOS_videos/test_evaluate', transform=train_transform, crop_size=512, crop_type='center', two_crop=False, target_transform=target_normalize(), mode='test')
    # loader = VideoFolderInstance2(root='/work/08804/smishra/ls6/LIVE-ShareChat_Data/train', transform=train_transform, crop_size=512, crop_type='center', two_crop=False, target_transform=None)
    train_loader = torch.utils.data.DataLoader(
        loader, batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True)

    for idx, data in enumerate(train_loader):
        print(idx)
        if idx == 0:
            out = data[0]
        else:
            out = torch.cat((out, data[0]), dim=1)

    import pdb;pdb.set_trace()
    
# test_dataloader_VideoFolderInstance2()
