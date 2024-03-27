import os
import shutil
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import augly.image as imaugs
import cv2

# setting random seed for reproducibility
np.random.seed(4200)

def orig(image_path:str, out_path:str,image_name:str): 
    output_path = os.path.join(out_path , f'{image_name}_orig.png')
    aug_image = imaugs.pad(image_path)   
    return imaugs.rotate(aug_image,degrees=0,),  output_path
    
    
def rotl8(image_path:str, out_path:str,image_name:str):
    output_path = os.path.join(out_path , f'{image_name}_rotate_L8.png')
    aug_image = imaugs.pad(image_path)
    return imaugs.rotate(aug_image,degrees=8,), output_path
    

def rotr8(image_path:str, out_path:str,image_name:str):
    output_path =  os.path.join(out_path , f'{image_name}_rotate_R8.png')
    aug_image = imaugs.pad(image_path)
    return imaugs.rotate(aug_image,degrees=352,), output_path
    

def rotl15(image_path:str, out_path:str,image_name:str):
    output_path = os.path.join(out_path , f'{image_name}_rotate_L15.png')
    aug_image = imaugs.pad(image_path)
    return imaugs.rotate(aug_image,degrees=15,), output_path
    

def rotr15(image_path:str, out_path:str,image_name:str):
    output_path =os.path.join(out_path , f'{image_name}_rotate_R15.png')
    aug_image = imaugs.pad(image_path)
    return imaugs.rotate(aug_image,degrees=345,), output_path
    

def hflip(image_path:str, out_path:str,image_name:str):
    output_path = os.path.join(out_path , f'{image_name}_hflip.png')
    aug_image = imaugs.pad(image_path)
    return imaugs.hflip(aug_image,), output_path
    

def bright30(image_path:str, out_path:str,image_name:str):
    output_path = os.path.join(out_path , f'{image_name}_bright_30.png')
    aug_image = imaugs.pad(image_path)
    return imaugs.brightness(aug_image,factor=1.3,), output_path
    

def dim30(image_path:str, out_path:str,image_name:str):
    output_path = os.path.join(out_path , f'{image_name}_dim_30.png')
    aug_image = imaugs.pad(image_path)
    return imaugs.brightness(aug_image,factor=0.7, ), output_path
    

def remove_black_pixels(image, output_path):
    # file = os.path.join(self.source_folder, self.file_name)
    # image = cv2.imread(image_path)
    
    #PIL image to CV2 image format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Mask of coloured pixels.
    mask = image > 0

    # Coordinates of coloured pixels.
    coordinates = np.argwhere(mask)

    # Binding box of non-black pixels.
    x0, y0, s0 = coordinates.min(axis=0)
    x1, y1, s1 = coordinates.max(axis=0) + 1  # slices are exclusive at the top

    # Get the contents of the bounding box.
    cropped = image[x0:x1, y0:y1]
    # overwrite the same file
    # file_cropped = os.path.join(self.destination_folder, self.file_name)
    cv2.imwrite(output_path, cropped)



transforms = [rotl8,rotr8,rotl15,rotr15,hflip,bright30,dim30,orig]

def augment_image(image_dir:str, out_dir:str, train_size:int = 500, test_size:int = 200 ):

    """Augments the images present in input_path directory and creates 
    the same structured directory in the ouput_path after augmentation.

    Args:
        image_dir (str): Path of the input image directory
        out_dir (str): path of output directory
        train_size (int, optional): training images per class. Defaults to 500.
        test_size (int, optional): testing images per class. Defaults to 200.
    """
    
    # for classes in os.listdir(image_dir):
    for classes in ['Hypertension']:
        images_in_class = list(os.listdir(os.path.join(image_dir,classes)))
        image_transform_dict = defaultdict(list)
        

        # training set     
        out_path = os.path.join(out_dir,'train',classes)
        os.makedirs(out_path,exist_ok=True)
        # images_train = np.random.choice(images_in_class, train_size, replace=True)
        # print(images_train)
        for i in tqdm(range(train_size),desc=f'Trainset: {classes}'):
            image = np.random.choice(images_in_class, 1)[0]
            # print(image)
            while len(image_transform_dict[image]) == len(transforms):
                 # if a specific image is choosen more than number of augmentations available
                # that image is repeated, so to avoid this we randomly choose another image in its place.
                 images_in_class.remove(image)
                 image = np.random.choice(images_in_class,1)[0]

                
            transform = np.random.choice(transforms, 1)[0]
            while transform in image_transform_dict[image]:
                transform = np.random.choice(transforms, 1)[0]
                # print(transform)

            aug_image, output_path = transform(image_path = os.path.join(image_dir,classes,image) , out_path = out_path, image_name = image.split('.')[-2])
            remove_black_pixels(aug_image, output_path)
            image_transform_dict[image].append(transform)
        
        # testing set
        out_path = os.path.join(out_dir,'test',classes) 
        os.makedirs(out_path,exist_ok=True)
        # images_test = np.random.choice(images_in_class, test_size, replace=True)
        
        for i in tqdm(range(test_size),desc=f'Testset: {classes}'):
            
            image = np.random.choice(images_in_class, 1)[0]
            
            while len(image_transform_dict[image]) == len(transforms):
                 # if a specific image is choosen more than number of augmentations available
                # that image is repeated, so to avoid this we randomly choose another image in its place.
                 images_in_class.remove(image)
                 image = np.random.choice(images_in_class,1)[0]
                 
                
            transform = np.random.choice(transforms, 1)[0]
            while transform in image_transform_dict[image]:
                transform = np.random.choice(transforms, 1)[0]
                # print('transform',  len(image_transform_dict[image]))
            aug_image, output_path = transform(image_path = os.path.join(image_dir,classes,image) , out_path = out_path, image_name = image.split('.')[-2])
            remove_black_pixels(aug_image, output_path)
            image_transform_dict[image].append(transform)
        
if __name__ == '__main__':
    image_dir ='data/ODIR_pure/Training'
    out_dir = 'data/ODIR_Aug_Resample'
    train_size = 500
    test_size = 200 
    augment_image(image_dir,out_dir,train_size,test_size)
    