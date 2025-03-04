import torch
import torchvision.transforms as transforms
import json
from torchvision import models
from PIL import Image
from utils import TRANSFORMS
import numpy as np
def convert_to_2channel(img):
    # result = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    result = 0.2989 * img[0,:, :] + 0.5870 * img[1,:, :] + 0.1140 * img[2,:, :]
    return result
def normalize(image, mean = None, std = None):
    if mean == None:
       mean = [0.485, 0.456, 0.406]
    if std == None:
       std = [0.229, 0.224, 0.225]
    if image.max() > 1:
        image = image.astype(np.float64)
        image /= 255
    image = (image - mean) / std
    # in addition, roll the axis so that they suit pytorch
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()

def get_sample_data(image_ind, images_path, masks_path, transform = TRANSFORMS):
  image_path = images_path[image_ind]
  mask_path = masks_path[image_ind]
  image_raw_name = image_path.split("/")[-1].split(".")[0]

  # Load the image
  image = Image.open(image_path)
  mask = Image.open(mask_path)
  transformed_image = transform(image)
  transformed_mask = transform(mask)
  return image_raw_name, transformed_image, transformed_mask
def get_neutral_background(image):
  height, width = image.shape[:2]
  corner_size = int(0.1 * height)  # This will be 22 pixels for your 224x224 image
  top_left = image[:corner_size, :corner_size, :]
  top_right = image[:corner_size, -corner_size:, :]
  bottom_left = image[-corner_size:, :corner_size, :]
  bottom_right = image[-corner_size:, -corner_size:, :]
  average_top_left = np.mean(top_left, axis=(0, 1))
  average_top_right = np.mean(top_right, axis=(0, 1))
  average_bottom_left = np.mean(bottom_left, axis=(0, 1))
  average_bottom_right = np.mean(bottom_right, axis=(0, 1))
  average_all_corners = np.mean([average_top_left, average_top_right, average_bottom_left, average_bottom_right], axis=0)
  average_all_corners_broadcasted = average_all_corners[np.newaxis, np.newaxis, :]
  return average_all_corners_broadcasted