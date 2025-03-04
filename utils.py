
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import torch
import json
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from metrics import deletion_score_batch
import shap


url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
with open(fname) as f:
    CLASS_NAMES = json.load(f)

DEFAULT_DEVICES = 'cuda' if torch.cuda.is_available() else 'cpu'
TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to (224, 224)
    transforms.ToTensor(),          # Convert to PyTorch tensor
])
def get_score_image(image_path, model, class_names = CLASS_NAMES, transform = TRANSFORMS):
  img =Image.open(image_path)
  img_resized = transform(img)
  score = model(img_resized.unsqueeze(0))
  img_index = torch.argmax(score, dim = 1).item()
  softmax_score = torch.nn.functional.softmax(score, dim=1)[0][img_index]
  img_names = class_names[str(img_index)][1]
  return img_names, softmax_score.item()

def get_score_from_array(image, model,img_indice = None, class_names=CLASS_NAMES, device = DEFAULT_DEVICES):
  image = torch.from_numpy(image).permute(0,3,1,2)
  model.to(device)
  image = image.to(device)
  scores = model(image)
  if img_indice is None:
      img_indice = torch.argmax(scores, dim=1).tolist()
  softmax_scores = torch.nn.functional.softmax(scores, dim=1)
  return softmax_scores[0, img_indice]

def getPredictedScore(img,model, classId = None):
    y_pred = model(torch.from_numpy(img).permute(0,3,1,2))
    class_ = torch.argmax(y_pred, dim = 1)
    ans = torch.nn.functional.softmax(y_pred,1)
    maxScore = torch.max(ans, dim = 1)
    if classId:
        return ans[0][classId].item()
    return maxScore[0].item()
def getTrueId(img, model, device = 'cpu'):
    model.to(device)
    img = (torch.from_numpy(img).permute(0,3,1,2)).to(device)
    y_pred = model(img)
    class_ = torch.argmax(y_pred, dim = 1)
    ans = torch.nn.functional.softmax(y_pred,1)
    return class_.item()



def get_partial_score_batch(images,model, img_indices=None, class_names=CLASS_NAMES, device = DEFAULT_DEVICES):
    # batch_images = torch.cat([torch.from_numpy(np.transpose(image, (2, 0, 1))).unsqueeze(0) for image in images], axis = 0)
    batch_images = torch.from_numpy(np.array(images)).permute(0,3,1,2)
    model.to(device)
    batch_images = batch_images.to(device)
    # print(batch_images.device)
    scores = model(batch_images)

    if img_indices is None:
        img_indices = torch.argmax(scores, dim=1).tolist()
    elif isinstance(img_indices, int):
        img_indices = [img_indices] * len(scores)

    softmax_scores = torch.nn.functional.softmax(scores, dim=1)
    softmax_scores_list = softmax_scores[torch.arange(len(img_indices)), img_indices].tolist()
    # softmax_scores_list = [softmax_scores[i, idx].item() for i, idx in enumerate(img_indices)]
    # img_names_list = [class_names[str(idx)][1] for idx in img_indices]

    # return img_names_list, softmax_scores_list
    return softmax_scores_list

def get_threshold_batch(val,to_explain, trueImageInd, percentile_area, x_values = None, step_size = 100, neutral_val = 0):

  if x_values == None:
    step_size_float = 100.0 / step_size
    x_values = np.arange(0, 100, step_size_float)
  else:
    x_values = np.array(x_values)
  percentile_list = 100 - x_values

  deletion_score_list  = deletion_score_batch(to_explain, trueImageInd, val, percentile_list, neutral_val)
  y_values = np.array(deletion_score_list)
  area_under_curve = np.trapz(y_values, x_values)
  print("Area under the curve:", area_under_curve)
  diff_x_values = np.diff(x_values)
  cumulative_area = np.cumsum(diff_x_values  * (y_values[:-1] + y_values[1:]) / 2)

  # Find the x-value where cumulative area is 50% of the total area
  target_area = percentile_area * area_under_curve
  print("Target area:", target_area)
  x_results = x_values[np.where(cumulative_area >= target_area)[0][0] + 1]
  print("x-result:", x_results)
  return x_results

def get_weight_batch(total_val, to_explain, trueImageInd, x_threshold, neutral_val):
  num_baseline = total_val.shape[0]
  weight_list = []
  for i in range(num_baseline):
    val = np.sum(total_val[i], axis = 0)
    weight_list.append(deletion_score_batch(to_explain, trueImageInd, val, [100 - x_threshold],neutral_val)[0])
  return weight_list
def get_point2remove(raw_shap_value,to_explain,trueImageInd, x_values = None, step_size = 100, ratio_score = 0.5, neutral_val = 0):
  if x_values == None:
    step_size_float = 50 / step_size
    x_values = np.arange(0, 30, step_size_float)
  else:
    x_values = np.array(x_values)
  percentile_list = 100 - x_values
  # print(percentile_list)
  deletion_score_list  = deletion_score_batch(to_explain, trueImageInd, raw_shap_value, percentile_list, neutral_val,False)
  # print(deletion_score_list)
  full_score = deletion_score_batch(to_explain, trueImageInd, raw_shap_value, [100], neutral_val)[0]
  for i in range(len(deletion_score_list)):
    if deletion_score_list[i]<=ratio_score*full_score:

      return x_values[i]
  return x_values[-1]

def get_auc_deletion(to_explain, trueImageInd, val, x_values = None, neutral_value = 0, image_show = False):
  if x_values is None:
    x_values = np.arange(0,100)
  raw_deletion_score_list = []
  # for i in range(0,100):
  # raw_deletion_score_list.append(deletion_score_batch(to_explain, trueImageInd, raw_shap_value, [100-i],average_all_corners_broadcasted)[0])
  raw_deletion_score_list = deletion_score_batch(to_explain, trueImageInd, val, x_values, neutral_value,image_show)
  y_values = np.array(raw_deletion_score_list)
  area_under_curve = np.trapz(y_values, x_values)
  return area_under_curve



def get_sorted_indices(val):
    flattened_array = val.flatten()
    sorted_indices = np.argsort(flattened_array)
    return sorted_indices
def get_top_k(val, k, sorted_indices = None):
    """
    Find top k biggest values
    """
    if sorted_indices is None:
      sorted_indices = get_sorted_indices(val)
    top_k_indices = sorted_indices[-k:]
    return top_k_indices

def create_mask_from_indices(val, k, sorted_indices = None):
    """
    Create a mask from the indices of the top k biggest values
    """
    top_k_indices = get_top_k(val, k, sorted_indices)
    mask = np.zeros_like(val, dtype=int)
    top_k_positions = np.unravel_index(top_k_indices, val.shape)
    mask[top_k_positions] = 1

    return mask


# def find_d_alpha(val, neutral_val = 0, target_score = 0.5, epsilon = 0.01, max_iter = 100):
#   #val_shape: 3,224,224
#   #Tính score gốc
#   #Thuật toán sử dụng binary search
#   #Xác định tỉ lệ số điểm cần xóa (bắt đầu từ khoảng 0 đến 100 --> tỉ lệ đầu là 50)
#   #Xóa top điểm cao nhất bằng cách thay giá trị của chúng bằng neutral_val
#   #Tính score sau khi xóa, nếu lớn hơn thì lấy giá trị xóa trong khoảng 0 -50 nếu nhỏ hơn thì lấy 50 - 100...làm tiếp đến khi score chênh lệch trong khoảng epsilon
#   pass
import numpy as np

def find_d_alpha(to_explain,val,trueImageInd = None, target_ratio=0.5, neutral_val=0, epsilon=0.01, max_iter=100):
    """
    Binary search to find the proportion of pixels to remove so that the model score approaches target_score.

    Args:
        val (np.array): Input importance scores (e.g., IG attributions), shape (3,224,224)
        model (function): A function that takes an input image and returns a score
        target_score (float): The desired model score after removal
        neutral_val (int or float): The value to replace removed pixels
        epsilon (float): Convergence threshold
        max_iter (int): Maximum iterations for binary search

    Returns:
        float: The optimal removal percentage
    """
    low, high = 0, 100  # Search range in percentage
    iter_count = 0
    full_score = get_score_from_array(to_explain, trueImageInd)
    print(f"Full score: {full_score}")
    target_score = full_score * target_ratio
    score = full_score
    score_high = full_score
    score_low = 0
    while high - low > 0 and iter_count < max_iter:
        iter_count += 1
        mid = (low + high) / 2  # Current percentage to remove

        val_copy = val.copy()
        threshold = np.percentile(val_copy, mid)
        mask = val >= threshold
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
        background_mask = 1 - mask
        partial_image = to_explain[0]*background_mask + mask*neutral_val
        partial_image = np.expand_dims(partial_image, axis=0).astype(np.float32)

        print("SP: ", sum(partial_image!=0))
        score = get_score_from_array(partial_image, trueImageInd)
        # plt.figure(figsize=(5, 5))
        # plt.imshow(partial_image[0])  # Convert (3,224,224) to (224,224,3) for display
        # plt.title(f"Iteration {iter_count} - Removed: {mid:.2f}% - Score: {score:.4f}")
        # plt.axis("off")
        # plt.show()
        print(f"Score: {score}")
        if  abs(score - target_score) < epsilon:
          break
        elif score > target_score:
            score_high = score
            print(f"High score: {score_high}, low_score: {score_low}")
            print(f"High: {high},Mid: {mid}, Low: {low}")

            high = mid  # Reduce removal percentage

        else:
            score_low = score
            print(f"High score: {score_high}, low_score: {score_low}")
            print(f"High: {high},Mid: {mid}, Low: {low}")
            low = mid   # Increase removal percentage


    return mid, score
def exact_find_d_alpha(to_explain,val,trueImageInd = None, target_ratio=0.5, neutral_val=0, epsilon=0.005, max_iter=100):
  low, high = 0, val.shape[0]*val.shape[1]
  sorted_indices = get_sorted_indices(val)
  full_score = get_score_from_array(to_explain, trueImageInd)
  print(f"Full score: {full_score}")
  target_score = full_score * target_ratio
  iter_count = 0
  while high - low > 0 and iter_count < max_iter:
    iter_count += 1
    mid = int((low + high) / 2 )
    mask = create_mask_from_indices(val, mid, sorted_indices)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
    background_mask = 1 - mask
    partial_image = to_explain[0]*background_mask + mask*neutral_val
    partial_image = np.expand_dims(partial_image, axis=0).astype(np.float32)
    score = get_score_from_array(partial_image, trueImageInd)

    # plt.figure(figsize=(5, 5))
    # plt.imshow(partial_image[0])  # Convert (3,224,224) to (224,224,3) for display
    # plt.title(f"Iteration {iter_count} - Removed: {mid:.2f}% - Score: {score:.4f}")
    # plt.axis("off")
    # plt.show()

    # print(f"Score: {score}, high: {high}, low: {low}, mid: {mid}")
    if  abs(score - target_score) < epsilon:
      break
    elif score > target_score:
      low = mid  # Reduce removal percentage
    else:
      high = mid   # Increase removal percentage
  return mid,score