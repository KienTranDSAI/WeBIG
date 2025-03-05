
# import os
# import sys

# __dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
# sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

# import numpy as np
# import matplotlib.pyplot as plt
# from utils import get_partial_score_batch

# def deletion_score_batch(to_explain, trueImageInd, val, percentile_list, neutral_val = 0, image_show = False):


#   batch_partial_images = []
#   for percentile in percentile_list:
#     threshold = np.percentile(val, percentile)
#     mask = val >= threshold
#     mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
#     background_mask = 1 - mask
#     partial_image = to_explain[0]*background_mask + mask*neutral_val
#     partial_image = partial_image.astype(np.float32)

#     batch_partial_images.append(partial_image)
#     if image_show:
#       plt.imshow(partial_image)
#       plt.show()
#   return get_partial_score_batch(batch_partial_images, trueImageInd)


# def get_auc_deletion(to_explain, trueImageInd, val, x_values = None, neutral_value = 0,image_show = False):
#   if x_values is None:
#     x_values = np.arange(0,101)
#   raw_deletion_score_list = []
#   # for i in range(0,100):
#   # raw_deletion_score_list.append(deletion_score_batch(to_explain, trueImageInd, raw_shap_value, [100-i],average_all_corners_broadcasted)[0])
#   raw_deletion_score_list = deletion_score_batch(to_explain, trueImageInd, val, 100 - x_values, neutral_value, image_show)
#   y_values = np.array(raw_deletion_score_list)
#   # print(y_values)
#   # plt.plot(y_values)
#   area_under_curve = np.trapz(y_values, x_values)
#   return area_under_curve


# def insertion_score_batch(to_explain, trueImageInd, val, percentile_list, neutral_val = 0, image_show = False):
#   batch_partial_images = []
#   for percentile in percentile_list:
#     threshold = np.percentile(val, percentile)
#     mask = val >= threshold
#     mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
#     background_mask = 1 - mask
#     partial_image = to_explain[0]*mask + background_mask*neutral_val
#     partial_image = partial_image.astype(np.float32)

#     batch_partial_images.append(partial_image)
#     if image_show:
#       plt.imshow(partial_image)
#       plt.show()
#   return get_partial_score_batch(batch_partial_images, trueImageInd)


# def get_auc_insertion(to_explain, trueImageInd, val, x_values = None, neutral_value = 0,image_show = False):
#   if x_values is None:
#     x_values = np.arange(0,101)
#   raw_insertion_score_list = []
#   raw_insertion_score_list = insertion_score_batch(to_explain, trueImageInd, val, 100 - x_values, neutral_value, image_show)
#   y_values = np.array(raw_insertion_score_list)
#   # print(y_values)
#   # plt.plot(y_values)
#   # print(y_values)x
#   area_under_curve = np.trapz(y_values, x_values)
#   return area_under_curve