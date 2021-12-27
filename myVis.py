from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torch
from PIL import Image
import cv2
import numpy as np

pthfile = r'E:/paper_exm/paper/106MoreResult/final_state.pth'
img='E:/paper_exm/subwayData/sbDataset106/sbDataset106More/images/val/abnormal_tst_img_908.jpg'
model = torch.load(pthfile)
print(model)
target_layers = [model.layer4[-1]]

img_cv = cv2.imread(img)
input_tensor = torch.from_numpy(np.transpose(img_cv, (2, 0, 1)))# Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# If target_category is None, the highest scoring category
# will be used for every image in the batch.
# target_category can also be an integer, or a list of different integers
# for every image in the batch.
target_category = 1

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(img_cv, grayscale_cam, use_rgb=True)
