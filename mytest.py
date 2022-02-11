
from mmdet.apis import init_detector, inference_detector
from mmdet.apis import show_result_pyplot
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'

imagepath = './shishi/'
savepath = './output1/'
config_file = './logs_nd/deformable_detr_1213.py'
checkpoint_file = './logs_nd/epoch_100.pth'

model = init_detector(config_file, checkpoint_file)


for filename in os.listdir(imagepath):
    img = os.path.join(imagepath, filename)
    result = inference_detector(model, img)
    out_file = os.path.join(savepath, filename)
    show_result_pyplot(model, img, result, out_file,score_thr=3e-2)
