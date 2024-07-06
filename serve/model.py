import torch.nn as nn
import os
from transformers import Blip2Processor, Blip2Model, Blip2Config
from PIL import Image
import torch
import os

model_dir = '/home/user01/projects/blip-serve/blip2-opt-2.7b'
config = Blip2Config.from_pretrained(model_dir)

gpu_index = 4

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)


class MyBlip2Model(nn.Module):
    def __init__(self):
        super(MyBlip2Model, self).__init__()
        self.inner_model = Blip2Model(config)

    def forward(self, input_ids):
        outputs = self.inner_model(**input_ids)
        return outputs


image_path = '/home/user01/mfsadi/projects/vlp/sample_images/sCQFiYGGC7ILLLN2LZjkG6EytwYaKJ0604vJyRhMytF9FmMe9f.jpg_800X800X70.jpg'
img = Image.open(image_path).convert("RGB")
processor = Blip2Processor.from_pretrained(model_dir)

model = Blip2Model.from_pretrained(model_dir)
model.to('cuda')
model.eval()

image_list = [img] * 5
inp = processor(images=image_list, text=['' for _ in image_list], return_tensors="pt").to('cuda')
res = model(**inp).vision_outputs['pooler_output']

ds = res.cpu().detach().numpy().tolist()
len(ds[2])
