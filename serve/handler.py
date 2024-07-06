from ts.torch_handler.base_handler import BaseHandler
import torch
from transformers import Blip2Processor, Blip2Model
import base64
from PIL import Image
from io import BytesIO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_dir = './blip2-opt-2.7b'
# model_path = os.path.join(model_dir, 'pytorch_model.bin')
processor = Blip2Processor.from_pretrained(model_dir)


class Handler(BaseHandler):

    def __init__(self):
        super(Handler, self).__init__()
        self.manifest = None
        self.device = None
        self.initialized = None
        self.model = None
        self.model_name = None

    def initialize(self, context):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.model = Blip2Model.from_pretrained(model_dir)
        self.model.to('cuda')
        self.model.eval()
        self.initialized = True

    @staticmethod
    def base64_to_pil(base64_str):
        image_data = base64.b64decode(base64_str)
        image_io = BytesIO(image_data)
        image = Image.open(image_io)
        return image

    @staticmethod
    def list_classes_from_module(module, parent_class=None):
        import inspect

        # Parsing the module to get all defined classes
        classes = [
            cls[1]
            for cls in inspect.getmembers(
                module,
                lambda member: inspect.isclass(member) and member.__module__ == module.__name__,
            )
        ]
        # filter classes that is subclass of parent_class
        if parent_class is not None:
            return [c for c in classes if issubclass(c, parent_class)]

        return classes

    def preprocess(self, image_list):
        inp = processor(images=image_list, text=['' for _ in image_list], return_tensors="pt").to('cuda')
        return inp

    def inference(self, input_ids, **kwargs):
        try:
            result = self.model(**input_ids).vision_outputs['pooler_output']
            print('Inference Done..')
        except Exception as e:
            print('Exception in Inference: ', e)
            return []
        return result

    def postprocess(self, result):
        try:
            print('Postprocess Start..')
            res = dict()
            for ind, r in enumerate(result):
                res[ind] = r.cpu().detach().numpy().tolist()
            print('Postprocess Done..')
            return res
        except Exception as e:
            print('Postprocess Exception:', e)
            return []

    def handle(self, data, context):
        print("Start Request..")
        image_base64_list = data[0]['body']['base64']
        image_list = list()
        for bs64_image in image_base64_list:
            img = self.base64_to_pil(bs64_image)
            image_list.append(img)
        print('Image Size: ', len(image_list))
        input_ids = self.preprocess(image_list)
        print('input length', len(input_ids['pixel_values']))
        inference_output = self.inference(input_ids)
        return [self.postprocess(inference_output)]
