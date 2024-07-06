import requests
import base64
from PIL import Image
from io import BytesIO
from tqdm import tqdm

url: str = f'http://127.0.0.1:8080/predictions/blip2'


def pil_to_base64(image, _format='JPEG'):
    image_io = BytesIO()
    image.save(image_io, format=_format)
    image_data = image_io.getvalue()
    base64_str = base64.b64encode(image_data).decode('utf-8')
    return base64_str


image_path = '70.jpg'
img = Image.open(image_path).convert("RGB")
base64_str = pil_to_base64(img)
data = {'base64': [base64_str] * 5}
batch = 10
for i in tqdm(range(batch)):
    res = requests.post(url=url, json=data)
    es = res.json()
