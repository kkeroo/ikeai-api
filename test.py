import requests
from PIL import Image 
import base64
# mask files path
filename_input = "test_images/mask.png"

base64_pic = None
image = None
# read mask file
with open(filename_input, "rb") as f:
    base64_pic = base64.b64encode(f.read()).decode("utf-8")

with open('test_images/man.png', "rb") as f:
    image = base64.b64encode(f.read()).decode("utf-8")

# url = 'https://prodapi.phot.ai/external/api/v2/user_activity/object-replacer'
# headers = {
#     'x-api-key': '658c0d2f3d4676690205a067_b3df3bb7eab8bc8dac21_apyhitools',
#     'Content-Type': 'application/json'
# }
# data = {
#     'file_name': 'YourInputFileName',  # Replace with the actual input file name as a string
#     'input_image_link': 'https://github.com/runwayml/stable-diffusion/blob/main/data/inpainting_examples/8399166846_f6fb4e4b8e_k.png?raw=true',  # Replace with the URL of your input image
#     'mask_image': base64_pic,  # Replace with the Base64-encoded data of your mask image
#     'prompt': 'YourInputPrompt'  # Replace with your specific input prompt
# }
 
# response = requests.post(url, headers=headers, json=data)
 
# if response.status_code == 200:
#     print(response.json())
# else:
#     print(f"Error: {response.status_code} - {response.text}")
    
from novita_client import NovitaClient, Img2ImgRequest, Samplers, ModelType, save_image, ProgressResponseStatusCode, ReplaceObjectRequest
from novita_client.utils import read_image_to_base64, image_to_base64
from dotenv import dotenv_values

url = "https://api.novita.ai"
try:
    api_key = dotenv_values(".env")["API_KEY"]
except Exception as e:
    api_key = None
    raise Exception("API_KEY not found in .env file")

client = NovitaClient(api_key, url)

mask = None 
with open('test_images/living-room-mask.png', "rb") as f:
    mask = base64.b64encode(f.read()).decode("utf-8")

image = None 
with open('test_images/living-room.png', "rb") as f:
    image = base64.b64encode(f.read()).decode("utf-8")



req = Img2ImgRequest(
    prompt="(((brown leather chair)))",
    negative_prompt='canvas frame, cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), wierd colors, blurry,  (((duplicate))), ((morbid)), ((mutilated)), [out of frame], (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), (((disfigured))), out of frame, ugly, (bad anatomy), gross proportions, ugly, tiling, out of frame, mutation, mutated, extra limbs, disfigured, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render',
    model_name='dreamshaper_331-inpainting_11232.safetensors',
    sampler_name='Euler a',
    init_images=[image],
    mask=mask,
    denoising_strength=0.9,
    cfg_scale=13,
    mask_blur=16,
    inpainting_fill=1,
    inpaint_full_res=1,
    inpaint_full_res_padding=32,
    inpainting_mask_invert=0,
    initial_noise_multiplier=1,
    width=768,
    height=512,
    batch_size=1,
    steps=30
)

res = client.img2img(req)

print(res)
task_id = res.data.task_id

import time

while(True):
    res = client.progress(task_id)
    print(res)
    time.sleep(1)
