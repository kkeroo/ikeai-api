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
from PIL import Image

url = "https://api.novita.ai"
try:
    api_key = dotenv_values(".env")["API_KEY"]
except Exception as e:
    api_key = None
    raise Exception("API_KEY not found in .env file")

client = NovitaClient(api_key, url)

res = client.img2video(
    model_name="SVD-XT",
    steps=30,
    frames_num=25,
    image=Image.open('download.png')
)

import requests

res = requests.get('https://api.novita.ai/v3/async/task-result?task_id=75707d02-db5e-41b3-9a4d-4b193e88681d', 
                   headers={'Authentication': 'BEARER '+api_key})
print(res)


