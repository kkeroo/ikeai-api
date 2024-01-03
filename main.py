from fastapi import FastAPI, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import cv2
import numpy as np
from dotenv import dotenv_values
from novita_client import NovitaClient, Img2ImgRequest, Samplers, ModelType, save_image, ProgressResponseStatusCode, ReplaceObjectRequest
from novita_client.utils import read_image_to_base64, image_to_base64
from style_prompts import STYLE_PROMPTS, NEGATIVE_PROMPT
import json
from segment_anything import sam_model_registry, SamPredictor
import base64

url = "https://api.novita.ai"
try:
    api_key = dotenv_values(".env")["API_KEY"]
except Exception as e:
    api_key = None
    raise Exception("API_KEY not found in .env file")

client = NovitaClient(api_key, url)

MODEL_TYPE = 'vit_b'
CHECKPOINT_PATH = 'sam_vit_b.pth'
DEVICE = 'cpu'
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def replace_api(image, xmin, ymin, xmax, ymax, prompt):
    mask_predictor.set_image(image)
    masks, scores, logits = mask_predictor.predict(
        box=np.array([xmin, ymin, xmax, ymax]),
        multimask_output=True
    )
    mask = masks[0].astype('uint8')
    kernel_size = 15
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    mask = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    alpha = 0.5
    result = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)

    # Encode into base64
    _, mask = cv2.imencode('.png', mask)
    _, image = cv2.imencode('.png', image)
    mask = base64.b64encode(mask).decode("utf-8")
    image = base64.b64encode(image).decode("utf-8")

    req = Img2ImgRequest(
        prompt="(((" + prompt + ")))",
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
    return task_id

async def img2img_api(image: Image, positive_prompt: str, negative_prompt: str):
    req = Img2ImgRequest(
        model_name="dvarchMultiPrompt_dvarchExterior_28334.safetensors",
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=768,
        sampler_name=Samplers.DPMPP_S_A_KARRAS,
        cfg_scale=9.5,
        steps=30,
        batch_size=2,
        seed=-1,
        # init_images=[read_image_to_base64("couch.jpg")]
        init_images=[image_to_base64(image)]
    )

    res = client.img2img(req)
    print(res)
    return res.data.task_id
    # save_image(res.data.imgs_bytes[0], "test.png")

async def check_progress(task_id: str):
    res = client.progress(task_id)
    return res

@app.get("/")
def read_root():
    return {"Status": "Success"}


@app.post("/design")
async def design(background_tasks: BackgroundTasks, style: str = Form(), positive_prompt: str = Form(), negative_prompt: str = Form(), image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    
    style_prompt = STYLE_PROMPTS[style]
    prompt = f"((interior design)), {style_prompt}, {positive_prompt}"
    negative_prompt = f"{NEGATIVE_PROMPT}, {negative_prompt}"
    
    task_id = await img2img_api(image, prompt, negative_prompt)
    background_tasks.add_task(check_progress, task_id)

    return {"message": "success", "prompt": prompt, "task_id": task_id}

@app.post('/replace')
async def replace(background_tasks: BackgroundTasks, marker: str = Form(...), canvasWidth: str = Form(), canvasHeight: str = Form(), prompt: str = Form(), image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    marker = json.loads(marker)
    image_width = image.shape[1]
    image_height = image.shape[0]
    w_ratio = image_width / int(canvasWidth)
    h_ratio = image_height / int(canvasHeight)
    xmin = int(float(marker['left']) * w_ratio)
    ymin = int(float(marker['top']) * h_ratio)
    xmax = int((float(marker['left']) + float(marker['width'])) * w_ratio)
    ymax = int((float(marker['top']) + float(marker['height'])) * h_ratio)

    task_id = await replace_api(image, xmin, ymin, xmax, ymax, prompt)
    background_tasks.add_task(check_progress, task_id)

    return {"message": "success", "task_id": task_id}

@app.get("/progress/{task_id}")
async def progress(task_id: str):
    task_status = await check_progress(task_id)
    status = "Unknown"
    images = []
    if task_status.data.status == ProgressResponseStatusCode.RUNNING:
        status = "Running"
    elif task_status.data.status == ProgressResponseStatusCode.FAILED:
        status = "Failed"
    elif task_status.data.status == ProgressResponseStatusCode.TIMEOUT:
        status = "Timeout"
    elif task_status.data.status == ProgressResponseStatusCode.SUCCESSFUL:
        status = "Completed"
        images = task_status.data.imgs

    return {"code": task_status.code, "images": images, "task_status": status}