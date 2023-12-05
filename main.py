from fastapi import FastAPI, Response, UploadFile, File, status, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import cv2
import numpy as np
from novita_client import NovitaClient, Img2ImgRequest, Samplers, ModelType, save_image, ProgressResponseStatusCode
from novita_client.utils import read_image_to_base64, image_to_base64

url = "https://api.novita.ai"
client = NovitaClient("6177bbc7-c152-4751-930b-258d523d513f", url)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def long_taking_task():
    import time
    time.sleep(10)
    return 40

async def img2img_api(image: Image):
    req = Img2ImgRequest(
        model_name="dvarchMultiPrompt_dvarchExterior_28334.safetensors",
        prompt="((white couch)), white carpet, latino style",
        negative_prompt="(brown couch), (brown carpet), bad, ugly, terrible",
        height=512,
        width=768,
        sampler_name=Samplers.DPMPP_S_A_KARRAS,
        cfg_scale=11.5,
        steps=30,
        batch_size=2,
        seed=-1,
        # init_images=[read_image_to_base64("couch.jpg")]
        init_images=[image_to_base64(image)]
    )
    res = client.img2img(req)
    return res.data.task_id
    # save_image(res.data.imgs_bytes[0], "test.png")

async def check_progress(task_id: str):
    res = client.progress(task_id)
    return res

@app.get("/")
def read_root():
    return {"Status": "Success"}


@app.post("/generate")
async def generate(background_tasks: BackgroundTasks, style: str = Form(), prompt: str = Form(), image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    
    task_id = await img2img_api(image)
    background_tasks.add_task(check_progress, task_id)
    # res = long_taking_task()

    # response.status_code = status.HTTP_201_CREATED
    return {"message": "success", "prompt": prompt, "task_id": task_id}

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