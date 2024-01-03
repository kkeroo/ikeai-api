# IkeAI API

## Instalation
You need python and pip. It is also recommended that you use virtual environment. Do this with:\
```python3 -m venv venv```\
```source venv/bin/activate```\
\
and then install all packages with:\
```pip install -r requirements.txt```\
\
You will also need to download SAM model weights [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)\
and create .env file where you paste API_KEY="NOVITA API KEY"

## Running
To run the server:\
```uvicorn main:app --reload```\
The server is listening on port 8000. The docs are available at ```/docs```
 
