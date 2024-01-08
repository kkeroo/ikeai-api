# IkeAI - Personalize, Visualize, Transform: Your Space, Reimagined 🚀
*Backend API*

Experience the future of interior design with our revolutionary AI Interior Designer tool. Unleash your creativity as our generative models transform your ideas into stunning, personalized living spaces. Design a completly new space or replace the furniture with few-click action.

This project was developed during the course **Interaction and Information Design** at **[FRI](https://www.fri.uni-lj.si/sl)**.

### Project overview
API uses Stable Diffusion to generate stunning images. Stable Diffusion model is not run on your machine instead it is outsourced to the [Novita.ai API](https://novita.ai). In order to run the backend service you need Novita.ai API KEY. Please paste the API KEY in the .env file. Sample env file (called *.sample_env*) is present in the repository. Copy this file and rename it to .env. To generate even better results *[Segment-Anything Model](https://segment-anything.com)* is added to segment any object in the area. You will need to download pre-trained SAM model weights (more on that in Installation section)

### Instalation
Since the frontend is built using FastAPI you need python and pip installed on your machine. PyTorch and Torchvision packages are included in *requirements.txt* file.
1. Clone the repository
2. Move to the project's root folder
3. (*OPTIONAL*) Create virtual environment `python3 -m venv venv`
4. (*OPTIONAL*) Activate virtual environment `source venv/bin/activate`
5. Install required packages `pip install -r requirements.txt`

You will also need to download SAM model weights [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and save it in the project's root folder.

### Starting the server
To start the Backend API you simply run: `uvicorn main:app --reload`. The server is listening on port 8000 by default. The API documentation is located on `/docs`.
