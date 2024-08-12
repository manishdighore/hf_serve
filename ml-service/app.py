from fastapi import FastAPI, APIRouter, UploadFile, File, Form, Query
import uvicorn
from inference import VisionInference,SpeechInference
import logging
import os
from utils import extract_json_from_string

logging.basicConfig(level=logging.INFO)

app = FastAPI()
router = APIRouter()

vision_inference = VisionInference()
#speech_inference = SpeechInference()

@router.get("/")
async def home():
  return {"message": "Deep Learning services"}


@router.post("/vision")
async def process_vision(
    image: UploadFile = File(...),
    instruction: str = Form(...),
    json_output: bool = Query(False)
):
    # Save the image file to a temporary directory
    file_path = f"tmp/{image.filename}"
    with open(file_path, "wb") as f:
        f.write(await image.read())
    
    # Process the image based on the instruction
    response = vision_inference.predict(file_path, instruction)

    # Remove the temporary image file
    os.remove(file_path)

    if json_output:
        return extract_json_from_string(response)
    else:
        return {"response": response}
    
@router.post("/speech")
async def process_speech(
    audio: UploadFile = File(...)
):
    # Save the audio file to a temporary directory
    file_path = f"tmp/{audio.filename}"
    with open(file_path, "wb") as f:
        f.write(await audio.read())
    
    # Process the audio file
    response = speech_inference.predict(file_path)

    # Remove the temporary audio file
    os.remove(file_path)

    return {"response": response}


# @router.post("/sql")
# async def process_prmpt(
#     prompt: str = Form(...)
# ):
    
#     response = SQLInference.predict(prompt)

#     return {"response": response}
    
app.include_router(router)

if __name__ == "__main__":
  uvicorn.run("app:app", reload=False, port=5000, host="0.0.0.0")