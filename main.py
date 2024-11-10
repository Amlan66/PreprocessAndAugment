from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
from text_processing import preprocessing as text_preprocessing
from text_processing import augmentation as text_augmentation
from image_processing import preprocessing as image_preprocessing
from image_processing import augmentation as image_augmentation
from audio_processing import preprocessing as audio_preprocessing
from audio_processing import augmentation as audio_augmentation
import io
import imghdr
import base64

app = FastAPI()

# Set up templates
templates = Jinja2Templates(directory="templates")

stored_data = {
    "original_content": "",
    "file_type": "",
    "processed_content": "",
    "augmented_content": ""
}

def is_image(file_contents):
    """Check if the file is an image."""
    image_type = imghdr.what(None, file_contents)
    return image_type in ['jpeg', 'jpg']

def is_audio(filename):
    """Check if the file is an audio file."""
    return filename.lower().endswith('.mp3')

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    
    if file.filename.endswith('.txt'):
        stored_data["file_type"] = "text"
        stored_data["original_content"] = contents.decode()
    elif is_image(contents):
        stored_data["file_type"] = "image"
        stored_data["original_content"] = f'data:image/jpeg;base64,{base64.b64encode(contents).decode()}'
    elif is_audio(file.filename):
        stored_data["file_type"] = "audio"
        audio_b64 = base64.b64encode(contents).decode()
        stored_data["original_content"] = f'data:audio/mp3;base64,{audio_b64}'
    else:
        return {"error": "Unsupported file type"}
    
    stored_data["processed_content"] = ""
    stored_data["augmented_content"] = ""
    
    return {
        "file_type": stored_data["file_type"],
        "content": stored_data["original_content"]
    }

@app.post("/process")
async def process_content(
    content_type: str = Form(...),
    method: str = Form(...),
    processing_type: str = Form(...)
):
    if content_type == "text":
        if processing_type == "preprocess":
            result = text_preprocessing.preprocess_text(
                stored_data["original_content"],
                method == "tokenize",
                method == "removePunct",
                method == "addPadding"
            )
        else:  # augment
            result = text_augmentation.augment_text(
                stored_data["original_content"],
                method == "synonymReplace",
                method == "randomInsert"
            )
    elif content_type == "image":
        # Extract the base64 image data
        content_bytes = base64.b64decode(stored_data["original_content"].split(',')[1])
        
        if processing_type == "preprocess":
            # Add default parameters for image preprocessing
            params = {
                'size': (224, 224) if method == 'resize' else None
            }
            result = image_preprocessing.preprocess_image(content_bytes, method, params)
        else:  # augment
            result = image_augmentation.augment_image(content_bytes, method)
    elif content_type == "audio":
        content_bytes = base64.b64decode(stored_data["original_content"].split(',')[1])
        if processing_type == "preprocess":
            result = audio_preprocessing.preprocess_audio(content_bytes, method)
        else:  # augment
            result = audio_augmentation.augment_audio(content_bytes, method)
    
    if processing_type == "preprocess":
        stored_data["processed_content"] = result
    else:
        stored_data["augmented_content"] = result
    
    return {"result": result}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        **stored_data
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 