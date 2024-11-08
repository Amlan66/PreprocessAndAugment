from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from text_processing import preprocessing as text_preprocessing
from text_processing import augmentation as text_augmentation
from image_processing import preprocessing as image_preprocessing
from image_processing import augmentation as image_augmentation
import io
import imghdr
import base64

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="templates"), name="static")

stored_data = {
    "original_content": "",
    "file_type": "",
    "processed_content": "",
    "augmented_content": ""
}

def is_image(file_contents):
    """Check if the file is an image by looking at its contents."""
    image_type = imghdr.what(None, file_contents)
    return image_type in ['jpeg', 'jpg']

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Determine file type
    if file.filename.endswith('.txt'):
        stored_data["file_type"] = "text"
        stored_data["original_content"] = contents.decode()
    elif is_image(contents):
        stored_data["file_type"] = "image"
        # Convert image to base64 for display
        stored_data["original_content"] = f'data:image/jpeg;base64,{base64.b64encode(contents).decode()}'
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
    processing_type: str = Form(...)  # 'preprocess' or 'augment'
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
    else:  # image
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