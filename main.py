from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from preprocessing import preprocess_text
from augmentation import augment_text
import io

app = FastAPI()

# Mount templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="templates"), name="static")

# Store uploaded data in memory
stored_data = {"original_text": "", "processed_text": "", "augmented_text": ""}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, **stored_data})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode()
    stored_data["original_text"] = text
    stored_data["processed_text"] = ""
    stored_data["augmented_text"] = ""
    return {"text": text}

@app.post("/preprocess")
async def preprocess(
    text: str = Form(...),
    tokenize: bool = Form(False),
    remove_punct: bool = Form(False),
    add_padding: bool = Form(False)
):
    processed = preprocess_text(text, tokenize, remove_punct, add_padding)
    stored_data["processed_text"] = processed
    return {"processed_text": processed}

@app.post("/augment")
async def augment(
    text: str = Form(...),
    synonym_replace: bool = Form(False),
    random_insert: bool = Form(False)
):
    augmented = augment_text(text, synonym_replace, random_insert)
    stored_data["augmented_text"] = augmented
    return {"augmented_text": augmented}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 