# https://huggingface.co/datasets/somosnlp-hackathon-2022/spanish-to-quechua

from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

model_name = 'hackathon-pln-es/t5-small-finetuned-spanish-to-quechua'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    text = request.json['text']
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return jsonify({'translated_text': translated_text})

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(debug=False, host='0,0,0,0')

"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import uvicorn

app = FastAPI()

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configurar templates
templates = Jinja2Templates(directory="templates")

# Cargar el modelo y el tokenizador
model_name = 'hackathon-pln-es/t5-small-finetuned-spanish-to-quechua'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/translate")
async def translate(request: Request):
    data = await request.json()
    text = data['text']
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return {"translated_text": translated_text}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
