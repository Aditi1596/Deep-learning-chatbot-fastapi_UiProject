from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.preprocessing.text import tokenizer_from_json
#import json
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
# -----------------------------------------
# FASTAPI APP
# -----------------------------------------
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -----------------------------------------
# LOAD MODEL + TOKENIZER
# -----------------------------------------


# Load tokenizer as JSON STRING
with open("model/tokenizer.json", "r") as f:
    tokenizer_json = f.read()   # <-- read as string

tokenizer = tokenizer_from_json(tokenizer_json)


max_len = 10

# -----------------------------------------
# User input model
# -----------------------------------------
class ChatRequest(BaseModel):
    message: str

# -----------------------------------------
# Homepage route
# -----------------------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -----------------------------------------
# Chat API
# -----------------------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    text = req.message

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)

    pred = model.predict(padded)
    word_index = np.argmax(pred)

    # Reverse lookup word
    for word, index in tokenizer.word_index.items():
        if index == word_index:
            answer = word
            break
    else:
        answer = "I did not understand that."

    return {"reply": answer}

