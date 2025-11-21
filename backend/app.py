from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.chatbot_engine import run_chatbot

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class Query(BaseModel):
    message: str
    session_id: str = "1"

@app.post("/chat")
def chat(query: Query):
    response = run_chatbot(query.message, query.session_id)
    return {"response": response}

