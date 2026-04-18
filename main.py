from backend.api import app

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://v0-nirvesh-ai-dashboard.vercel.app", "http://localhost:3000","https://nirveshai-financialadvisor.streamlit.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
