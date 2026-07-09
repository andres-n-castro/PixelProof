from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from dotenv import load_dotenv
from database.models import Base
from database.database import engine
from api.user_router import router as user_router
from api.video_router import router as video_router

load_dotenv()

app = FastAPI()
app.include_router(user_router)
app.include_router(video_router)

Base.metadata.create_all(bind=engine)

origins = [
  "http://localhost:3000"
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)

@app.get("/")
def root():
  return {"status : online"}


