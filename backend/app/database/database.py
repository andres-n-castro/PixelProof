from sqlalchemy import create_engine
from typing import Annotated
from fastapi import Depends
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
import os

load_dotenv()

#connects to postgres db and starts sqlalchemy session
engine = create_engine(os.getenv("DATABASE_URL"), echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

async def get_db():
  db = SessionLocal()
  try:
    yield db
  finally:
    db.close()

CurrentSession = Annotated[Session, Depends(get_db)]