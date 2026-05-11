from fastapi import FastAPI, HTTPException
from .database.models import Base, sessionmaker, create_engine

'''
#connects to postgres db and starts sqlalchemy session
engine = create_engine("url for database server", echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
  db = SessionLocal()
  try:
    yield db
  finally:
    db.close()
'''

app = FastAPI()
#create and connect the user and video endpoints

@app.get("/")
def root():
  return {"status : online"}


@app.get("/items")
def get_item(item_id : int):
  return f"returning item {item_id}"