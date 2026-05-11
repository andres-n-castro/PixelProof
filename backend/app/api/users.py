from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException
from app.main import get_db, app
from app.database.schemas import UserCreate, UserRead
import app.database.repositories as repo

@app.post("/users/{user_id}")
async def create_user(user : UserCreate, db : Session = Depends(get_db())):
  return await repo.create_user(db, user)
  
@app.patch("/users/")
async def update_user(package : UserRead, db : Session = Depends(get_db), ):
  return await repo.update_user(db, package)


@app.delete("/users/{user_id}")
async def delete_user(user_id : int, db : Session = Depends(get_db), ):
  return await repo.delete_user(db, user_id)

@app.get("/users/{user_id}")
async def get_user(user_id: int, db : Session = Depends(get_db), ):
  return await repo.get_user(db, user_id)