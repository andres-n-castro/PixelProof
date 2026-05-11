import uuid
import sqlalchemy as sa
from sqlalchemy.orm import Session
from app.database.schemas import UserCreate, UserRead
from app.database.models import User

def create_user(db : Session, user : UserCreate):
  user_data = user.model_dump()
  new_user = User(**user_data)

  db.add(new_user)
  db.commit()
  db.refresh(new_user)
  return new_user

def update_user(db : Session, user : UserRead):

  user_data = user.model_dump()

  stmt = (
    sa.update(User)
    .wehere(User.id == user.id)
    .values(**user_data)
    .returning(User)
  )

  updated_obj = db.scalars(stmt).first()
  db.commit()

  return updated_obj

def delete_user(db : Session, user_id : uuid.UUID):

  stmt = (
    sa.delete(User)
    .where(User.id == user_id)
    .returning(User)
  )

  result = db.execute(stmt)
  if result.first != None:
    db.commit()
    return True
  else:
    return False

def get_user(db : Session, user_id : uuid.UUID):
  stmt = (
    sa.select(User)
    .where(User.id == user_id)
  )

  db_obj = db.scalars(stmt).first()
  return db_obj