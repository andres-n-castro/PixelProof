import uuid
import sqlalchemy as sa
from sqlalchemy.orm import Session
from database.schemas import UserUpdate
from database.models import User
from typing import Any

def create_user(db : Session, user : User) -> User:
  db.add(user)
  db.commit()
  db.refresh(user)
  return user

def update_user(db: Session, user: UserUpdate, user_id: uuid.UUID) -> Any | None:
  update_user = user.model_dump(exclude_unset=True)

  stmt = (
    sa.update(User)
    .where(User.id == user_id)
    .values(**update_user)
    .returning(User)
  )

  updated_obj = db.scalars(stmt).first()
  db.commit()

  return updated_obj

def delete_user(db : Session, user_id : uuid.UUID) -> bool:

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

def get_user(db : Session, user_id : uuid.UUID) -> Any | None:
  stmt = (
    sa.select(User)
    .where(User.id == user_id)
  )

  db_obj = db.scalars(stmt).first()
  return db_obj


def get_user_email(db: Session, user_email: str) -> User | None:

  stmt = (
    sa.select(User)
    .where(User.email == user_email)
  )

  db_obj = db.scalars(stmt).first()

  return db_obj