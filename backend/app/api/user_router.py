
from database.schemas import UserCreate, UserRead, UserCredentials, UserUpdate
from database.repositories import user_repository as repo
from database.database import CurrentSession
from pwdlib import PasswordHash
from database.models import User
from fastapi import HTTPException, APIRouter
from auth.auth_handler import sign_jwt
import uuid

router = APIRouter()

password_hash = PasswordHash.recommended()

@router.post("/user/register", status_code=201, response_model=UserRead)
async def register(user: UserCreate, db: CurrentSession):

  #verify if user already exists
  existing_user = repo.get_user_email(db, user.email)

  if existing_user:
    raise HTTPException(status_code=404, detail="user already exists")

  #hash password
  hashed_password = password_hash.hash(user.password)

  #create user object in db user schema
  new_user = User(
    fullname=user.fullname,
    email=user.email,
    password=hashed_password,
    last_login=None,
  )

  #add user to db
  return repo.create_user(db, new_user)

@router.post("/user/login", status_code=200)
async def login(db: CurrentSession, user_credentials: UserCredentials):

  #retrieve user object from database
  db_user = repo.get_user_email(db, user_credentials.email)

  if not db_user:
    raise HTTPException(status_code=401, detail="Incorrect Email or Password")
  
  #validate email and password
  isPasswordValid = password_hash.verify(
    user_credentials.password,
    db_user.password
  )

  if not isPasswordValid:
    raise HTTPException(status_code=401, detail="Incorrect Email or Password")
  
  #create JWT
  token_response = sign_jwt(db_user.id)
  return token_response
  
@router.patch("/user/{user_id}", status_code=200)
async def update(user_id: uuid.UUID, package: UserUpdate, db: CurrentSession ):
  return repo.update_user(db=db, user=package, user_id=user_id)

@router.delete("/user/{user_id}", status_code=204)
async def delete(user_id: uuid.UUID, db: CurrentSession ):
  return repo.delete_user(db, user_id)

@router.get("/user/{user_id}")
async def get_user(user_id: uuid.UUID, db: CurrentSession ):
  return repo.get_user(db, user_id)
