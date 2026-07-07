
from app.main import app
from app.database.schemas import UserCreate, UserRead, UserCredentials, VideoCreate, VideoRead
import app.database.repositories as repo
from main import CurrentSession
from pwdlib import PasswordHash
from database.models import User
from fastapi import HTTPException, BackgroundTasks, UploadFile
from auth.auth_handler import sign_jwt

password_hash = PasswordHash.recommended()

@app.post("/users/register", status_code=201)
async def register(user: UserCreate, db: CurrentSession):

  #verify if user already exists
  existing_user = repo.get_user_email(db, user.email)

  if existing_user:
    raise HTTPException(detail="user already exists")

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
  return await repo.create_user(db, new_user)

@app.post("users/login", status_code=200)
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
  jwt = token_response["access_token"]
  return jwt
  
@app.patch("/users")
async def update(package : UserRead, db: CurrentSession ):
  return await repo.update_user(db, package)

@app.delete("/users/{user_id}")
async def delete(user_id : int, db: CurrentSession ):
  return await repo.delete_user(db, user_id)

#convert this into a dependency function
@app.get("/users/{user_id}")
async def get_user(user_id: int, db : CurrentSession ):
  return await repo.get_user(db, user_id)
