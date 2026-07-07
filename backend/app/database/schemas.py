import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List

class VideoCreate(BaseModel):
  video_name: str
  prediction: str | None = None
  status: str
  user_id: int
  path: str | None = None

class VideoRead(BaseModel):
  id: uuid.UUID
  video_name: str
  prediction: str
  status: str
  created_at: datetime
  user_id: int
  path: str | None = None

class VideoUpdate(BaseModel):
  id: uuid.UUID
  video_name: str | None = None
  prediction: str | None = None
  status: str | None = None
  created_at: datetime | None = None
  user_id: int | None = None
  path: str | None = None

class UserCreate(BaseModel):
  fullname: str
  email: str
  password: str

class UserRead(BaseModel):
  id: uuid.UUID
  fullname: str
  email: str
  hashed_password: str
  video_list: Optional[List[VideoRead]]
  created_at: datetime
  last_login: datetime

class UserCredentials(BaseModel):
  id: uuid.UUID
  email: str
  password: str