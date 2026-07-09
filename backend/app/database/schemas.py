import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict
from typing import Optional, List

class VideoCreate(BaseModel):
  video_name: str
  video_path: str | None = None
  prediction: str | None = None
  status: str
  user_id: uuid.UUID

  model_config = ConfigDict(from_attributes=True)

class VideoRead(BaseModel):
  id: uuid.UUID
  video_name: str
  video_path: str | None = None
  prediction: str
  status: str
  created_at: datetime
  user_id: uuid.UUID

  model_config = ConfigDict(from_attributes=True)

class VideoUpdate(BaseModel):
  video_name: str | None = None
  video_path: str | None = None
  prediction: str | None = None
  status: str | None = None
  created_at: datetime | None = None
  user_id: uuid.UUID | None = None

  model_config = ConfigDict(from_attributes=True)

class UserCreate(BaseModel):
  fullname: str
  email: str
  password: str

  model_config = ConfigDict(from_attributes=True)

class UserRead(BaseModel):
  id: uuid.UUID
  fullname: str
  email: str
  videos: Optional[List[VideoRead]]
  last_login: Optional[datetime] = None

  model_config = ConfigDict(from_attributes=True)

class UserCredentials(BaseModel):
  email: str
  password: str

  model_config = ConfigDict(from_attributes=True)

class UserUpdate(BaseModel):
  fullname: str | None = None
  email: str | None = None
  password: str | None = None

  model_config = ConfigDict(from_attributes=True)