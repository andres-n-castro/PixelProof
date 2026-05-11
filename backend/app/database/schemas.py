import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List

class VideoCreate(BaseModel):
  video_name : str
  result : int

class VideoRead(BaseModel):
  id : uuid.UUID
  video_name : str
  result : int
  created_at : datetime
  user_id : int

class UserCreate(BaseModel):
  fullname : str
  email : str
  hashed_password : str
  last_login : datetime

class UserRead(BaseModel):
  id : uuid.UUID
  fullname : str
  email : str
  hashed_password : str
  video_list : Optional[List[VideoRead]]
  created_at : datetime
  last_login : datetime