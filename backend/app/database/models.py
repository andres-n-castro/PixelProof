import uuid
from sqlalchemy import String, ForeignKey, Uuid, DateTime
from sqlalchemy import func
from sqlalchemy.orm import DeclarativeBase
from typing import List, Optional
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime

class Base(DeclarativeBase):
  pass

class User(Base):
  __tablename__ = "user"

  id : Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
  fullname : Mapped[str] = mapped_column(String(60))
  password : Mapped[Optional[str]] = mapped_column(String(256))
  email : Mapped[str] = mapped_column(String(254))
  created_at : Mapped[datetime] = mapped_column(server_default=func.now())
  last_login : Mapped[datetime | None] = mapped_column(DateTime)
  videos : Mapped[List["Video"]] = relationship(back_populates="user")

  def __repr__(self) -> str:
    return f"User(name={self.fullname}, password={self.password}, email={self.email}, videos_list={self.videos}, last_login={self.last_login}), created_at={self.created_at}"


class Video(Base):
  __tablename__ = "video"

  id : Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
  video_name : Mapped[str] = mapped_column(String(256)) 
  video_path : Mapped[str] = mapped_column(String(1024))
  prediction : Mapped[str] = mapped_column(String(60))
  status: Mapped[str] = mapped_column(String(60))
  user_id : Mapped[uuid.UUID] = mapped_column(ForeignKey("user.id"))
  user : Mapped["User"] = relationship(back_populates="videos")
