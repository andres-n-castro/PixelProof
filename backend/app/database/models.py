import uuid
from sqlalchemy import create_engine, String, ForeignKey, Uuid, Integer
from sqlalchemy.orm import DeclarativeBase
from typing import List, Optional
from sqlalchemy.orm import Mapped, mapped_column, relationship, sessionmaker
from datetime import datetime

class Base(DeclarativeBase):
  pass

class User(Base):
  __tablename__ = "user"

  id : Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
  fullname : Mapped[str] = mapped_column(String(30))
  password : Mapped[Optional[str]] = mapped_column(String(30))
  email : Mapped[str] = mapped_column(String(30))
  created_at : Mapped[datetime] = mapped_column()
  last_login : Mapped[str] = mapped_column(String(30))
  videos : Mapped[List["Video"]] = relationship(back_populates="user")

  def __repr__(self) -> str:
    return f"User(name={self.fullname}, password={self.password}, email={self.email}, videos_list={self.videos_list}, last_login={self.last_login}), created_at={self.created_at}"


class Video(Base):
  __tablename__ = "video"

  id : Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
  video_name : Mapped[str] = mapped_column(String(30)) 
  result : Mapped[int] = mapped_column(Integer)
  user_id : Mapped[int] = mapped_column(ForeignKey("user.id"))
  user : Mapped["User"] = relationship(back_populates="videos")
