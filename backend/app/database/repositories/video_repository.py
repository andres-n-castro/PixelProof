import uuid
import sqlalchemy as sa
from sqlalchemy.orm import Session
from database.schemas import VideoUpdate
from database.models import Video

def create_video(db : Session, video : Video) -> Video:
  db.add(video)
  db.commit()
  db.refresh(video)
  return video

def update_video(db : Session, video : VideoUpdate, video_id: uuid.UUID) -> Video:

  video_data = video.model_dump(exclude_unset=True)

  stmt = (
    sa.update(Video)
    .where(Video.id == video_id)
    .values(**video_data)
    .returning(Video)
  )
  
  db_obj = db.scalars(stmt).first()
  db.commit()

  return db_obj


def delete_video(db : Session, video_id : uuid.UUID) -> bool:

  stmt = (
    sa.delete(Video)
    .where(Video.id == video_id)
    .returning(Video)
  )

  result = db.execute(stmt)

  if result.first() != None:
    db.commit()
    return True
  else:
    return False

def get_video(db : Session, video_id : uuid.UUID) -> Video:

  stmt = (
    sa.select(Video)
    .where(Video.id == video_id)
  )

  db_obj = db.scalars(stmt).first()
  return db_obj