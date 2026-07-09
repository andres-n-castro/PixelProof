from database.database import CurrentSession
from database.repositories import video_repository as repo
from database.schemas import VideoUpdate
from database.models import Video
from fastapi import HTTPException, UploadFile, APIRouter
from auth.auth_get_user import CurrentUser
from celery_tasks import run_model
from pathlib import Path
import shutil
import uuid

router = APIRouter()

#validate file and authentication
@router.post("/user/{user_id}/videos", status_code=202)
async def process(user_id: uuid.UUID, video: UploadFile,  user_payload: CurrentUser, db: CurrentSession):
  
  if Path(video.filename).suffix.lower() != ".mp4":
    raise HTTPException(status_code=400, detail="incorrect file type")
  
  if user_payload["user_id"] != str(user_id):
    raise HTTPException(status_code=403, detail="forbidden")
  
  video_payload = Video(
    video_name=video.filename,
    prediction="",
    status="processing",
    video_path="",
    user_id=user_id
  )

  video_row = repo.create_video(db=db, video=video_payload)

  uploads_dir = Path("uploads")
  uploads_dir.mkdir(parents=True, exist_ok=True)
  video_path = str(uploads_dir / f"{video_row.id}.mp4")

  with open(video_path, "wb") as file:
    shutil.copyfileobj(video.file, file)
  
  _ = repo.update_video(db=db, video=VideoUpdate(video_path=video_path), video_id=video_row.id)

  run_model.delay(video_id=video_row.id)

  return (f"New Video Sample created! Status: {video_row.status}", video_row.id)

@router.get("/videos/{video_id}/status", status_code=200)
async def progress(video_id: uuid.UUID, db: CurrentSession):
  try:
    video_obj = repo.get_video(db=db, video_id=video_id)
  except:
    raise HTTPException(status_code=404, detail="item not found")
  
  return video_obj.status

@router.get("/videos/{video_id}", status_code=200)
async def get_video(video_id: uuid.UUID, db: CurrentSession):
  try:
    video_obj = repo.get_video(db=db, video_id=video_id)
  except:
    raise HTTPException(status_code=404, detail="item not found")
  
  return video_obj.prediction