from database.database import CurrentSession
from database.repositories import video_repository as repo
from database.schemas import VideoCreate, VideoUpdate, VideoRead
from fastapi import HTTPException, UploadFile, APIRouter
from auth.auth_get_user import CurrentUser
from celery_tasks import run_model
from pathlib import Path
import shutil

router = APIRouter()

#validate file and authentication
@router.post("/videos", status_code=202)
async def process(video: UploadFile,  user_payload: CurrentUser, db: CurrentSession):
  
  if Path(video.filename).suffix.lower() != ".mp4":
    raise HTTPException(status_code=400, detail="incorrect file type")
  
  video_payload = VideoCreate(
    video_name=video.filename,
    status="processing",
    user_id=user_payload["user_id"]
  )

  video_row = repo.create_video(db=db, video=video_payload)

  #save video somewhere local then pass the path to the celery task
  video_path = f"uploads/{video_row.id}.mp4"
  with open(video_path, "wb") as file:
    shutil.copyfileobj(video.file, file)
  
  _ = repo.update_video(db=db, video=VideoUpdate(video_path=video_path))

  run_model.delay(video_id=video_row.id)

  return (f"New Video Sample created! Status: {video_row.status}", video_row.id)

@router.get("/videos/{video_id}/status")
async def progress(video_id: int):
  try:
    video_obj = repo.get_video(db=CurrentSession, video_id=video_id)
  except:
    raise HTTPException(status_code=404, detail="item not found")
  
  return video_obj.status

@router.get("/videos/{video_id}")
async def get_video(video_id: int):
  try:
    video_obj = repo.get_video(db=CurrentSession, video_id=video_id)
  except:
    raise HTTPException(status_code=404, detail="item not found")
  
  return video_obj.prediction