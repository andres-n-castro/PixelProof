from celery import Celery
import os

celery_app = Celery(
  'celery_app',
  broker=os.getenv("REDIS_URL"),
  backend=os.getenv("RESULT_BACKEND"),
  include=['celery_tasks']
  )
