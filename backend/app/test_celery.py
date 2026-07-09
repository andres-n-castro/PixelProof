from celery_app import celery_app

if __name__ == "__main__":
  print("registered tasks:")
  for task_name in sorted(celery_app.tasks.keys()):
    print(task_name)