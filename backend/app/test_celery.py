from celery_tasks import add

result = add.delay(2, 3)

print("task submitted")
print(f"task id: {result.id}")
print(f"result: {result.get(timeout=10)}")