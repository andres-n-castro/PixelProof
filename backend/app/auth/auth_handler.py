from datetime import datetime, timezone, timedelta
from typing import Dict
import jwt
from decouple import config
from fastapi import HTTPException


JWT_SECRET = config("secret")
JWT_ALGORITHM = config("algorithm")

#returns a dict of where the key is str "access_token" and key is user's token
def token_response(token: str) -> dict:
  return {
    "access_token": token
  }

#function that creates the the jwt token using user id and specific experiation time
def sign_jwt(user_id: str) -> Dict[str, str]:
  payload ={
    "user_id": user_id,
    "exp": datetime.now(timezone.utc) + timedelta(hours=1)
  }

  token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

  return token_response(token)


#function that decodes the jwt token string into 
def decode_jwt(token: str) -> dict:
  try:
    decoded_token = jwt.decode(
      token,
      JWT_SECRET,
      algorithms=JWT_ALGORITHM
    )
    return decoded_token
  except:
    raise HTTPException(status_code=401)

