from fastapi import Depends
from auth.auth_handler import decode_jwt
from auth.auth_bearer import JWTBearer
from typing import Annotated


async def get_current_user(token: Annotated[str, Depends(JWTBearer)]):
  user_payload = decode_jwt(token)
  return user_payload
  
#reference for authentication handling
CurrentUser = Annotated[dict, Depends(get_current_user)]