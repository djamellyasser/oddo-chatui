from fastapi import APIRouter, Depends

from ..security import authenticate

router = APIRouter()


@router.get("/health", dependencies=[Depends(authenticate)])
async def health():
    return {"status": "ok"}
