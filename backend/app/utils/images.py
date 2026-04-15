
import base64
from typing import List
from fastapi import UploadFile

async def files_to_base64(files: List[UploadFile]) -> list[str]:
    out = []
    for f in files or []:
        content = await f.read()
        out.append(base64.b64encode(content).decode('utf-8'))
    return out
