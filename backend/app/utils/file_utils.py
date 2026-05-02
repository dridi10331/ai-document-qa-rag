from pathlib import Path
import re

import aiofiles
from fastapi import UploadFile

SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def sanitize_filename(filename: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", filename).strip("._")
    return cleaned or "file"


async def save_upload_file(upload_file: UploadFile, dest_path: Path) -> int:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    async with aiofiles.open(dest_path, "wb") as out_file:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            await out_file.write(chunk)
    await upload_file.close()
    return size
