import aiohttp
import tempfile
from fastapi import HTTPException



async def download_video(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    content = await response.read()
                    tmp_file.write(content)
                    return tmp_file.name
    raise HTTPException(status_code=400, detail="Video download failed")