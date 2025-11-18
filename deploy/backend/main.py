from fastapi import FastAPI, Request, UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import tempfile
from typing import Optional

import torch
import torch.nn as nn
import torchaudio
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinga2Templates
from pydantic import BaseModel


from import 




try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE=True
except Exception:
    PYDUB_AVAILABLE=False







app=FastAPI()

app.mount("/static",StaticFiles(directory="frontend/static"),name="static")
templates=Jinga2Templates(directory="frontend")

CHECKPOINT_PATH="result/lstm_ctc_checkpoint.pth"
