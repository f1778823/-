from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

import os

cert_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "certificate.crt")
key_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "private.key")

if os.path.isfile(cert_file) and os.path.isfile(key_file):
    app.certfile = cert_file
    app.keyfile = key_file



app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(contents)

    # 加載音頻並計算其 Mel 頻譜
    y, sr = librosa.load(file_path, sr=None)
    if y is not None and len(y) > 0:
        print("音頻數據讀取成功，數據長度:", len(y))
    else:
        print("音頻數據為空或未讀取成功")
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    if S_DB is not None and np.any(S_DB):
        print("頻譜生成成功")
    else:
        print("頻譜生成失敗或為空")
    # 使用 Matplotlib 繪製 Mel 頻譜
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig("spectrogram.png")
    plt.close()

    # 刪除音頻文件以清理空間
    os.remove(file_path)

    # 返回生成的頻譜圖像文件
    return FileResponse("spectrogram.png", media_type="image/png")