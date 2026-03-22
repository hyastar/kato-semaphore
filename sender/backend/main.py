import sys
import pickle
import struct
import socket
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

sys.path.append(__file__.rsplit('\\', 1)[0])

from send.modem_utils import text_to_bin, modulate_signal

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModulationRequest(BaseModel):
    text: str
    mod_type: str

def send_via_socket(mod_type: str, signal: np.ndarray, host='127.0.0.1', port=11244):
    """使用 pickle 序列化发送数据"""
    data = {"mod_type": mod_type, "signal": signal.tolist()}
    payload = pickle.dumps(data)
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(struct.pack('>I', len(payload)))
        s.sendall(payload)

@app.post("/modulate_and_send")
async def modulate_and_send(request: ModulationRequest):
    try:
        binary_string = text_to_bin(request.text)
        signal, fs = modulate_signal(binary_string, request.mod_type)
        send_via_socket(request.mod_type, signal)
        
        return {
            "status": "success",
            "message": "信号已发送",
            "input_text": request.text,
            "binary_string": binary_string,
            "modulation_type": request.mod_type,
            "signal_length": len(signal)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11001)
