import faulthandler
faulthandler.enable()

import os
import sys
import socket
import threading
import time
import urllib.request
import urllib.error
import uvicorn

# 1. SILENCE EVERYTHING: Redirect all standard output and errors to a log file
os.makedirs("logs", exist_ok=True)
log_fd = open("logs/tanglawan.log", "a", encoding="utf-8")
sys.stdout = log_fd
sys.stderr = log_fd

# 2. HELPER FUNCTION TO PRINT ONLY TO THE TERMINAL SCREEN
def console_print(text):
    sys.__stdout__.write(text + "\n")
    sys.__stdout__.flush()

def get_network_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

HOST_IP = get_network_ip()
LOCAL_URL = "http://127.0.0.1:8000"
NETWORK_URL = f"http://{HOST_IP}:8000"

def _wait_for_server():
    # Inform you so it doesn't look frozen!
    console_print("\n⏳ Booting AI Engine... (This takes 15-30 seconds, please wait)")
    
    deadline = time.time() + 120.0
    while time.time() < deadline:
        try:
            urllib.request.urlopen(LOCAL_URL, timeout=1)
            break
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    else:
        console_print("\n" + "=" * 60)
        console_print("⚠️ TANGLAWAN SERVER STARTED (Engine load timed out)")
        console_print(f"👉 CLICK HERE TO OPEN: {NETWORK_URL}")
        console_print("=" * 60 + "\n")
        return

    console_print("\n" + "=" * 60)
    console_print("🚀 TANGLAWAN AI IS LIVE — AI ENGINE FULLY LOADED")
    console_print(f"👉 CLICK HERE TO OPEN: {NETWORK_URL}")
    console_print("=" * 60 + "\n")

if __name__ == "__main__":
    threading.Thread(target=_wait_for_server, daemon=True).start()

    # Uvicorn and FastAPI prints will be swallowed by the log file
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        access_log=False,
        use_colors=False,
    )