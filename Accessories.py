import time
import psutil
import Globals as gb
import torch

_start = time.perf_counter()
_lastFrameTimeSeconds = 0

def getTimeSeconds():
    return time.perf_counter()-_start

def getTimeMiliseconds():
    return (time.perf_counter()-_start)*1000

def printSeconds():
    print(getTimeSeconds()," seconds passed.")

def printMiliseconds():
    print(getTimeMiliseconds()," miliseconds passed.")

def FrameEligiblebyTime():
    global _lastFrameTimeSeconds
    time_cache = getTimeSeconds()
    if time_cache-_lastFrameTimeSeconds>gb.secondsForOneFrame:
        _lastFrameTimeSeconds = time_cache
        return True
    else:
        return False

specLog1 = ""
def getnSetSpecs():
    ram = psutil.virtual_memory()
    total_ram_gb = ram.total / (1024 ** 3)
    total_vram_gb = 0
    if torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    cpu_freq_max = psutil.cpu_freq().max
    cpu_freq_current = psutil.cpu_freq().current    

    gb.DINO_MODEL_VERSION = "dinov3-vitb16"
    gb.YOLO_MODEL_VERSION = "Models/YOLO/yolo26s.pt"
    #select DINO and YOLO model, seconds for one frame, DINO only or hybrid, grid size(to be implemented later, not right now) 

    if torch.cuda.is_available() and total_vram_gb >= 8:
        tier = "HIGH"
    elif torch.cuda.is_available() or (physical_cores >= 6 and total_ram_gb >= 16):
        tier = "MID"
    else:
        tier = "LOW"

    if tier == "HIGH":
        gb.DINO_MODEL_VERSION = "Models/DINO/dinov3-vitl16"
        gb.YOLO_MODEL_VERSION = "Models/YOLO/yolo26l.pt"
        gb.secondsForOneFrame = 0.1
        gb.DINO_ONLY = False

    elif tier == "MID":
        gb.DINO_MODEL_VERSION = "Models/DINO/dinov3-vitb16"
        gb.YOLO_MODEL_VERSION = "Models/YOLO/yolo26m.pt"
        gb.secondsForOneFrame = 0.5
        gb.DINO_ONLY = False

    else:  # LOW
        gb.DINO_MODEL_VERSION = "facebook/dinov2-small"
        gb.YOLO_MODEL_VERSION = "Models/YOLO/yolo26s.pt"
        gb.secondsForOneFrame = 1.0
        gb.DINO_ONLY = True

    global specLog1
    specLog1 = (
        f"[Specs] Tier: {tier} | RAM: {total_ram_gb:.1f} GB | VRAM: {total_vram_gb:.1f} GB"
        f"\nCores: {physical_cores} physical / {logical_cores} logical"
        f"\nCPU: {cpu_freq_current:.0f}/{cpu_freq_max:.0f} MHz"
        f"\nSelected DINO: {gb.DINO_MODEL_VERSION} | YOLO: {gb.YOLO_MODEL_VERSION}"
        f"\nFrame interval: {gb.secondsForOneFrame}s | DINO_ONLY: {gb.DINO_ONLY}")

def printSpecs():
    global specLog1
    gb.gui.log(specLog1)
    specLog1 = None

