import time

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

def FrameEligiblebyTime(secondsForOneFrame):
    global _lastFrameTimeSeconds
    time_cache = getTimeSeconds()
    if time_cache-_lastFrameTimeSeconds>secondsForOneFrame:
        _lastFrameTimeSeconds = time_cache
        return True
    else:
        return False
