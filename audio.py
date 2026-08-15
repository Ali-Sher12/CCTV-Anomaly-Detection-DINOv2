import pyttsx3
import threading


class AudioEngine:
    def __init__(self):
        self.thread = None

    def say_internal(self,text):
        engine = pyttsx3.init() # aparently creating a new engine is better because it is tied to its thread
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def say(self,text):
        if self.thread is None or self.thread.is_alive() == False:
            if text == "CRITICAL":
                text = "MAJOR ANOMALY DETECTED"
            elif text == "NORMAL":
                return
            self.thread = threading.Thread(target=self.say_internal, args=(text,), daemon = True)
            self.thread.start()