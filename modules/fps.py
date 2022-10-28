import time


class FPS:
    def __init__(self):
        self.__pTime = 0

    def start(self):
        cTime = time.time()
        try:
            fps = 1 / (cTime - self.__pTime)
        except:
            pass
        self.__pTime = cTime

        return fps
