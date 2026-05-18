import numpy as np
import sounddevice as sd
import time
import random

SAMPLE_RATE = 12000
geiger = 1

def audio_callback(outdata, frames, time, status):
    # Waveform amplitude must be in range -1..1
    global geiger
    outdata[:] = -1
    click_indices = np.random.choice(outdata.size, geiger)
    print(click_indices)
    outdata[click_indices] = 1.0

stream = sd.OutputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE/50))
stream.start()

while True:
    time.sleep(0.5)
    print(geiger)
    geiger += 1
