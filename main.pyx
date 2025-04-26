"""
Cool app to detect musical notes from microphone input.
"""
import pyaudio
import numpy as np
import matplotlib.pyplot as plt

import scipy.signal

import pyfftw


pyfftw.config.NUM_THREADS = 4

pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'

p = pyaudio.PyAudio()

N = 8096
sample_rate = 48000

data = np.zeros(N, dtype=np.float32)

def close_figure(event):
    """
    Close the matplotlib window
    """
    if event.key == 'q':
        plt.close(event.canvas.figure)
        stream.close()
        p.terminate()
        raise SystemExit

def callback(in_data, frame_count, time_info, status):
    """
    Callback function to process audio input.
    """
    global data
    data = np.frombuffer(in_data, dtype=np.float32)

    return (None, pyaudio.paContinue)

# Open the microphone stream
stream = p.open(format=pyaudio.paFloat32,
       input=True,
       channels=1,
       rate=sample_rate,
       frames_per_buffer=N,
       stream_callback=callback)

stream.start_stream()

plt.ion() # Start the graph
fig, ax = plt.subplots()
graph, = ax.plot([], [])
ax.set_xlim(0, sample_rate / 2)
ax.set_ylim(0, 5)
ax.set_title('Realtime FFT of Audio Input')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude')

fig.canvas.mpl_connect('key_press_event', close_figure)

window = scipy.signal.windows.hann(N)
fft_freq = pyfftw.interfaces.numpy_fft.fftfreq(N, d=1/sample_rate) # Get the frequency bins

while stream.is_active():
    windowed_data = data * window

    fft_data = abs(pyfftw.interfaces.numpy_fft.fft(windowed_data)) # Get the FFT data

    dominant_freq = abs(fft_freq[np.argmax(fft_data)]) # Grab the fundamental frequency
    print(f"Dominant frequency is {dominant_freq:.2f} Hz")

    graph.set_ydata(fft_data) # Draw the data
    graph.set_xdata(fft_freq)

    plt.draw()
    plt.pause(0.001)

plt.close('all') # Clean up
stream.close()
p.terminate()


