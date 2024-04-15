import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import scipy.fftpack
from docx import Document
from docx.shared import Inches
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

data, fs = sf.read('sound1.wav', dtype='float32')

print(data.dtype)
print(data.shape)
# print(data)

sd.play(data, fs)
status = sd.wait()

sf.write('sound1.wav', data, fs)

plt.subplot(3, 1, 1)
plt.plot(data[:, 0])
plt.subplot(3, 1, 2)
plt.plot(data[:, 1])
plt.subplot(3, 1, 3)
plt.plot(np.mean(data, axis=1))
plt.show()

data, fs = sf.read('sin_440Hz.wav', dtype=np.int32)
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(0,data.shape[0])/fs,data)

plt.subplot(2,1,2)
yf = scipy.fftpack.fft(data)
plt.plot(np.arange(0,fs,1.0*fs/(yf.size)),np.abs(yf))
plt.show()

fsize=2**8
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(0,data.shape[0])/fs,data)
plt.subplot(2,1,2)
yf = scipy.fftpack.fft(data,fsize)
plt.plot(np.arange(0,fs/2,fs/fsize),20*np.log10( np.abs(yf[:fsize//2])))
plt.show()


