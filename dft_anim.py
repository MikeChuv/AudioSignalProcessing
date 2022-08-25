from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation 
   
n_dft = 128
time_data = np.arange(n_dft)
# kv = np.arange(-n_dft / 2, n_dft / 2)


fig = plt.figure() 
axis = plt.axes(xlim =(0, n_dft),
				ylim =(-1, 1)) 
axis.grid()

line, = axis.plot([], [], lw = 3)    

def init():
	line.set_data([], [])
	return line,
   
def animate(i):

	k_extra = i # kv[i]
	signal = np.exp(1j * 2 * np.pi * k_extra * np.arange(n_dft) / n_dft)

	line.set_data(time_data, np.real(signal))

	return line,
   
anim = FuncAnimation(fig, animate, init_func = init,
					 frames = n_dft, interval = 64, blit = True)
  
   
anim.save('continuousSineWave.mp4', 
		  writer = 'ffmpeg', fps = 15)