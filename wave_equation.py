from matplotlib import pyplot as plt
import numpy as np 
from numpy import typing as np_types
from scipy.fft import fft, ifft, fft2, ifft2
import matplotlib.animation as animation
import typing

#c_func_type = typing.Union[typing.Callable[[np_types.ArrayLike[float]], np_types.ArrayLike[float]], None]
derivative_kernel = [-1/12, 4/3, -5/2, 4/3, -1/12]

class WavePropagation:
    def __init__(
            self, t_max, x_max, dt=0.01, dx=1,
            c: typing.Callable = None):
        self.t = 0
        self.frame_count = 0
        self.t_max = t_max
        self.x_max = x_max
        self.dt = dt
        self.dx = dx
        self.c_function = c

        self.border_width = 5
        self.damping = 0.01
        
        self.x = np.arange(0, x_max, self.dx)
        
    def init_wave(self, u0): 
        self.u = u0
        self.u_v = np.zeros_like(u0)
    
    def add_wave(self, u0):
        self.u += u0
        
    def update(self):
        self.u[0:self.border_width] = self.u[self.border_width+1] * 0.99
        self.u[-self.border_width-1:-1] = self.u[-self.border_width-1] * 0.99
        self.u_v[0:self.border_width] = self.u_v[self.border_width+1] * 0.99
        self.u_v[-self.border_width-1:-1] = self.u_v[-self.border_width-1] * 0.99

        c = self.c_function(2 * np.pi) / 10

        # calculate the second derivative of u with 4th order accuracy
        laplacian = np.convolve(self.u, derivative_kernel, 'same') / self.dx ** 2

        self.u_v += c ** 2 * laplacian * self.dt
        self.u += self.u_v * self.dt

        self.u *= 1 - self.damping * self.dt

        self.t += self.dt
        self.frame_count += 1
        
    def run(self):
        while self.t < self.t_max:
            self.update()


t_max = 10
x_max = 2
dt = 0.01
dx = 0.01

x = np.arange(0, x_max, dx) #+ x_max / 3
wave_freq = 500
u0 = np.exp(-wave_freq * (x - x_max / 2) ** 2) * 0.08

def wave_speed(wavelength):
    speed = np.sqrt((9.8 * wavelength / 2 / np.pi + 2 * np.pi * 0.0728 / 1000 / wavelength) * np.tanh(2 * np.pi * 10 / wavelength)) * 1
    #speed = wavelength * 0 + 1
    try:
        speed[speed == np.nan] = 0
    except TypeError:
        pass
    return speed

sim = WavePropagation(t_max, x_max, dt, dx, wave_speed)
sim.init_wave(u0)
fig, ax = plt.subplots()
fig.set_size_inches(11, 7)

def init_animate():
    sim.line = ax.plot(sim.x, sim.u)[0]
    
def update_animate(data):
    if sim.frame_count % 100 == 50:
        offset = np.random.uniform(-x_max/3, x_max/3)
        x = np.arange(offset, x_max + offset, dx)

        freq_factor = np.random.rand()
        wave_freq = np.interp(freq_factor, [0, 1], [500, 10000])
        u0 = np.exp(-wave_freq * (x - x_max / 2) ** 2) * np.interp(freq_factor, [0, 1], [0.05, 0.02]) * 10
        sim.add_wave(u0)
    
    sim.update() # 更新波形
    sim.line.set_data(sim.x, sim.u)
    return sim.line,

# 啟動動畫
ani = animation.FuncAnimation(fig, update_animate, 
                              init_func=init_animate, interval=10) 

plt.ylim([-1, 1])

plt.show()
