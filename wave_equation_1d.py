from matplotlib import pyplot as plt
import numpy as np 
from numpy import typing as np_types
from scipy.fft import fft, ifft, fft2, ifft2
import matplotlib.animation as animation
import typing
from numba import jit
import numba as nb

#c_func_type = typing.Union[typing.Callable[[np_types.ArrayLike[float]], np_types.ArrayLike[float]], None]
derivative_kernel = [-1/12, 4/3, -5/2, 4/3, -1/12]

def wave_equation_step(u, u_v, x, dx, dt, c_function, damping):
    u_freq = np.fft.fft(u)
    u_v_freq = np.fft.fft(u_v)

    return main_loop(u, u_v, x, dx, dt, c_function, damping, u_freq, u_v_freq)

#@jit
def main_loop(u, u_v, x, dx, dt, c_function, damping, u_freq, u_v_freq):
    new_u = np.zeros_like(u)
    new_u_v = np.zeros_like(u_v)

    for n in range(len(u_freq)):
        # get the wave number of current wave
        k = 2 * np.pi * np.fft.fftfreq(len(u), d=dx)[n]
        if k == 0:
            k = 0.1

        # get the wave speed of current wave
        c = c_function(2 * np.pi / np.abs(k))

        # reconstruct the decomposed wave
        wave = np.real(np.exp(1j * k * x) * u_freq[n])
        wave_v = np.real(np.exp(1j * k * x) * u_v_freq[n])

        # calculate the second derivative of the wave with 4th order accuracy
        laplacian = np.convolve(wave, derivative_kernel, 'same') / dx ** 2

        # update the wave
        wave_v += c ** 2 * laplacian * dt
        wave += wave_v * dt

        # damping
        wavelength_factor = 1 / 2 / np.pi * np.abs(k)
        wave *= 1 - damping * wavelength_factor * dt

        # add the wave back
        new_u += wave * dx / 2
        new_u_v += wave_v * dx / 2

    return new_u.copy(), new_u_v.copy()

class WavePropagation:
    def __init__(
            self, t_max, x_max, dt=0.01, dx=1.,
            c_function: typing.Callable = None):
        self.t = 0.
        self.frame_count = 0
        self.t_max = t_max
        self.x_max = x_max
        self.dt = dt
        self.dx = dx
        self.c_function = c_function

        self.border_width = 5
        self.damping = 0.02
        
        self.x = np.arange(0, x_max, self.dx)
        
    def init_wave(self, u0): 
        self.u = u0
        self.u_v = np.zeros_like(u0)

        self.u[:self.border_width] = self.u[self.border_width+1] * 0.9
        self.u[-self.border_width-1:] = self.u[-self.border_width] * 0.9
        self.u_v[:self.border_width] = self.u_v[self.border_width+1] * 0.9
        self.u_v[-self.border_width-1:] = self.u_v[-self.border_width] * 0.9

        # set the boundary to 0
        #self.u[:self.border_width] = 0
        #self.u[-self.border_width-1:] = 0
        #self.u_v[:self.border_width] = 0
        #self.u_v[-self.border_width-1:] = 0

    
    def add_wave(self, u0):
        self.u += u0
        
    def update(self):
        self.u[:self.border_width] = self.u[self.border_width+1] * 0.99
        self.u[-self.border_width-1:] = self.u[-self.border_width] * 0.99
        self.u_v[:self.border_width] = self.u_v[self.border_width+1] * 0.99
        self.u_v[-self.border_width-1:] = self.u_v[-self.border_width] * 0.99

        # set the boundary to 0
        #self.u[:self.border_width] = 0
        #self.u[-self.border_width-1:] = 0
        #self.u_v[:self.border_width] = 0
        #self.u_v[-self.border_width-1:] = 0

        self.u, self.u_v = wave_equation_step(self.u, self.u_v, self.x, self.dx, self.dt, self.c_function, self.damping)

        self.t += self.dt
        self.frame_count += 1
        
    def run(self):
        while self.t < self.t_max:
            self.update()


t_max = 10
x_max = 2
dt = 0.03
dx = 0.01

x = np.arange(0.5, x_max+0.5, dx) #+ x_max / 3
wave_freq = 200
#u0 = np.exp(-wave_freq * (x - x_max / 2) ** 2) * 0.1
#u0 = np.interp(x, [0, x_max], [-0.1, 0.1])
u0 = np.zeros_like(x)

@jit
def wave_speed(wavelength):
    speed = np.sqrt((9.8 * wavelength / 2 / np.pi + 2 * np.pi * 0.0728 / 1000 / wavelength) * np.tanh(2 * np.pi * 10 / wavelength)) * 1
    if speed == np.nan:
        speed = 0
    return speed

sim = WavePropagation(t_max, x_max, dt, dx, wave_speed)
sim.init_wave(u0)
fig, ax = plt.subplots()
fig.set_size_inches(11, 7)

def init_animate():
    sim.line = ax.fill_between(
        sim.x[sim.border_width:-sim.border_width-1],
        sim.u[sim.border_width:-sim.border_width-1],
        -1,
        color="skyblue"
    )
    
def update_animate(data):
    if sim.frame_count % 100 == 50 and sim.frame_count < 500:
        offset = np.random.uniform(-x_max/3, x_max/3)
        x = np.arange(offset, x_max + offset, dx)

        freq_factor = np.random.rand() ** 2
        wave_freq = np.interp(freq_factor, [0, 1], [500, 10000])
        u0 = np.exp(-wave_freq * (x - x_max / 2) ** 2) / wave_freq * 120 # * np.interp(freq_factor, [0, 1], [0.05, 0.02])
        sim.add_wave(u0)
    
    sim.update() # 更新波形
    ax.collections.clear()
    sim.line = ax.fill_between(
        sim.x[sim.border_width:-sim.border_width-1],
        sim.u[sim.border_width:-sim.border_width-1],
        -1,
        color="skyblue"
    )
    ax.set_title(f"t = {sim.t:.2f}")

    return sim.line,

# 啟動動畫
ani = animation.FuncAnimation(fig, update_animate, 
                              init_func=init_animate, interval=1) 

plt.ylim([-1, 1])

plt.show()
