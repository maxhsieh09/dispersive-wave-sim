from matplotlib import pyplot as plt
import numpy as np 
from numpy import typing as np_types
from scipy.fft import fft, ifft, fft2, ifft2
import scipy
import matplotlib.animation as animation
import typing
import numba

derivative_kernel = [[0.25, 0.5, 0.25],
                     [0.5, -3.0, 0.5],
                     [0.25, 0.5, 0.25]]

@numba.jit
def wave_equation_step(u_freq, u_v_freq, k_table, u, u_v, c_function, damping, xx, yy, dt):
    new_u = np.zeros_like(u)
    new_u_v = np.zeros_like(u_v)

    for n in range(u_freq.shape[0]):
        for m in range(u_freq.shape[1]):
            # get the wave number of current wave
            k_x = k_table[n]
            k_y = k_table[m]
            k = np.sqrt(k_x**2 + k_y**2)
            
            if k == 0:
                k = 0.01

            # get the wave speed of current wave
            c = c_function(2 * np.pi / np.abs(k))

            # reconstruct the decomposed wave
            wave = np.real(np.exp(1j * (k_x * xx + k_y * yy)) * u_freq[n, m])
            wave_v = np.real(np.exp(1j * (k_x * xx + k_y * yy))
                                * (u_v_freq[n, m] + c ** 2 * u_freq[n, m] * -k**2 * dt))

            # update the wave
            #wave_v += c ** 2 * laplacian * dt
            wave += wave_v * dt

            # damping
            wavelength_factor = 1 / 2 / np.pi * np.abs(k)
            wave *= 1 - damping * wavelength_factor * dt

            # add the wave back
            new_u += wave * (dx / 2)**2
            new_u_v += wave_v * (dx / 2)**2
    return new_u, new_u_v

class WavePropagation:
    def __init__(
            self, t_max, x_max, dt=0.01, dx=1.,
            c_function: typing.Callable = None):
        self.t = 0
        self.frame_count = 0
        self.t_max = t_max
        self.x_max = x_max
        self.dt = dt
        self.dx = dx
        self.c_function = c_function

        self.border_width = 5
        self.damping = 0.03
        
        x = np.arange(0, x_max, self.dx)
        y = np.arange(0, x_max, self.dx)
        self.xx, self.yy = np.meshgrid(x, y)
        
    def init_wave(self, u0: np.ndarray): 
        self.u = u0
        self.u_v = np.zeros_like(u0)

        self.k_table = 2 * np.pi * np.fft.fftfreq(len(self.u), d=self.dx)

        #self.u[:self.border_width] = self.u[self.border_width+1] * 0.9
        #self.u[-self.border_width-1:] = self.u[-self.border_width] * 0.9
        #self.u_v[:self.border_width] = self.u_v[self.border_width+1] * 0.9
        #self.u_v[-self.border_width-1:] = self.u_v[-self.border_width] * 0.9

        # set the boundary to 0
        #self.u[:self.border_width] = 0
        #self.u[-self.border_width-1:] = 0
        #self.u_v[:self.border_width] = 0
        #self.u_v[-self.border_width-1:] = 0

    
    def add_wave(self, u0):
        self.u += u0
        
    def update(self):
        #self.u[:self.border_width] = self.u[self.border_width+1] * 0.99
        #self.u[-self.border_width-1:] = self.u[-self.border_width] * 0.99
        #self.u_v[:self.border_width] = self.u_v[self.border_width+1] * 0.99
        #self.u_v[-self.border_width-1:] = self.u_v[-self.border_width] * 0.99

        # set the boundary to 0
        self.u[:self.border_width] = 0
        self.u[-self.border_width-1:] = 0
        self.u_v[:self.border_width] = 0
        self.u_v[-self.border_width-1:] = 0

        self.u[:, :self.border_width] = 0
        self.u[:, -self.border_width-1:] = 0
        self.u_v[:, :self.border_width] = 0
        self.u_v[:, -self.border_width-1:] = 0

        u_freq = fft2(self.u)
        u_v_freq = fft2(self.u_v)

        new_u, new_u_v = wave_equation_step(u_freq, u_v_freq, self.k_table, self.u, self.u_v, self.c_function, self.damping,
                                            self.xx, self.yy, self.dt)

        # the copying that I forgot, avoids annoying linking behaviours
        self.u = new_u.copy()
        self.u_v = new_u_v.copy()

        #self.u *= 1 - self.damping * self.dt

        self.t += self.dt
        self.frame_count += 1
        
    def run(self):
        while self.t < self.t_max:
            self.update()


t_max = 10
x_max = 2
dt = 0.03
dx = 0.05

x = np.arange(0, x_max, dx)
y = np.arange(0, x_max, dx)
xx, yy = np.meshgrid(x, y)

wave_freq = 200
u0 = np.exp(-wave_freq * ((xx - x_max / 3) ** 2 + (yy - x_max / 3) ** 2)) * 0.2
#u0 = np.interp(x, [0, x_max], [-0.1, 0.1])
#u0 = np.zeros_like(xx)

@numba.jit
def wave_speed(wavelength):
    speed = np.sqrt((9.8 * wavelength / 2 / np.pi + 2 * np.pi * 0.0728 / 1000 / wavelength) * np.tanh(2 * np.pi * 10 / wavelength)) * 1
    
    if speed == 0:
        speed = 0.1
    return speed

sim = WavePropagation(t_max, x_max, dt, dx, wave_speed)
sim.init_wave(u0)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.set_size_inches(11, 11)

inside_border = slice(sim.border_width, -sim.border_width-1)

def init_animate():
    #sim.line = ax.imshow(sim.u, vmin=-0.5, vmax=0.5, animated=True, interpolation="bilinear")
    sim.line = ax.plot_surface(
        sim.xx[inside_border, inside_border],
        sim.yy[inside_border, inside_border],
        sim.u[inside_border, inside_border],
        antialiased=False
    )
    return sim.line,
    
def update_animate(data):
    if sim.frame_count % 100 == 50 and sim.frame_count < 300:
        offset = np.random.uniform(-x_max/3, x_max/3)
        xx = sim.xx + offset
        yy = sim.yy + offset

        freq_factor = np.random.rand()
        wave_freq = np.interp(freq_factor, [0, 1], [500, 10000])
        u0 = np.exp(-wave_freq * ((xx - x_max / 2) ** 2 + (yy - x_max / 2) ** 2)) / wave_freq * 500
        sim.add_wave(u0)
    
    sim.update() # 更新波形
    ax.clear()

    sim.line = ax.plot_surface(
        sim.xx[inside_border, inside_border],
        sim.yy[inside_border, inside_border],
        sim.u[inside_border, inside_border],
        antialiased=False
    )

    ax.set_zlim([-1, 1])
    ax.set_title(f"t = {sim.t:.2f}, {sim.frame_count} steps")

    return sim.line,

# 啟動動畫
ani = animation.FuncAnimation(fig, update_animate, 
                              init_func=init_animate, interval=1) 

ax.set_zlim([-1, 1])

plt.show()
