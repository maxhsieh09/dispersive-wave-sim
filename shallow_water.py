import numpy as np
import scipy
import cv2
import pygame
import logging
import sys

logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler(sys.stdout)])

window_size = (600, 600)
t_max = 10
x_max = 0.5
dt = 0.003
dx = 0.002


class HeightField2D:
    def __init__(self, t_max, x_max, dt, dx, density=1000):
        self.t = 0
        self.frame_count = 0

        self.t_max = t_max
        self.x_max = x_max
        self.dt = dt
        self.dx = dx
        self.size = int(x_max / dx), int(x_max / dx)
        self.density = density
        self.border_width = 10

        x_coords = np.arange(0, x_max, self.dx)
        y_coords = np.arange(0, x_max, self.dx)
        self.x_coords, self.y_coords = np.meshgrid(x_coords, y_coords)

        self.fluid_height = np.zeros(self.size)

    @property
    def surface_height(self):
        return self.fluid_height
    
    def gradient(self, field):
        return np.gradient(field, self.dx)
    
    def laplacian(self, field):
        derivative_kernel = [
            [0.25, 0.5, 0.25],
            [0.5, -3.0, 0.5],
            [0.25, 0.5, 0.25]
        ]
        return scipy.signal.convolve2d(field, derivative_kernel, 'same') / self.dx ** 2
    
    def update(self):
        self.t += self.dt
        self.frame_count += 1

    def to_surface(self, min=-1.0, max=1.0):
        image = np.interp(self.surface_height, [min, max], [0, 255])
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), window_size, interpolation=cv2.INTER_CUBIC)

        return pygame.surfarray.make_surface(image)
    

class ShallowWater(HeightField2D):
    def __init__(self, t_max, x_max, dt, dx, density=1000):
        super().__init__(t_max, x_max, dt, dx, density)

        # The velocity vectors (u, v) of the fluid
        self.u = np.zeros(self.size)
        self.v = np.zeros(self.size)

        self.bed_height = np.zeros(self.size)

    @property
    def surface_height(self):
        return self.fluid_height + self.bed_height

    def update(self):
        x_gradient = np.gradient(self.fluid_height * self.u, self.dx)[0]
        y_gradient = np.gradient(self.fluid_height * self.v, self.dx)[1]
        self.fluid_height += -(x_gradient + y_gradient) * self.dt

        gravity_gradient = self.gradient(
            self.density * self.fluid_height * self.u ** 2 +
            self.density * -9.8 * self.fluid_height ** 2,
        )[0]

        uv_gradient = self.gradient(
            self.density * self.fluid_height * self.u * self.v,
        )[1]

        super().update()


class WaveEquation(HeightField2D):
    def __init__(self, t_max, x_max, dt, dx, wave_speed):
        super().__init__(t_max, x_max, dt, dx)

        self.wave_v = np.zeros(self.size)
        self.wave_speed = wave_speed
        self.damping = 0.7

    def update(self):
        # set the boundary to 0
        #self.fluid_height[:self.border_width] = 0
        #self.fluid_height[-self.border_width-1:] = 0
        #self.fluid_height[:, :self.border_width] = 0
        #self.fluid_height[:, -self.border_width-1:] = 0

        #self.wave_v[:self.border_width] = 0
        #self.wave_v[-self.border_width-1:] = 0
        #self.wave_v[:, :self.border_width] = 0
        #self.wave_v[:, -self.border_width-1:] = 0

        self.wave_v += self.wave_speed ** 2 * self.laplacian(self.fluid_height) * self.dt - self.damping * self.wave_v * self.dt
        self.fluid_height += self.wave_v * self.dt

        super().update()


def gaussian_wave(sim, x, y, std):
    xx = sim.x_coords + x
    yy = sim.y_coords + y

    u0 = np.exp(-(0.5 / std ** 2) * ((xx - x_max / 2) ** 2 + (yy - x_max / 2) ** 2))
    return u0


sim = WaveEquation(t_max, x_max, dt, dx, 0.5)

sim.fluid_height = gaussian_wave(sim, 0, 0, 0.01) * 3

pygame.init()
screen = pygame.display.set_mode(window_size)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            x = (0.5 - event.pos[0] / window_size[0]) * x_max
            y = (0.5 - event.pos[1] / window_size[1]) * x_max
            sim.fluid_height += gaussian_wave(sim, y, x, 0.01)
        
        if event.type == pygame.MOUSEMOTION:
            if pygame.mouse.get_pressed()[0]:
                x = (0.5 - event.pos[0] / window_size[0]) * x_max
                y = (0.5 - event.pos[1] / window_size[1]) * x_max
                sim.fluid_height += gaussian_wave(sim, y, x, 0.01) * 0.1

    screen.fill((0, 0, 0))
    sim.update()
    screen.blit(sim.to_surface(), (0, 0))
    pygame.display.flip()

pygame.quit()
