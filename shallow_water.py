import numpy as np
import scipy
import cv2
import pygame
import logging
import sys
import typing

logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler(sys.stdout)])

window_size = (600, 600)
t_max = 10
x_max = 0.5
dt = 0.015
dx = 0.001

base_color = np.array([167, 207, 250]) / 255

env = cv2.imread("env.hdr", cv2.IMREAD_ANYDEPTH)


def height_to_normal_map(height_map, spacing=1.0):
    """
    Convert a height map into a normal map.

    Parameters:
    - height_map: 2D numpy array of height values.
    - spacing: Real-world distance between each point in the height map.
    
    Returns:
    - normal_map: 3D numpy array where each pixel contains the normal vector [nx, ny, nz].
    """
    # Get gradients in x and y direction (central difference)
    dx, dy = np.gradient(height_map, spacing)

    # The z component of the normal is always 1 since we're assuming the normal is based on a height map.
    dz = np.ones_like(height_map)

    # Stack the gradients to form the normal vectors [nx, ny, nz]
    normals = np.stack((-dx, -dy, dz), axis=-1)

    # Normalize the normal vectors
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normal_map = normals / (norm + 1e-8)  # Prevent division by zero

    return normal_map


def fresnel(angle, n2):
    r0 = ((1 - n2) / (1 + n2)) ** 2
    return r0 + (1 - r0) * (1 - np.cos(angle)) ** 5


def linear_to_gamma(color):
    return color ** (1 / 2.2)


class HeightField2D:
    def __init__(self, t_max: float, x_max: float, dt: float, dx: float, density=1000.):
        self.t = 0
        self.frame_count = 0

        self.t_max = t_max
        self.x_max = x_max
        self.dt = dt
        self.dx = dx
        self.density = density
        self.border_width = 10

        x_coords = np.arange(0, x_max, self.dx)
        y_coords = np.arange(0, x_max, self.dx)
        self.size = len(x_coords), len(y_coords)
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

    def to_surface(self, min=-1.0, max=1.0, shaded=False):
        if shaded:
            scaled_height = cv2.resize(self.surface_height, window_size, interpolation=cv2.INTER_CUBIC) / 10
            normal = height_to_normal_map(scaled_height, self.dx)

            light_dir = np.array([-1.0, -1.0, 1.0])
            light_dir = light_dir / np.linalg.norm(light_dir)

            image = np.einsum('ijk,k->ij', normal, light_dir)
            #view_angle = np.arccos(normal[:, :, 2])
            #fresnel_factor = fresnel(view_angle, 1.33)

            #view_dir = self.x_coords

            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            #image *= base_color
            image = linear_to_gamma(image)
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
            #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        else:
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
    
    def clamp(self, field):
        return np.clip(np.nan_to_num(field, posinf=0, neginf=0), -100000, 100000)

    def update(self):
        # Calculate the change rate of fluid height
        x_gradient = np.gradient(self.fluid_height * self.u, self.dx)[0]
        y_gradient = np.gradient(self.fluid_height * self.v, self.dx)[1]
        d_eta = (x_gradient + y_gradient)
        self.fluid_height += d_eta * self.dt

        # Calculate the change rate of u
        gravity_gradient = self.gradient(
            self.density * self.fluid_height * self.u ** 2 +
            0.5 * self.density * -9.8 * self.fluid_height ** 2
        )[0]

        uv_gradient = self.gradient(
            self.density * self.fluid_height * self.u * self.v,
        )[1]

        gravity_gradient = self.clamp(gravity_gradient)
        uv_gradient = self.clamp(uv_gradient)

        d_eta_u = -(gravity_gradient + uv_gradient) / self.density
        d_u = (d_eta_u - d_eta * self.u) / self.fluid_height
        self.u += d_u * self.dt

        # Calculate the change rate of v
        gravity_gradient = self.gradient(
            self.density * self.fluid_height * self.v ** 2 +
            0.5 * self.density * -9.8 * self.fluid_height ** 2
        )[1]

        uv_gradient = self.gradient(
            self.density * self.fluid_height * self.u * self.v,
        )[0]

        gravity_gradient = self.clamp(gravity_gradient)
        uv_gradient = self.clamp(uv_gradient)

        d_eta_v = -(gravity_gradient + uv_gradient) / self.density
        d_v = (d_eta_v - d_eta * self.v) / self.fluid_height
        self.v += d_v * self.dt

        # clamp overflow
        self.u = self.clamp(self.u)
        self.v = self.clamp(self.v)
        self.fluid_height = self.clamp(self.fluid_height)

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


class NormalModes(HeightField2D):
    def __init__(self, t_max, x_max, dt, dx, wave_speed):
        super().__init__(t_max, x_max, dt, dx)

        self.mode_weights = np.zeros(self.size) # Complex numbers that represent the phases

        self.mode_wavelengths = np.zeros(self.size)
        for n_x in range(self.size[0]):
            for n_y in range(self.size[1]):
                wavenumber = ...
                self.mode_wavelengths[n_x, n_y] = ...
        
        self.mode_maps = np.zeros(self.size + self.size)
        for n_x in range(self.size[0]):
            for n_y in range(self.size[1]):
                self.mode_maps[n_x, n_y] = self.mode(n_x, n_y)
        
        self.wave_speed = wave_speed

    def mode(self, n_x, n_y):
        return np.cos(2 * np.pi * n_x * self.x_coords / self.x_max) * np.cos(2 * np.pi * n_y * self.y_coords / self.x_max)
    
    @property
    def fluid_height(self):
        return self.mode_maps * self.mode_weights
    
    def find_weights(self, u):
        weights = np.zeros(self.size)
        for n_x in range(self.size[0]):
            for n_y in range(self.size[1]):
                weights[n_x, n_y] = np.sum(self.mode_maps[n_x, n_y] * u[n_x, n_y])
        return weights

    def update(self):
        pass


class FFTWave(HeightField2D):
    def __init__(self, t_max, x_max, dt, dx, c_function: typing.Callable):
        super().__init__(t_max, x_max, dt, dx)

        self.c_function = c_function
        self.damping = 0.008

        self.components = np.zeros(self.size, dtype=complex)
        self.fftfreq = np.fft.fftfreq(self.size[0], self.dx)
        self.fftfreq = np.meshgrid(self.fftfreq, self.fftfreq)
        self.fftfreq = np.sqrt(self.fftfreq[0] ** 2 + self.fftfreq[1] ** 2)

    @property
    def fluid_height(self):
        return np.real(np.fft.ifft2(self.components))
    
    @fluid_height.setter
    def fluid_height(self, value):
        self.components = np.fft.fft2(value)

    def add_wave(self, wave):
        self.components += np.fft.fft2(wave)

    def update(self):
        k = 2 * np.pi * self.fftfreq
        k[k == 0] = 0.1
        phase_shift = -k * self.c_function(2 * np.pi / np.abs(k)) * self.dt

        self.components *= np.exp(phase_shift * 1j)

        # damping
        wavelength_factor = 1 / 2 / np.pi * np.abs(k)
        self.components *= np.clip(1 - self.damping * wavelength_factor * dt, 0.1, 1)

        super().update()


def wave_speed(wavelength):
    speed = np.sqrt((9.8 * wavelength / 2 / np.pi + 2 * np.pi * 0.0728 / 1000 / wavelength) * np.tanh(2 * np.pi * 10 / wavelength)) * 1
    speed = np.nan_to_num(speed)
        
    return speed


def gaussian_wave(sim, x, y, std):
    xx = sim.x_coords + x
    yy = sim.y_coords + y

    u0 = np.exp(-(0.5 / std ** 2) * ((xx - x_max / 2) ** 2 + (yy - x_max / 2) ** 2))
    return u0


#sim = ShallowWater(t_max, x_max, dt, dx)
#sim = WaveEquation(t_max, x_max, dt, dx, 0.5)
sim = FFTWave(t_max, x_max, dt, dx, wave_speed)

sim.fluid_height = gaussian_wave(sim, 0, 0, 0.01) * 0.2 #+ 0.1
#sim.bed_height = sim.x_coords / 3
#sim.fluid_height -= sim.bed_height
#sim.fluid_height = np.clip(sim.fluid_height, 0, None)

pygame.init()
screen = pygame.display.set_mode(window_size)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.mouse.get_pressed()[0]:
                x = (0.5 - event.pos[0] / window_size[0]) * x_max
                y = (0.5 - event.pos[1] / window_size[1]) * x_max
                #sim.fluid_height += gaussian_wave(sim, y, x, 0.01) * 0.05
                sim.add_wave(gaussian_wave(sim, y, x, 0.004) * 0.1)
        
        if event.type == pygame.MOUSEMOTION:
            if pygame.mouse.get_pressed()[0]:
                x = (0.5 - event.pos[0] / window_size[0]) * x_max
                y = (0.5 - event.pos[1] / window_size[1]) * x_max
                #sim.fluid_height += gaussian_wave(sim, y, x, 0.01) * 0.01
                sim.add_wave(gaussian_wave(sim, y, x, 0.004) * 0.02)

    screen.fill((0, 0, 0))
    sim.update()
    screen.blit(sim.to_surface(min=-0.1, max=0.1, shaded=True), (0, 0))
    pygame.display.flip()

pygame.quit()
