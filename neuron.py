import pygame
import numpy as np
import scipy
import time
import random
import h5py # Cross-platform and open
import glob

WIDTH, HEIGHT = 32, 32
PIXEL_SIZE = 32  # How big each pixel will appear on the screen
SCREEN_SIZE = WIDTH * PIXEL_SIZE, HEIGHT * PIXEL_SIZE

class globals_class:
    suffix = ".h5"
    selected = 0 # Which variable is currently selected for "tweaking"?
    def __init__(self):
        self.vars = {}
    def sorted_keys(self):
        return sorted(self.vars)
    def selected_key(self):
        return self.sorted_keys()[self.selected]
    def set(self, name, val):
        self.vars[name] = val
    def get(self, name):
        return self.vars[name]
    def set_selected(self, value):
        self.vars[self.selected_key()] = value
    def get_selected(self):
        return self.vars[self.selected_key()]
    def save(self, filename):
        with h5py.File(filename + self.suffix, "w") as f:
            f.create_dataset("active_array", data=active_array)
            for key, value in self.vars.items():
                f.attrs[key] = value
    def load(self, filename):
        with h5py.File(filename + self.suffix, "r") as f:
            global active_array
            active_array = f["active_array"][:]
            energy_array[:] = 0
            lit_array[:] = False
            self.vars = {key: f.attrs[key] for key in f.attrs.keys()}
            print(self.vars)
    def list_files(self):
        print(glob.glob("*" + self.suffix))
    def list_vars(self):
        l = self.sorted_keys()
        for i in range(len(l)):
            print(l[i], self.vars[l[i]], ["","<-"][i==self.selected])

globals = globals_class()

# Dynamics
globals.set("MIN_LIT_ENERGY", 0.05) # Can only continue burning, if at least this much energy available
globals.set("MIN_STRIKE_IGNITION", 0.05) # Can only be lit, if at least this much ignition energy is available (and MIN_LIT_ENERGY is met)
globals.set("ENERGY_SUPPLY_PER_TICK", 0.005)
globals.set("ENERGY_CONSUMED_PER_TICK", 0.03) # When lit

pygame.init()

# Create the screen
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption('Neuron 2025')
surface = pygame.Surface((WIDTH, HEIGHT))

# State

active_array = np.zeros((WIDTH, HEIGHT), dtype=bool) # Is this location active (fed with energy, capable of ignition)?
energy_array = np.zeros((WIDTH, HEIGHT), dtype=float) # How much energy is available at this location? This is equivalent to gas plus oxygen available, constantly being replenished, and used-up while lit
lit_array = np.zeros((WIDTH, HEIGHT), dtype=bool) # Is this pixel lit (burning energy)?
ignition_array = np.zeros((WIDTH, HEIGHT), dtype=float) # This is a temporary calculation, not a state variable - just made global so we can display it

def update_display():
    white_int = 1 + 256 + 256*256
    if show_ignition:
        i = np.clip((ignition_array * 256).astype(int),0,255) # turn into a byte (and in particular, chop off any pesky fraction!))
        pygame.surfarray.blit_array(surface, i * white_int)
    else:
        e = np.clip((energy_array * 256).astype(int),0,255) # turn into a byte (and in particular, chop off any pesky fraction!))

        # show energy as white
        # pygame.surfarray.blit_array(surface, e * white_int)

        # energy as blue, lit as green, active as red
        pygame.surfarray.blit_array(surface, e + 255 * 256 * lit_array + 255 * 256 * 256 * active_array )

    for n in neurons:
        v = min(255, 64+int(n.value * 192))
        surface.set_at( n.xy, (0,v,0))

    pixels = pygame.transform.scale(surface, SCREEN_SIZE)
    for x in range(WIDTH):
        pygame.draw.line(pixels, (32,32,32), (x*PIXEL_SIZE,0), (x*PIXEL_SIZE,SCREEN_SIZE[1]), 1)
    for y in range(HEIGHT):
        pygame.draw.line(pixels, (32,32,32), (0, y*PIXEL_SIZE), (SCREEN_SIZE[0], y*PIXEL_SIZE), 1)

    screen.blit(pixels, (0, 0))
        
    pygame.display.flip()

def tick_matrix():
    global energy_array, lit_array, ignition_array
    energy_array[active_array] = np.minimum(1, energy_array[active_array] + globals.get("ENERGY_SUPPLY_PER_TICK")) # Gas flows in
    energy_array[lit_array] -= globals.get("ENERGY_CONSUMED_PER_TICK") # Gas used-up if lit
    lit_array[energy_array < globals.get("MIN_LIT_ENERGY")] = 0 # Flame goes out when energy used-up

    ignition_array = scipy.ndimage.gaussian_filter(lit_array.astype(float), sigma=1.0, mode="constant") # "mode=constant" makes it treat pixels off the edge of the screen as having no energy (default is to reflect!)
    lit_array[ (ignition_array > globals.get("MIN_STRIKE_IGNITION")) & (energy_array >= globals.get("MIN_LIT_ENERGY")) & (active_array==True)] = 1 # Light anything sufficiently-close to a source of ignition, and with enough energy available

class Neuron:
    def __init__(self, xy):
        self.xy = xy
        self.value = 0

    def tick(self):
        if self.value > 0:
            self.value -= 0.1

    def trigger(self):
        self.value = 1.0

neurons = [Neuron] * 0

def add_neuron(xy):
    neurons.append(Neuron(xy))

def find_neuron(xy):
    for n in neurons:
        if n.xy == xy:
            return(n)
    return None

def trigger_neuron(xy):
    n = find_neuron(xy)
    if n:
        n.trigger()
        return n
    return None
        

# Main loop
running = True
paused = False
show_ignition = False
while running:
    t1 = time.time()
    do_step = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            grid_x, grid_y = mouse_x // PIXEL_SIZE, mouse_y // PIXEL_SIZE

            if event.button==1:
                active_array[grid_x, grid_y] ^= True # XOR with True, i.e. invert
                energy_array[grid_x, grid_y] = 0
                lit_array[grid_x, grid_y] = False
                #if trigger_neuron( (grid_x, grid_y) ):
                #   pass
                #else:
                #    energy_array[grid_x, grid_y] = 1.0 - energy_array[grid_x, grid_y]
            else:
                if active_array[grid_x, grid_y] & (energy_array[grid_x, grid_y] >= globals.get("MIN_LIT_ENERGY")):
                    lit_array[grid_x, grid_y] = 1
                # add_neuron( (grid_x, grid_y) )
        elif event.type == pygame.KEYDOWN:
            print(event.key)
            if event.mod == 64 and event.key == ord('c'): # ^C
                running = False
            if event.key == 27: # ESC
                running = False
            if event.key == ord('p'):
                paused = not paused
                print("Paused=",paused)
            if event.key == ord('i'):
                show_ignition = not show_ignition
                print("Show Ignition=",show_ignition)
            if event.key == ord(' '):
                print("Step")
                do_step = True
            if event.key == ord('r'):
                for i in range(int(WIDTH*HEIGHT/10)): # Ignite a 1/10th of random pixels
                    lit_array[random.randrange(WIDTH), random.randrange(HEIGHT)] = 1
            if event.key == ord('s'):
                globals.list_files()
                filename = input("Enter filename to save: ")
                if filename:
                    globals.save(filename)
            if event.key == ord('l'):
                globals.list_files()
                filename = input("Enter filename to load: ")
                if filename:
                    globals.load(filename)
            if event.key == ord('v'):
                globals.list_vars()
            if event.key == pygame.K_UP:
                globals.selected -= 1
                globals.selected = globals.selected % len(globals.vars)
                globals.list_vars()
            if event.key == pygame.K_DOWN:
                globals.selected += 1
                globals.selected = globals.selected % len(globals.vars)
                globals.list_vars()
            if event.key == pygame.K_LEFT:
                globals.set_selected(globals.get_selected() / 1.05)
                globals.list_vars()
            if event.key == pygame.K_RIGHT:
                globals.set_selected(globals.get_selected() * 1.05)
                globals.list_vars()
            if event.key == ord('f'):   # Fill entire array
                active_array[:] = True
                energy_array[:] = 0
                lit_array[:] = False
            if event.key == ord('c'):   # Clear entire array
                if(input("clear array - sure?").strip().lower().startswith('y')):
                    active_array[:] = False
                    energy_array[:] = 0
                    lit_array[:] = False
            if event.key == ord('z'):   # Zero (deactivate) entire array
                energy_array[:] = 0
                lit_array[:] = False
        else:
            pass
            # print(event)

    # Update the display with the new pixel array
    update_display()

    t2 = time.time()

    if (not paused) or do_step:
        for n in neurons:
            n.tick()
        tick_matrix()

    # print(energy_array[0,0], lit_array[0,0], ignition_array[0,0])
    # print(1/(t2-t1),"fps to update display,", 1/(time.time()-t2),"fps to update matrix")

# Quit pygame
pygame.quit()

