import pygame
import numpy as np
import scipy
import time, random, glob, sys
import h5py # Cross-platform and open

pygame.init()
pygame.key.set_repeat(1000,100)
displays = pygame.display.get_desktop_sizes()
display = 0
if len(sys.argv) > 1:
    display = int(sys.argv[1])
(screen_width, screen_height) = displays[display]
screen = pygame.display.set_mode((screen_width, screen_height), display=display, flags=pygame.FULLSCREEN | pygame.SCALED)

WIDTH = HEIGHT = 32 # Number of neurons in each dimension
surface_width_pixels = surface_height_pixels = min(screen_width, screen_height) # Rendering area is a square using lhs of screen
pixel_scale = surface_width_pixels / WIDTH

class Console: # Maintain a text console
    def __init__(self):
        font_name = pygame.font.match_font("couriernew")
        self.char_width_pixels = 10
        self.char_height_pixels = 16
        self.font = pygame.font.Font(font_name, 16) 
        self.chars_wide = int((screen_width - surface_width_pixels) / self.char_width_pixels)
        self.chars_high = int(screen_height / self.char_height_pixels) 
        self.strings = ["" for i in range(self.chars_high)] # A string for every row

    def add_char(self, c):
        if len(self.strings[-1]) >= self.chars_wide:
            self.strings = self.strings[1:]
            self.strings.append("")
        self.strings[-1] += c

    def add_line(self, s):
        while s != "": # Wrap string if necessary
            self.strings = self.strings[1:]
            self.strings.append(s[0:self.chars_wide])
            s = s[self.chars_wide:]

    def render(self, screen):
        for i in range(len(self.strings)):
            textpixels = self.font.render(self.strings[i],True,(255,255,255))
            screen.blit(textpixels, (surface_width_pixels, i * self.char_height_pixels))

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


# Create the screen
pygame.display.set_caption('Neuron 2025')
surface = pygame.Surface((WIDTH, HEIGHT))

# State

active_array = np.zeros((WIDTH, HEIGHT), dtype=bool) # Is this location active (fed with energy, capable of ignition)?
energy_array = np.zeros((WIDTH, HEIGHT), dtype=float) # How much energy is available at this location? This is equivalent to gas plus oxygen available, constantly being replenished, and used-up while lit
lit_array = np.zeros((WIDTH, HEIGHT), dtype=bool) # Is this pixel lit (burning energy)?
ignition_array = np.zeros((WIDTH, HEIGHT), dtype=float) # This is a temporary calculation, not a state variable - just made global so we can display it

def update_display(screen, console):
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

    pixels = pygame.transform.scale(surface, (surface_width_pixels, surface_height_pixels))
    for x in range(WIDTH+1):
        pygame.draw.line(pixels, (32,32,32), (x*pixel_scale,0), (x*pixel_scale,surface_height_pixels), 1)
    for y in range(HEIGHT+1):
        pygame.draw.line(pixels, (32,32,32), (0, y*pixel_scale), (surface_width_pixels, y*pixel_scale), 1)

    screen.fill((0,0,0))
    screen.blit(pixels, (0, 0))
    console.render(screen)
    pygame.display.flip()

def tick_matrix():
    global energy_array, lit_array, ignition_array
    energy_array[active_array] = np.minimum(1, energy_array[active_array] + globals.get("ENERGY_SUPPLY_PER_TICK")) # Gas flows in
    energy_array[lit_array] -= globals.get("ENERGY_CONSUMED_PER_TICK") # Gas used-up if lit
    lit_array[energy_array < globals.get("MIN_LIT_ENERGY")] = 0 # Flame goes out when energy used-up

    ignition_array = scipy.ndimage.gaussian_filter(lit_array.astype(float), sigma=1.0, mode="constant") # "mode=constant" makes it treat pixels off the edge of the screen as having no energy (default is to reflect!)
    lit_array[ (ignition_array > globals.get("MIN_STRIKE_IGNITION")) & (energy_array >= globals.get("MIN_LIT_ENERGY")) & (active_array==True)] = 1 # Light anything sufficiently-close to a source of ignition, and with enough energy available

def set_active_array(xy, state):
    active_array[xy] = state
    energy_array[xy] = 0
    lit_array[xy] = False

console = Console()
console.add_line("Hello everyone")
console.add_line("This is groovy")
console.add_line("Let's do some shit")

# Main loop
running = True
paused = False
show_ignition = False
dragging_state = False # Whether we are clearing or setting neurons as we drag (based on state when first clicked)
while running:
    t1 = time.time()
    do_step = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            grid_x, grid_y = int(pygame.mouse.get_pos()[0] / pixel_scale), int(pygame.mouse.get_pos()[1] / pixel_scale)
            if grid_x < WIDTH:
                if event.button==1:
                    set_active_array( (grid_x,grid_y), not active_array[grid_x, grid_y])
                    dragging_state = active_array[grid_x, grid_y]
                else:
                    if active_array[grid_x, grid_y] & (energy_array[grid_x, grid_y] >= globals.get("MIN_LIT_ENERGY")):
                        lit_array[grid_x, grid_y] = 1
        elif event.type == pygame.MOUSEMOTION:
            if event.buttons[0]:
                grid_x, grid_y = int(pygame.mouse.get_pos()[0] / pixel_scale), int(pygame.mouse.get_pos()[1] / pixel_scale)
                if grid_x < WIDTH:
                    set_active_array( (grid_x,grid_y), dragging_state)
        elif event.type == pygame.KEYDOWN:
            if (event.key == ord('c')) and (event.mod & pygame.KMOD_CTRL): # ^C
                running = False
            elif event.key == 27: # ESC
                running = False
            elif event.key == ord('p'):
                paused = not paused
            elif event.key == ord(' '):
                paused = True
                do_step = True
            elif event.key == ord('i'):
                show_ignition = not show_ignition
                print("Show Ignition=",show_ignition)
            elif event.key >= ord('0') and event.key <= ord('9'):
                console.add_char(chr(event.key))
            elif event.key == ord('r'):
                for i in range(int(WIDTH*HEIGHT/10)): # Ignite a 1/10th of random pixels
                    lit_array[random.randrange(WIDTH), random.randrange(HEIGHT)] = 1
            elif event.key == ord('s'):
                globals.list_files()
                filename = input("Enter filename to save: ")
                if filename:
                    globals.save(filename)
            elif event.key == ord('l'):
                globals.list_files()
                filename = input("Enter filename to load: ")
                if filename:
                    globals.load(filename)
            elif event.key == ord('v'):
                globals.list_vars()
            elif event.key == pygame.K_UP:
                globals.selected -= 1
                globals.selected = globals.selected % len(globals.vars)
                globals.list_vars()
            elif event.key == pygame.K_DOWN:
                globals.selected += 1
                globals.selected = globals.selected % len(globals.vars)
                globals.list_vars()
            elif event.key == pygame.K_LEFT:
                globals.set_selected(globals.get_selected() / 1.05)
                globals.list_vars()
            elif event.key == pygame.K_RIGHT:
                globals.set_selected(globals.get_selected() * 1.05)
                globals.list_vars()
            elif event.key == ord('f'):   # Fill entire array
                active_array[:] = True
                energy_array[:] = 0
                lit_array[:] = False
            elif event.key == ord('c'):   # Clear entire array
                if(input("clear array - sure?").strip().lower().startswith('y')):
                    active_array[:] = False
                    energy_array[:] = 0
                    lit_array[:] = False
            elif event.key == ord('z'):   # Zero (deactivate) entire array
                energy_array[:] = 0
                lit_array[:] = False
        else:
            pass

    # Update the display with the new pixel array
    update_display(screen, console)

    t2 = time.time()

    if (not paused) or do_step:
        tick_matrix()

# Quit pygame
pygame.quit()

