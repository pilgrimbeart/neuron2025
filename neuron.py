import pygame
import numpy as np
import scipy
import time, random, glob, sys
import h5py # Cross-platform and open

class Cells: # Maintain a grid of cells of excitable media
    def __init__(self, screen, xy, size):
        self.grid_size = (32,32)
        # Graphics
        self.screen = screen
        self.xy = xy
        self.size = size
        self.pixel_scale = self.size[0] / self.grid_size[0] # Assume pixels are square
        self.surface = pygame.Surface(self.grid_size) # One pixel for every cell (then gets upscaled to fit screen)
        # Cells
        self.active_array =   np.zeros(self.grid_size, dtype=bool)  # Is this location active (able to change: fed with energy, capable of ignition)?
        self.energy_array =   np.zeros(self.grid_size, dtype=float) # How much energy is available at this location? This is equivalent to gas plus oxygen available, constantly being replenished, and used-up while lit
        self.lit_array =      np.zeros(self.grid_size, dtype=bool)  # Is this pixel lit (burning energy)?
        self.ignition_array = np.zeros(self.grid_size, dtype=float) # This is a temporary calculation, not a state variable - just made global so we can display it
        self.reset(False)

    def reset(self, state):
        self.active_array[:] = state
        self.zero()

    def zero(self):
        self.energy_array[:] = 0
        self.lit_array[:] = False

    def render(self):
        white_int = 1 + 256 + 256*256
        if show_ignition:
            i = np.clip((self.ignition_array * 256).astype(int),0,255) # turn into a byte (and in particular, chop off any pesky fraction!))
            pygame.surfarray.blit_array(self.surface, i * white_int)
        else:
            e = np.clip((self.energy_array * 256).astype(int),0,255) # turn into a byte (and in particular, chop off any pesky fraction!))
    
            # show energy as white
            # pygame.surfarray.blit_array(self.surface, e * white_int)
    
            # energy as blue, lit as green, active as red
            pygame.surfarray.blit_array(self.surface, e + 255 * 256 * self.lit_array + 255 * 256 * 256 * self.active_array )

        pixels = pygame.transform.scale(self.surface, self.size)
        for x in range(self.grid_size[0]+1):
            pygame.draw.line(pixels, (32,32,32), (x*self.pixel_scale,0), (x*self.pixel_scale,self.size[1]), 1)
        for y in range(self.grid_size[1]+1):
            pygame.draw.line(pixels, (32,32,32), (0, y*self.pixel_scale), (self.size[0], y*self.pixel_scale), 1)

        screen.blit(pixels, self.xy)

    def find_cell(self, screen_xy):
        cell_x, cell_y = int(screen_xy[0] / self.pixel_scale), int(screen_xy[1] / self.pixel_scale)
        hit = (0 <= cell_x < self.grid_size[0]) and (0 <= cell_y < self.grid_size[1])
        return hit,cell_x,cell_y

    def set_active(self, xy, state):
        self.active_array[xy] = state
        self.energy_array[xy] = 0

    def get_active(self, xy):
        return self.active_array[xy]

    def set_light(self, cell_xy): 
        if self.active_array[cell_xy] and (self.energy_array[cell_xy] >= globals.get("MIN_LIT_ENERGY")):
            self.lit_array[cell_xy] = 1

    def update(self):
        self.energy_array[self.active_array] = np.minimum(1, self.energy_array[self.active_array] + globals.get("ENERGY_SUPPLY_PER_TICK")) # Gas flows in
        self.energy_array[self.lit_array] -= globals.get("ENERGY_CONSUMED_PER_TICK") # Gas used-up if lit
        self.lit_array[self.energy_array < globals.get("MIN_LIT_ENERGY")] = 0 # Flame goes out when energy used-up

        self.ignition_array = scipy.ndimage.gaussian_filter(self.lit_array.astype(float), sigma=1.0, mode="constant") # "mode=constant" makes it treat pixels off the edge of the screen as having no energy (default is to reflect!)
        self.lit_array[ (self.ignition_array > globals.get("MIN_STRIKE_IGNITION")) & (self.energy_array >= globals.get("MIN_LIT_ENERGY")) & (self.active_array==True)] = 1 # Light anything sufficiently-close to a source of ignition, and with enough energy available



class Console: # Maintain a text console
    def __init__(self,screen, xy,size):
        self.screen = screen
        self.xy = xy
        self.size = size
        self.char_width_pixels = 10
        self.char_height_pixels = 16
        self.font = pygame.font.Font(pygame.font.match_font("couriernew"), 16) 
        self.chars_wide = int(size[0] / self.char_width_pixels)
        self.chars_high = int(size[1] / self.char_height_pixels) 
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

    def render(self):
        for i in range(len(self.strings)):
            textpixels = self.font.render(self.strings[i],True,(255,255,255))
            self.screen.blit(textpixels, (self.xy[0], self.xy[1] + i * self.char_height_pixels))

    def is_click_within(self, xy):
        return (self.xy[0] <= xy[0] < self.xy[0]+self.size[0]) and (self.xy[1] <= xy[1] < self.xy[1]+self.size[1])

class Chart: # Maintain a chart
    def __init__(self, scrfeen, xy, size):
        self.screen = screen
        self.xy = xy
        self.size = size

    def render(self):
        pygame.draw.rect(self.screen, (0,0,32), (self.xy[0], self.xy[1], self.size[0], self.size[1]), width=0)

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
pygame.init()
pygame.key.set_repeat(1000,100)
displays = pygame.display.get_desktop_sizes()
display = 0
if len(sys.argv) > 1:
    display = int(sys.argv[1])
(screen_width, screen_height) = displays[display]
screen = pygame.display.set_mode((screen_width, screen_height), display=display, flags=pygame.FULLSCREEN | pygame.SCALED)
pygame.display.set_caption('Neuron 2025')

cells = Cells(screen, (0,0), (screen_height, screen_height))
console = Console(screen, (screen_height, int(screen_height/2)), (screen_width - screen_height, screen_height-screen_height/2))
console.add_line("Hello everyone")
console.add_line("This is groovy")
console.add_line("Let's do some shit")
# sys.stdout = console.add_line
print("Hello matey")

chart = Chart(screen, (screen_height, 0), (screen_width - screen_height, int(screen_height/2)))

# Main loop
running = True
paused = False
probe_mode = False
show_ignition = False
dragging_state = False # Whether we are clearing or setting neurons as we drag (based on state when first clicked)
while running:
    t1 = time.time()
    do_step = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if console.is_click_within(pygame.mouse.get_pos()):
                probe_mode = not probe_mode
            else:
                (hit, grid_x, grid_y) = cells.find_cell(pygame.mouse.get_pos())
                if hit:
                    if event.button==1:
                        cells.set_active( (grid_x,grid_y), not cells.get_active((grid_x, grid_y)))
                        dragging_state = cells.get_active((grid_x, grid_y))
                    else:
                        cells.set_light( (grid_x, grid_y) )
        elif event.type == pygame.MOUSEMOTION:
            if event.buttons[0]:
                (hit, grid_x, grid_y) = cells.find_cell(pygame.mouse.get_pos())
                if hit:
                    cells.set_active( (grid_x,grid_y), dragging_state)
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
                for i in range(int(cells.grid_size[0] * cells.grid_size[1] / 10)) : # Ignite a 1/10th of random pixels
                    cells.set_light( (random.randrange(cells.grid_size[0]), random.randrange(cells.grid_size[1])) )
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
                cells.reset(True)
            elif event.key == ord('c'):   # Clear entire array
                if(input("clear array - sure?").strip().lower().startswith('y')):
                    cells.reset(False)
            elif event.key == ord('z'):   # Zero (deactivate) entire array
                cells.zero()
        else:
            pass

    # Update the display with the new pixel array
    screen.fill((0,0,0))
    cells.render()
    console.render()
    chart.render()
    pygame.display.flip()

    t2 = time.time()

    if (not paused) or do_step:
        cells.update()

# Quit pygame
pygame.quit()

