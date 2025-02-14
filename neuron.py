import pygame
import numpy as np
import scipy
import math
import time, random, glob, sys
import h5py # Cross-platform and open

# Colouring of cells, charts etc.:
# An enabled cell is shown dark grey even when not alight
# ENERGY is GREEN - the energy available in a cell
# FLAME is BLUE - how bright a cell is burning
# ILLUMINATION is WHITE - if in Illumination mode
energy_colour = (0,255,0)
flame_colour = (0,0,255)
illumination_colour = (255,255,255)

class State: # A 2D array of excitable media. This is the entire model of the behaviour.
    # If we "light" a cell in one binary step from off to on, as soon as the illumination of it reaches a threshold, then neighbouring cells will light on successive simulation steps, which isn't what we want.
    # (because it's not what would happen in any real-world coupled system, and it makes patterns propagate very fast so you need lots of cells to build systems)
    # So we want to apply first-order filter to illumination, i.e. x' = x*k + i*(1-k) mixing.
    # But we need to make the behaviour time-invariant (so not dependant on simulator iteration speed)
    # Therefore we use exponentials which for a first-order linear differential equation are:
    #     x' = input + (x-input) * exp(-kT)
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.enabled_array =   np.zeros(self.grid_size, dtype=bool)  # Is this location fed with energy, capable of ignition?
        self.energy_array =   np.zeros(self.grid_size, dtype=float) # How much energy is available at this location? This is equivalent to gas plus oxygen available, constantly being replenished, and used-up while lit
        self.flame_array =    np.zeros(self.grid_size, dtype=float)  # intensity of flame at this site (0 if not lit)
        self.illumination_array = np.zeros(self.grid_size, dtype=float) # Total illumination of this cell. Temporary not a state variable - persisted so we can chart it

    def set_all_enableds(self, state):
        self.enabled_array[:] = state

    def reset_energy_and_flame(self):
        self.energy_array[:] = 0
        self.flame_array[:] = 0

    def update(self, elapsed_s):
        # Gas flows in at a linear rate
        self.energy_array[self.enabled_array] = np.minimum(1, self.energy_array[self.enabled_array] + globals.get("SUPPLY/S") * elapsed_s) 
        # How illuminated is this cell (from itself and neighbours)
        self.illumination_array = globals.get("COUPLING_GAIN") * scipy.ndimage.gaussian_filter(self.flame_array, sigma=globals.get("COUPLING_DIST"), mode="constant") 
        # self.illumination_array = illum + (self.illumination_array - illum) * math.exp(-globals.get("COUPLING_RATE") * elapsed_s)
        # Light any cell which is enabled, not already lit, and sufficiently-illuminated
        self.flame_array[   (self.enabled_array==True) & 
                            (self.flame_array==0) &
                            (self.illumination_array >= globals.get("MIN_STRIKE")) ] = globals.get("STRIKE_LEVEL") 
        # Extinguish any cell which is burning at below the sustaining level
        self.flame_array[   (self.flame_array < globals.get("MIN_FLAME")) ] = 0
        # Consume energy according to how bright cell burning
        self.energy_array = np.maximum(0,self.energy_array - self.flame_array * elapsed_s) 

class Cells: # Maintain a grid of cells of state
    def __init__(self, screen, xy, size):
        self.grid_size = (32,32)
        # Graphics
        self.screen = screen
        self.xy = xy
        self.size = size
        self.pixel_scale = self.size[0] / self.grid_size[0] # Assume pixels are square
        self.surface = pygame.Surface(self.grid_size) # One pixel for every cell (then gets upscaled to fit screen)
        self.focus = False
        # Cells
        self.state = State(self.grid_size)
        self.reset(False)
        # Probes
        self.probe_font = pygame.font.Font(pygame.font.match_font("couriernew"), 16) 
        self.probes = []

    def reset(self, state):
        self.state.set_all_enableds(state)
        self.zero()

    def zero(self):
        self.state.reset_energy_and_flame()

    def render(self, show_illumination, show_probes):
        # Cells
        white_int = 1 + 256 + 256*256
        S = self.state
        if show_illumination:
            i = np.clip((S.illumination_array * 256).astype(int),0,255) # turn into a byte (and in particular, chop off any pesky fraction!))
            pygame.surfarray.blit_array(self.surface, i * white_int)
        else:
            e = np.clip((S.energy_array * 256).astype(int),0,255) # turn into a byte (and in particular, chop off any pesky fraction!))
            f = np.clip((S.flame_array * 256).astype(int),0,255)
    
            # show energy as white
            # pygame.surfarray.blit_array(self.surface, e * white_int)
    
            # energy as blue, flame as green, enabled as red
            pygame.surfarray.blit_array(self.surface, e + 255 * 256 * f + 255 * 256 * 256 * S.enabled_array )

        pixels = pygame.transform.scale(self.surface, self.size)
        screen.blit(pixels, self.xy)

        # Grid
        for x in range(self.grid_size[0]+1):
            pygame.draw.line(self.screen, (32,32,32), (self.xy[0]+x*self.pixel_scale,self.xy[1]), (self.xy[0]+x*self.pixel_scale,self.xy[1]+self.size[1]), 1)
        for y in range(self.grid_size[1]+1):
            pygame.draw.line(self.screen, (32,32,32), (self.xy[0], self.xy[1]+y*self.pixel_scale), (self.xy[0]+self.size[0], self.xy[1]+y*self.pixel_scale), 1)

        # Probes
        if show_probes:
            for i in range(len(self.probes)):
                textpixels = self.probe_font.render(str(i),True,(255,255,255))
                self.screen.blit(textpixels, (self.xy[0] + self.probes[i][0] * self.pixel_scale, self.xy[1] + self.probes[i][1] * self.pixel_scale) )

        if self.focus:
            pygame.draw.rect(self.screen, (255,255,255), (self.xy[0], self.xy[1], self.size[0], self.size[1]), width=1)

    def find_cell(self, screen_xy):
        cell_x, cell_y = int(screen_xy[0] / self.pixel_scale), int(screen_xy[1] / self.pixel_scale)
        hit = (0 <= cell_x < self.grid_size[0]) and (0 <= cell_y < self.grid_size[1])
        return hit,cell_x,cell_y

    def set_enabled(self, xy, state):
        self.state.enabled_array[xy] = state
        self.state.energy_array[xy] = 0
        self.state.flame_array[xy] = 0

    def get_enabled(self, xy):
        return self.state.enabled_array[xy]

    def set_light(self, cell_xy): 
        S = self.state
        if S.enabled_array[cell_xy] and (S.energy_array[cell_xy] >= globals.get("MIN_STRIKE")):
            S.flame_array[cell_xy] = globals.get("STRIKE_LEVEL")

    def update(self, elapsed_s):
        self.state.update(elapsed_s)

    def add_probe(self, cell_xy):
        self.probes.append(cell_xy)

    def is_click_within(self, xy):
        return (self.xy[0] <= xy[0] < self.xy[0]+self.size[0]) and (self.xy[1] <= xy[1] < self.xy[1]+self.size[1])

class Chart: # Maintain a chart
    def __init__(self, screen, xy, size):
        self.screen = screen
        self.xy = xy
        self.size = size
        self.ybot = self.xy[1] + self.size[1]
        self.timescale_s = 3 # How many seconds does the screen width represent?
        self.scale = (size[0] / self.timescale_s, self.size[1] / 1.0) # Assumes that input range is (seconds, 0..1)
        self.energy_points = [] # (t,val)
        self.illumination_points = [] # (t,val)
        self.flame_points = [] # (t,val)
        self.trig_time = None # The time at the left of the pane
        self.focus = False
        self.font = pygame.font.Font(pygame.font.match_font("couriernew"), 16) 

    def render(self):
        pygame.draw.rect(self.screen, (0,0,64), (self.xy[0], self.xy[1], self.size[0], self.size[1]), width=0) # BG

        self.screen.blit(self.font.render("ENERGY", True, energy_colour), (self.xy[0]+self.size[0]-140,self.xy[1]))
        self.screen.blit(self.font.render("FLAME", True, flame_colour), (self.xy[0]+self.size[0]-140,self.xy[1]+20))
        self.screen.blit(self.font.render("ILLUMINATION", True, illumination_colour), (self.xy[0]+self.size[0]-140,self.xy[1]+40))

        y = self.ybot - globals.get("MIN_STRIKE") * self.scale[1]
        pygame.draw.line(self.screen, (64,64,64), (self.xy[0],y), (self.xy[0]+self.size[0],y) )
        self.screen.blit(self.font.render("MIN_STRIKE", True, (64,64,64)), (self.xy[0], y) )

        y = self.ybot - globals.get("MIN_FLAME") * self.scale[1]
        self.screen.blit(self.font.render("MIN_FLAME", True, (64,64,64)), (self.xy[0], y) )
        pygame.draw.line(self.screen, (64,64,64), (self.xy[0],y), (self.xy[0]+self.size[0],y) )

        if len(self.energy_points)>1:
            pygame.draw.lines(self.screen, energy_colour, False, self.energy_points, 1) 
        if len(self.flame_points)>1:
            pygame.draw.lines(self.screen, flame_colour, False, self.flame_points, 1)
        if len(self.illumination_points)>1:
            pygame.draw.lines(self.screen, illumination_colour, False, self.illumination_points, 1)
        if self.focus:
            pygame.draw.rect(self.screen, (255,255,255), (self.xy[0]+1, self.xy[1]+1, self.size[0]-2, self.size[1]-2), width=1)

    def update(self, cells):
        if self.trig_time is None:
            return
        T = time.time() - self.trig_time
        x = self.xy[0] + T * self.scale[0]
        if T < self.timescale_s:
            for probe in cells.probes:
                S = cells.state
                self.energy_points.append( (x, self.ybot - S.energy_array[probe] * self.scale[1]) )
                self.flame_points.append( (x, self.ybot - S.flame_array[probe] * self.scale[1]) )
                self.illumination_points.append( (x, self.ybot - S.illumination_array[probe] * self.scale[1]) )
                # print("E",S.energy_array[probe],"F",S.flame_array[probe],"I",S.illumination_array[probe])

    def trig(self): # reset the timebase
        self.energy_points = []
        self.illumination_points = []
        self.flame_points = []
        self.trig_time = time.time()

    def is_click_within(self, xy):
        return (self.xy[0] <= xy[0] < self.xy[0]+self.size[0]) and (self.xy[1] <= xy[1] < self.xy[1]+self.size[1])

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
        self.fps_smoothing = 0
        self.focus = False

    def scroll(self):
        self.strings = self.strings[1:]
        self.strings.append("")

    def add_char(self, c):
        if c == chr(10):
            self.scroll()
        else:
            if len(self.strings[-1]) >= self.chars_wide:
                self.scroll()
            self.strings[-1] += c
        sys.stderr.write(c)

    def write(self, s): # Called by stdout (with chr(10) for a newline)
        for c in s:
            self.add_char(c)

    def flush(self): # Necessary method to support stdout redirection
        pass

    def render(self, s_per_frame):
        for i in range(len(self.strings)):
            self.screen.blit(self.font.render(self.strings[i],True,(255,255,255)), (self.xy[0], self.xy[1] + i * self.char_height_pixels))
        self.fps_smoothing = self.fps_smoothing * 0.99 + s_per_frame * 0.01 # Otherwise it jitters so much you can't read it
        self.screen.blit(self.font.render(str(int(1/self.fps_smoothing)) + "fps", True, (64,64,0)), (self.xy[0]+self.size[0] - 70, self.xy[1]) )
        if self.focus:
            pygame.draw.rect(self.screen, (255,255,255), (self.xy[0]+1, self.xy[1]+1, self.size[0]-2, self.size[1]-2), width=1)

    def is_click_within(self, xy):
        return (self.xy[0] <= xy[0] < self.xy[0]+self.size[0]) and (self.xy[1] <= xy[1] < self.xy[1]+self.size[1])

class Globals: 
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
    def save(self, filename, cells):
        with h5py.File(filename + self.suffix, "w") as f:
            f.create_dataset("enabled_array", data=cells.state.enabled_array)
            f.create_dataset("energy_array", data=cells.state.energy_array)
            f.create_dataset("flame_array", data=cells.state.flame_array)
            f.create_dataset("probes", data=cells.probes)
            for key, value in self.vars.items():
                f.attrs[key] = value
    def load(self, filename, cells):
        try:
            with h5py.File(filename + self.suffix, "r") as f:
                cells.state.enabled_array = f["enabled_array"][:]
                if "energy_array" in f:
                    cells.state.energy_array = f["energy_array"][:]
                else:
                    cells.state.energy_array[:] = 0
                if "flame_array" in f:
                    cells.state.flame_array = f["flame_array"][:]
                else:
                    cells.state.flame_array[:] = False
                if "probes" in f:
                    cells.probes = f["probes"][:]
                else:
                    cells.probes = []
                self.vars = {key: f.attrs[key] for key in f.attrs.keys()}
        except Exception as e:
            print(f"Error loading file '{filename}': {e}")
    def list_vars(self):
        print()
        l = self.sorted_keys()
        for i in range(len(l)):
            print(l[i], self.vars[l[i]], ["","<-"][i==self.selected])

globals = Globals()

# Dynamics
globals.set("MIN_STRIKE", 0.5) # Can only be lit, if at least this much illumination is happening (and MIN_LIT_ENERGY is met)
globals.set("MIN_FLAME", 0.2) # Can only continue burning, if at least this illuminated
globals.set("SUPPLY/S", 1.0) # Always
globals.set("COUPLING_DIST", 1.0) # Rate at which illumination affects neighbouring cells 
globals.set("COUPLING_GAIN", 1.0) # "Reach" from one cell to the next
globals.set("STRIKE_LEVEL",0.5) # The level at which flame ignites


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
sys.stdout = console
print("Hello everyone")
print("This is groovy")
print("Let's do some shit")

chart = Chart(screen, (screen_height, 0), (screen_width - screen_height, int(screen_height/2)))

# Main loop
running = True
paused = False
add_probe = False
show_illumination = False
dragging_state = False # Whether we are clearing or setting neurons as we drag (based on state when first clicked)
this_frame_start = time.time()
globals.load("recent", cells)
while running:
    last_frame_start = this_frame_start
    this_frame_start = time.time()
    do_step = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            P = pygame.mouse.get_pos()
            if chart.is_click_within(P):
                chart.focus = True
                cells.focus = False
                console.focus = False
            elif cells.is_click_within(P) and not cells.focus:
                cells.focus = True
                chart.focus = False
                console.focus = False
            elif console.is_click_within(P):
                console.focus = True
                cells.focus = False
                chart.focus = False
            (hit, grid_x, grid_y) = cells.find_cell(pygame.mouse.get_pos())
            if hit:
                if add_probe:
                    cells.add_probe( (grid_x, grid_y) )
                    add_probe = False
                else:
                    if event.button==1:
                        cells.set_enabled( (grid_x,grid_y), not cells.get_enabled((grid_x, grid_y)))
                        dragging_state = cells.get_enabled((grid_x, grid_y))
                    else:
                        cells.set_light( (grid_x, grid_y) )
                        chart.trig()
        elif event.type == pygame.MOUSEMOTION:
            if event.buttons[0]:
                (hit, grid_x, grid_y) = cells.find_cell(pygame.mouse.get_pos())
                if hit:
                    cells.set_enabled( (grid_x,grid_y), dragging_state)
        elif event.type == pygame.KEYDOWN:
            if event.key == 27: # ESC
                running = False
            elif event.key == ord(' '):
                paused = True
                do_step = True
            elif event.key >= ord('0') and event.key <= ord('9'):
                console.add_char(chr(event.key))
            elif event.key == ord('a'):
                add_probe = True
            elif (event.key == ord('c')) and (event.mod & pygame.KMOD_CTRL): # ^C
                running = False
            elif event.key == ord('c'):   # Clear entire array
                cells.reset(False)
            elif event.key == ord('f'):   # Fill entire array
                cells.reset(True)
            elif event.key == ord('i'):
                show_illumination = not show_illumination
                print("Show Illumination=",show_illumination)
            elif event.key == ord('l'):
                globals.list_files()
                filename = input("Enter filename to load: ")
                if filename:
                    globals.load(filename, cells)
            elif event.key == ord('p'):
                paused = not paused
            elif event.key == ord('r'):
                for i in range(int(cells.grid_size[0] * cells.grid_size[1] / 10)) : # Ignite a 1/10th of random pixels
                    cells.set_light( (random.randrange(cells.grid_size[0]), random.randrange(cells.grid_size[1])) )
            elif event.key == ord('s'):
                globals.list_files()
                filename = input("Enter filename to save: ")
                if filename:
                    globals.save(filename, cells)
            elif event.key == ord('v'):
                globals.list_vars()
            elif event.key == ord('z'):   # Zero (deactivate) entire array
                cells.zero()
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
        else:
            pass

    # Update the display with the new pixel array
    screen.fill((0,0,0))
    console.render(this_frame_start-last_frame_start)
    chart.render()
    cells.render(show_illumination, True)
    pygame.display.flip()

    if (not paused) or do_step:
        cells.update(this_frame_start - last_frame_start)
        chart.update(cells)

# Quit pygame
globals.save("recent", cells)
pygame.quit()

