import pygame
import numpy as np
import scipy
import time, random, glob, sys
import h5py # Cross-platform and open

class State: # A 2D array of excitable media
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.active_array =   np.zeros(self.grid_size, dtype=bool)  # Is this location active (enabled, able to change: fed with energy, capable of ignition)?
        self.energy_array =   np.zeros(self.grid_size, dtype=float) # How much energy is available at this location? This is equivalent to gas plus oxygen available, constantly being replenished, and used-up while lit
        self.lit_array =      np.zeros(self.grid_size, dtype=bool)  # Is this pixel lit (burning energy)?
        self.illumination_array = np.zeros(self.grid_size, dtype=float) # This is a temporary calculation, not a state variable - just made global so we can display it

    def set_all_actives(self, state):
        self.active_array[:] = state

    def reset_energy_and_lit(self):
        self.energy_array[:] = 0
        self.lit_array[:] = False

    def update(self, elapsed_s):
        self.energy_array[self.active_array] = np.minimum(1, self.energy_array[self.active_array] + globals.get("SUPPLY/S") * elapsed_s) # Gas flows in
        self.energy_array[self.lit_array] -= globals.get("CONSUME/S") * elapsed_s # Gas used-up if lit
        self.lit_array[self.energy_array < globals.get("MIN_LIT")] = 0 # Flame goes out when energy used-up

        self.illumination_array = scipy.ndimage.gaussian_filter(self.lit_array.astype(float), sigma=1.0, mode="constant") # "mode=constant" makes it treat pixels off the edge of the screen as having no energy (default is to reflect!)
        self.lit_array[ (self.illumination_array > globals.get("MIN_STRIKE")) & (self.energy_array >= globals.get("MIN_LIT")) & (self.active_array==True)] = 1 # Light anything sufficiently-illuminated, and with enough energy available

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
        self.state.set_all_actives(state)
        self.zero()

    def zero(self):
        self.state.reset_energy_and_lit()

    def render(self, show_probes):
        # Cells
        white_int = 1 + 256 + 256*256
        S = self.state
        if show_ignition:
            i = np.clip((S.ignition_array * 256).astype(int),0,255) # turn into a byte (and in particular, chop off any pesky fraction!))
            pygame.surfarray.blit_array(self.surface, i * white_int)
        else:
            e = np.clip((S.energy_array * 256).astype(int),0,255) # turn into a byte (and in particular, chop off any pesky fraction!))
    
            # show energy as white
            # pygame.surfarray.blit_array(self.surface, e * white_int)
    
            # energy as blue, lit as green, active as red
            pygame.surfarray.blit_array(self.surface, e + 255 * 256 * S.lit_array + 255 * 256 * 256 * S.active_array )

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

    def set_active(self, xy, state):
        self.state.active_array[xy] = state
        self.state.energy_array[xy] = 0

    def get_active(self, xy):
        return self.state.active_array[xy]

    def set_light(self, cell_xy): 
        S = self.state
        if S.active_array[cell_xy] and (S.energy_array[cell_xy] >= globals.get("MIN_LIT")):
            S.lit_array[cell_xy] = 1

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
        self.timescale_s = 1 # How many seconds does the screen width represent?
        self.scale = (size[0] / self.timescale_s, self.size[1] / 1.0) # Assumes that input range is (seconds, 0..1)
        self.energy_points = []
        self.illumination_points = []
        self.trig_time = None # The time at the left of the pane
        self.focus = False
        self.font = pygame.font.Font(pygame.font.match_font("couriernew"), 16) 

    def render(self):
        pygame.draw.rect(self.screen, (0,0,32), (self.xy[0], self.xy[1], self.size[0], self.size[1]), width=0) # BG

        y = self.ybot - globals.get("MIN_STRIKE") * self.scale[1]
        pygame.draw.line(self.screen, (32,32,128), (self.xy[0],y), (self.xy[0]+self.size[0],y) )
        self.screen.blit(self.font.render("MIN_STRIKE", True, (32,32,128)), (self.xy[0], y) )

        y = self.ybot - globals.get("MIN_LIT") * self.scale[1]
        self.screen.blit(self.font.render("MIN_LIT", True, (128,32,32)), (self.xy[0], y) )
        pygame.draw.line(self.screen, (128,32,32), (self.xy[0],y), (self.xy[0]+self.size[0],y) )

        if len(self.energy_points)>1:
            pygame.draw.lines(self.screen, (192, 192, 192), False, self.energy_points, 1)
        if len(self.illumination_points)>1:
            pygame.draw.lines(self.screen, (255, 255, 192), False, self.illumination_points, 1)
        if self.focus:
            pygame.draw.rect(self.screen, (255,255,255), (self.xy[0]+1, self.xy[1]+1, self.size[0]-2, self.size[1]-2), width=1)

    def update(self, cells):
        if self.trig_time is None:
            return
        T = time.time() - self.trig_time
        x = self.xy[0] + T * self.scale[0]
        if T < self.timescale_s:
            for probe in cells.probes:
                self.energy_points.append( (x, self.ybot - cells.state.energy_array[probe] * self.scale[1]) )
                self.illumination_points.append( (x, self.ybot - cells.state.illumination_array[probe] * self.scale[1]) )

    def trig(self): # reset the timebase
        self.energy_points = []
        self.illumination_points = []
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
    def save(self, filename, state):
        with h5py.File(filename + self.suffix, "w") as f:
            f.create_dataset("active_array", data=state.active_array)
            f.create_dataset("energy_array", data=state.energy_array)
            f.create_dataset("lit_array", data=state.lit_array)
            for key, value in self.vars.items():
                f.attrs[key] = value
    def load(self, filename, state):
        try:
            with h5py.File(filename + self.suffix, "r") as f:
                state.active_array = f["active_array"][:]
                if "energy_array" in f:
                    state.energy_array = f["energy_array"][:]
                else:
                    state.energy_array[:] = 0
                if "lit_array" in f:
                    state.lit_array = f["lit_array"][:]
                else:
                    state.lit_array[:] = False
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
globals.set("MIN_STRIKE", 0.08) # Can only be lit, if at least this much ignition energy is available (and MIN_LIT_ENERGY is met)
globals.set("MIN_LIT", 0.2) # Can only continue burning, if at least this much energy available
globals.set("SUPPLY/S", 1.0) # Always
globals.set("CONSUME/S", 5.0) # When lit


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
show_ignition = False
dragging_state = False # Whether we are clearing or setting neurons as we drag (based on state when first clicked)
this_frame_start = time.time()
globals.load("recent", cells.state)
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
            else:
                (hit, grid_x, grid_y) = cells.find_cell(pygame.mouse.get_pos())
                if hit:
                    if add_probe:
                        cells.add_probe( (grid_x, grid_y) )
                        add_probe = False
                    else:
                        if event.button==1:
                            cells.set_active( (grid_x,grid_y), not cells.get_active((grid_x, grid_y)))
                            dragging_state = cells.get_active((grid_x, grid_y))
                        else:
                            cells.set_light( (grid_x, grid_y) )
                            chart.trig()
        elif event.type == pygame.MOUSEMOTION:
            if event.buttons[0]:
                (hit, grid_x, grid_y) = cells.find_cell(pygame.mouse.get_pos())
                if hit:
                    cells.set_active( (grid_x,grid_y), dragging_state)
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
                show_ignition = not show_ignition
                print("Show Ignition=",show_ignition)
            elif event.key == ord('l'):
                globals.list_files()
                filename = input("Enter filename to load: ")
                if filename:
                    globals.load(filename)
            elif event.key == ord('p'):
                paused = not paused
            elif event.key == ord('r'):
                for i in range(int(cells.grid_size[0] * cells.grid_size[1] / 10)) : # Ignite a 1/10th of random pixels
                    cells.set_light( (random.randrange(cells.grid_size[0]), random.randrange(cells.grid_size[1])) )
            elif event.key == ord('s'):
                globals.list_files()
                filename = input("Enter filename to save: ")
                if filename:
                    globals.save(filename)
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
    cells.render(True)
    pygame.display.flip()

    if (not paused) or do_step:
        cells.update(this_frame_start - last_frame_start)
        chart.update(cells)

# Quit pygame
globals.save("recent", cells.state)
pygame.quit()

