import pygame
import numpy as np
import scipy
import math
import json
import time, random, glob, sys

# Y axis goes downward
# 2D numpy arrays are indexed [x,y]

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

    def update(self, delta_s):
        # I = gauss(F)      How illuminated is this cell (from itself and neighbours)
        self.illumination_array = globals.get("COUPLING_GAIN") * scipy.ndimage.gaussian_filter(self.flame_array, sigma=globals.get("COUPLING_DIST"), mode="constant") 

        # F.strike(I)       Light any sufficiently-illuminated cell which is enabled and not already lit
        self.flame_array[   (self.enabled_array==True) & 
                            (self.flame_array==0) &
                            (self.illumination_array >= globals.get("MIN_STRIKE")) ] = globals.get("STRIKE_LEVEL") 

        # F.extinguish(E,F) Extinguish any cell which is burning at below the sustaining level
        self.flame_array[   (self.flame_array < globals.get("MIN_FLAME")) | (self.energy_array == 0) ] = 0

        # F tends to E      Flame brightness depends on how much energy is available
        self.flame_array = np.where(self.flame_array > 0, self.energy_array + (self.flame_array - self.energy_array) * math.exp(-1.0/globals.get("FLAME_INERTIA") * delta_s), self.flame_array)

        # E tends to gauss(E)  Energy spreads out (so a flame can steal it from its' neighbours)
        # self.energy_array = np.maximum(0, 

        # E -= F            Flame consumes energy according to how bright it's burning
        self.energy_array = np.maximum(0, self.energy_array - self.flame_array * globals.get("FLAME_CONSUME") * delta_s) 

        # E += supply       Energy flows in at a linear rate
        self.energy_array[self.enabled_array] = np.minimum(1, self.energy_array[self.enabled_array] + globals.get("SUPPLY/S") * delta_s) 

    def shift(self, dx, dy):
        def scroll(arr):
            arr = np.roll(arr, shift=dx, axis=0)
            arr = np.roll(arr, shift=dy, axis=1)
            if dx > 0:
                arr[0:dx, :] = 0
            if dx < 0:
                arr[dx:, :] = 0
            if dy > 0:
                arr[:,0:dy] = 0
            if dy < 0:
                arr[:,dy:] = 0
            return arr
        self.enabled_array = scroll(self.enabled_array)
        self.energy_array = scroll(self.energy_array)
        self.flame_array = scroll(self.flame_array)
        self.illumination_array = scroll(self.illumination_array)

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
        self.probe_font = pygame.font.Font(pygame.font.match_font("couriernew"), 16) 

    def reset(self, state):
        self.state.set_all_enableds(state)
        self.zero()

    def zero(self):
        self.state.reset_energy_and_flame()

    def shift(self,dx,dy):
        self.state.shift(dx,dy)

    def render(self, show_only, probes):
        # Cells
        white_int = 1 + 256 + 256*256
        S = self.state
        if show_only != "":
            a = [S.energy_array, S.flame_array, S.illumination_array]["EFI".index(show_only)]
            i = np.clip((a * 256).astype(int),0,255) # turn into a byte (and in particular, chop off any pesky fraction!))
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
        for i in range(len(probes)):
            textpixels = self.probe_font.render(str(i),True,(255,255,255))
            self.screen.blit(textpixels, (self.xy[0] + probes[i]["xy"][0] * self.pixel_scale, self.xy[1] + probes[i]["xy"][1] * self.pixel_scale) )

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

    def update(self, delta_s):
        self.state.update(delta_s)

    def is_click_within(self, xy):
        return (self.xy[0] <= xy[0] < self.xy[0]+self.size[0]) and (self.xy[1] <= xy[1] < self.xy[1]+self.size[1])

class Chart: # Maintain a chart (an oscilloscope which probes the cell array)
    def __init__(self, screen, xy, size):
        self.screen = screen
        self.xy = xy
        self.size = size
        self.ybot = self.xy[1] + self.size[1]
        self.set_timescale_s(1)
        self.retrig = False # When we get to right of screen, restart?
        self.time_since_trig = None # Elapsed time since trigger. None for "not triggered".
        self.energy_points = [] # (t,val)
        self.illumination_points = [] # (t,val)
        self.flame_points = [] # (t,val)
        self.focus = False
        self.font = pygame.font.Font(pygame.font.match_font("couriernew"), 16) 
        self.probes = [] # A list of dicts, each containing "xy", "energy_chart", "flame_chart" etc.

    def set_timescale_s(self, timescale_s):
        self.timescale_s = timescale_s # How many seconds does the screen width represent?
        self.scale = (self.size[0] / self.timescale_s, self.size[1] / 1.0) # Assumes that input range is (seconds, 0..1)

    def render(self):
        pygame.draw.rect(self.screen, (0,0,32), (self.xy[0], self.xy[1], self.size[0], self.size[1]), width=0) # BG

        self.screen.blit(self.font.render(("%3.1f" % self.timescale_s)+"s", True, (255,255,255)), (self.xy[0]+self.size[0]-40,self.xy[1]))

        self.screen.blit(self.font.render("ENERGY", True, energy_colour), (self.xy[0]+self.size[0]-140,self.xy[1]))
        self.screen.blit(self.font.render("FLAME", True, flame_colour), (self.xy[0]+self.size[0]-140,self.xy[1]+20))
        self.screen.blit(self.font.render("ILLUMINATION", True, illumination_colour), (self.xy[0]+self.size[0]-140,self.xy[1]+40))

        y = self.ybot - globals.get("MIN_STRIKE") * self.scale[1]
        pygame.draw.line(self.screen, (64,64,64), (self.xy[0],y), (self.xy[0]+self.size[0],y) )
        self.screen.blit(self.font.render("MIN_STRIKE", True, (64,64,64)), (self.xy[0], y) )

        y = self.ybot - globals.get("MIN_FLAME") * self.scale[1]
        self.screen.blit(self.font.render("MIN_FLAME", True, (64,64,64)), (self.xy[0], y) )
        pygame.draw.line(self.screen, (64,64,64), (self.xy[0],y), (self.xy[0]+self.size[0],y) )

        for probe in self.probes:
            if len(probe["energy_chart"])>1:
                pygame.draw.lines(self.screen, illumination_colour, False, probe["illumination_chart"], 1) # At back
                pygame.draw.lines(self.screen, energy_colour, False, probe["energy_chart"], 1) 
                pygame.draw.lines(self.screen, flame_colour, False, probe["flame_chart"], 1)

        if self.focus:
            pygame.draw.rect(self.screen, (255,255,255), (self.xy[0]+1, self.xy[1]+1, self.size[0]-2, self.size[1]-2), width=1)

    def update(self, delta_s, cells):
        if self.time_since_trig is None:
            return
        self.time_since_trig += delta_s
        x = self.xy[0] + self.time_since_trig * self.scale[0]
        if self.time_since_trig < self.timescale_s:
            for probe in self.probes:
                S = cells.state
                probe["energy_chart"].append( (x, self.ybot - S.energy_array[probe["xy"]] * self.scale[1]) )
                probe["flame_chart"].append( (x, self.ybot - S.flame_array[probe["xy"]] * self.scale[1]) )
                probe["illumination_chart"].append( (x, self.ybot - S.illumination_array[probe["xy"]] * self.scale[1]) )
        else:
            if self.retrig:
                self.trig()

    def trig(self): # reset the timebase
        for p in self.probes:
            p["energy_chart"] = []
            p["flame_chart"] = []
            p["illumination_chart"] = []
        self.time_since_trig = 0

    def add_probe(self, cell_xy):
        self.probes.append({"xy":cell_xy, "energy_chart":[], "flame_chart":[], "illumination_chart":[]})

    def delete_probe(self, cell_xy):
        elem = None
        for p in self.probes:
            if p["xy"] == cell_xy:
                elem = p
        if elem is not None:
            self.probes.remove(elem)

    def shift_probes(self, dx, dy, cells): # We need cells so we can remove any probes which fall off the edge of the display
        to_delete = []
        for p in self.probes:
            p["xy"] = (p["xy"][0] + dx, p["xy"][1] + dy)
            if (p["xy"][0] < 0) or (p["xy"][0] >= cells.state.grid_size[0]) or \
                (p["xy"][1] < 0) or (p["xy"][1] >= cells.state.grid_size[1]): # Fallen off edge
                    to_delete.append(p)
        for d in to_delete:
            self.probes.remove(d)

    def is_click_within(self, xy):
        return (self.xy[0] <= xy[0] < self.xy[0]+self.size[0]) and (self.xy[1] <= xy[1] < self.xy[1]+self.size[1])

class Console: # Maintain a text console including an input line
    def __init__(self,screen, xy,size, execute):
        self.screen = screen
        self.xy = xy
        self.size = size
        self.char_width_pixels = 10
        self.char_height_pixels = 16
        self.font = pygame.font.Font(pygame.font.match_font("couriernew"), 16) 
        self.chars_wide = int(size[0] / self.char_width_pixels)
        self.chars_high = int(size[1] / self.char_height_pixels) 
        self.input = ""
        self.strings = ["" for i in range(self.chars_high-1)] # A string for every row, going top down, minus one for the input line
        self.fps_smoothing = 0
        self.focus = False
        self.execute = execute

    def scroll(self):
        self.strings = self.strings[1:] + [""]

    def add_console_char(self, c):
        if c == chr(10):
            self.scroll()
        else:
            if len(self.strings[-1]) >= self.chars_wide:
                self.scroll()
            self.strings[-1] += c

    def add_input_char(self, c):
        if c==chr(8): # Del
            self.input = self.input[0:-1]
        elif c==chr(13): # Enter
            self.add_console_char(">")
            for x in self.input:
                self.add_console_char(x)
            self.scroll()
            self.execute(self.input)
            self.input = ""
        else:
            self.input = self.input + c

    def write(self, s): # Called by stdout (with chr(10) for a newline)
        for c in s:
            self.add_console_char(c)

    def flush(self): # Necessary method to support stdout redirection
        pass

    def render(self, s_per_frame, is_paused):
        render_strings = self.strings + [">" + self.input]
        for i in range(len(render_strings)):
            self.screen.blit(self.font.render(render_strings[i],True,(255,255,255)), (self.xy[0], self.xy[1] + i * self.char_height_pixels))
        self.fps_smoothing = self.fps_smoothing * 0.99 + s_per_frame * 0.01 # Otherwise it jitters so much you can't read it
        if is_paused:
            s = "PAUSED"
            colour = (255,255,255)
        else:
            s = str(int(1/self.fps_smoothing)) + "fps"
            colour = (64,64,0)
        self.screen.blit(self.font.render(s, True, colour), (self.xy[0]+self.size[0] - 70, self.xy[1]) )
        if self.focus:
            pygame.draw.rect(self.screen, (255,255,255), (self.xy[0]+1, self.xy[1]+1, self.size[0]-2, self.size[1]-2), width=1)

    def is_click_within(self, xy):
        return (self.xy[0] <= xy[0] < self.xy[0]+self.size[0]) and (self.xy[1] <= xy[1] < self.xy[1]+self.size[1])

class Globals:  # Maintain global variables, and deal with saving & loading all program state
    suffix = ".json"
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
    def list_files(self):
        for f in sorted(glob.glob("*.json")):
            print(f,"",end="")
        print()
    def save_file(self, filename, cells, chart):
        obj = { "enabled" : cells.state.enabled_array.tolist(), "energy" : cells.state.energy_array.tolist(), "flame" : cells.state.flame_array.tolist(),
               "probes" : chart.probes,
               "vars" : self.vars}
        with open(filename + self.suffix,"wt") as f:
            json.dump(obj, f, indent=4)
    def load_file(self, filename, cells, chart):
        try:
            with open(filename + self.suffix,"rt") as f:
                obj = json.load(f)
            cells.state.enabled_array = np.array(obj["enabled"])
            cells.state.energy_array = np.array(obj["energy"])
            cells.state.flame_array = np.array(obj["flame"])
            chart.probes = obj["probes"]
            for i in range(len(chart.probes)):
                chart.probes[i]["xy"] = tuple(chart.probes[i]["xy"]) # JSON can't store tuples, and we need them as tuples so numpy understands they're co-ordinates
            self.vars.update(obj["vars"])
        except Exception as e:
            print(f"Error loading file '{filename}': {e}")
    def list_vars(self):
        print()
        l = self.sorted_keys()
        for i in range(len(l)):
            print(l[i], self.vars[l[i]], ["","<-"][i==self.selected])

def command_execute(s):
    words = s.strip().split(" ")
    if words[0]=="ls":
        globals.list_files()
    elif words[0]=="save":
        globals.save_file(words[1], cells, chart)
    elif words[0]=="load":
        globals.load_file(words[1], cells, chart)
    else:
        print("Unrecognised command '"+s+"'")

globals = Globals()

# Dynamics
globals.set("COUPLING_DIST", 1.0) # Rate at which illumination affects neighbouring cells 
globals.set("COUPLING_GAIN", 4.0) # "Reach" from one cell to the next
globals.set("FLAME_CONSUME", 8) # How much energy the flame consumes (relative to its size)
globals.set("FLAME_INERTIA", 0.07) # How quickly the size of the flame responds to the energy available
globals.set("MIN_FLAME", 0.1) # Can only continue burning, if at least this illuminated
globals.set("MIN_STRIKE", 0.24) # Can only be lit, if at least this much illumination is happening (and MIN_LIT_ENERGY is met)
globals.set("STRIKE_LEVEL",0.2) # The level at which flame ignites
globals.set("SUPPLY/S", 1.0) # constant energy flow in

# Create the screen
pygame.init()
pygame.key.set_repeat(500,100)
displays = pygame.display.get_desktop_sizes()
display = 0
if len(sys.argv) > 1:
    display = int(sys.argv[1])
(screen_width, screen_height) = displays[display]
screen = pygame.display.set_mode((screen_width, screen_height), display=display, flags=pygame.FULLSCREEN | pygame.SCALED)
pygame.display.set_caption('Neuron 2025')

cells = Cells(screen, (0,0), (screen_height, screen_height))
cells.focus = True
console = Console(screen, (screen_height, int(screen_height/2)), (screen_width - screen_height, screen_height-screen_height/2), command_execute)
sys.stdout = console
print("Hello everyone")

chart = Chart(screen, (screen_height, 0), (screen_width - screen_height, int(screen_height/2)))

# Main loop
running = True
paused = False
modify_probe = ""
show_only = ""
dragging_state = False # Whether we are clearing or setting neurons as we drag (based on state when first clicked)
this_frame_start = time.time()
globals.load_file("recent", cells, chart)
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
                if modify_probe=="add":
                    chart.add_probe( (grid_x, grid_y) )
                    modify_probe = ""
                elif modify_probe=="delete":
                    chart.delete_probe( (grid_x, grid_y) )
                    modify_probe = ""
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
        elif event.type == pygame.TEXTINPUT: # Easier than keypresses for text input (capitalisation etc.)
            if console.focus:
                console.add_input_char(event.text)
        elif event.type == pygame.KEYDOWN:
            if event.key == 27: # ESC
                running = False
            elif (event.key == ord('c')) and (event.mod & pygame.KMOD_CTRL): # ^C
                running = False
            elif event.key == 9: # TAB
                if cells.focus:
                    cells.focus = False
                    chart.focus = True
                elif chart.focus:
                    chart.focus = False
                    console.focus = True
                else:
                    console.focus = False
                    cells.focus = True
            else:
                if console.focus:
                    if event.key in [13,8]: # These characters don't arrive from TEXTINPUT
                        console.add_input_char(chr(event.key))
                if chart.focus:
                    if event.key == ord('-'): # Zoom out, i.e. make timescale larger
                        chart.set_timescale_s(chart.timescale_s * 1.25)
                    elif event.key == ord("="): # Same as "+" so zoom in
                        chart.set_timescale_s(chart.timescale_s / 1.25)
                    elif event.key == ord('t'):
                        chart.retrig = not chart.retrig
                if cells.focus:
                    if event.key == ord(' '):
                        paused = True
                        do_step = True
                    elif event.key == ord('a'):
                        modify_probe = "add"
                    elif event.key == ord('c'):   # Clear entire array
                        cells.reset(False)
                    elif event.key == ord('d'):
                        modify_probe = "delete"
                    elif event.key == ord('f'):   # Fill entire array
                        cells.reset(True)
                    elif event.key == ord('e'):
                        show_only = "" if show_only== "E" else "E"
                    elif event.key == ord('i'):
                        show_only = "" if show_only== "I" else "I"
                    elif event.key == ord('p'):
                        paused = not paused
                    elif event.key == ord('r'):
                        for i in range(int(cells.grid_size[0] * cells.grid_size[1] / 10)) : # Ignite a 1/10th of random pixels
                            cells.set_light( (random.randrange(cells.grid_size[0]), random.randrange(cells.grid_size[1])) )
                    elif event.key == ord('z'):   # Zero (deactivate) entire array
                        cells.zero()
                    elif (event.mod & pygame.KMOD_SHIFT) and (event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]):
                        dx = (event.key==pygame.K_RIGHT) - (event.key==pygame.K_LEFT)
                        dy = (event.key==pygame.K_DOWN) - (event.key==pygame.K_UP)
                        cells.shift(dx,dy)
                        chart.shift_probes(dx,dy,cells)
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

    # Generally we pass "delta real time" into simulation from frame to frame, so that it is not affected by frame-rate (i.e. so simuluation unfolds in lock-step with real time)
    # If we see a massive delta for some reason (garbage-collection or some other disruption to our machine), we don't want to ask the simulation to leap TOO far into the future (because that will likely cause errors)
    # So we limit the maximum delta (so simulation time falls slightly behind real-time)
    delta = this_frame_start - last_frame_start 
    if delta > 1/50.0: 
        # print("ignoring slow fps",1/delta)
        delta = 1/50.0

    # Update display
    screen.fill((0,0,0))
    console.render(delta, paused)
    chart.render()
    cells.render(show_only, chart.probes)
    pygame.display.flip()

    #if(time.time() % 5 < 0.05):
    #    print(time.time())

    if (not paused) or do_step:
        cells.update(delta)
        chart.update(delta, cells)

# Quit pygame
globals.save_file("recent", cells, chart)
pygame.quit()

