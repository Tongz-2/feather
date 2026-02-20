import cv2
import numpy as np
import mss
import time
import ctypes
from ctypes import windll, Structure, c_long, c_int, byref, create_string_buffer
from ctypes.wintypes import RECT, POINT, DWORD, LONG, WORD
import heapq
import os
import random
import pyautogui
import keyboard  # For listening to keyboard input
import pygetwindow as gw  # To get the browser window's region dynamically
import sys

# -------------------------------
# CONFIG
# -------------------------------

TILE_SIZE = 80  # Use this based on your feather and player sizes
GRID_WIDTH = 20
GRID_HEIGHT = 15

ASSETS_DIR = "assets"
THRESHOLD = 0.65   # Edge-matching threshold

# -------------------------------
# LOAD TEMPLATES (EDGE VERSION)
# -------------------------------

def load_edge_templates(names):
    templates = []
    for name in names:
        path = os.path.join(ASSETS_DIR, name)
        img = cv2.imread(path, 0)
        if img is None:
            print(f"Could not load {path}")
            continue
        edges = cv2.Canny(img, 50, 150)
        templates.append(edges)
    return templates

player_templates = load_edge_templates([
    "player_up.png",
    "player_down.png",
    "player_left.png",
    "player_right.png"
])

feather_template_raw = cv2.imread(os.path.join(ASSETS_DIR, "feather.png"), 0)
feather_template = cv2.Canny(feather_template_raw, 50, 150)

chicken_templates = load_edge_templates([
    "chicken_right.png",
    "chicken_left.png",
    "chicken1_right.png",
    "chicken1_left.png"
])

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def find_templates(screen_edges, templates):
    matches = []
    for template in templates:
        result = cv2.matchTemplate(screen_edges, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= THRESHOLD)
        for pt in zip(*locations[::-1]):
            matches.append((pt[0], pt[1], template))
            print(f"Detected template at: {pt}")  # Debug: Show detected template positions
    return matches

def pixel_to_tile(x, y):
    return x // TILE_SIZE, y // TILE_SIZE

def build_grid(blocked_tiles):
    grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    for x, y in blocked_tiles:
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            grid[y][x] = 1
    return grid

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])

    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        x, y = current
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if grid[ny][nx] == 1:
                    continue

                tentative = g_score[current] + 1
                if (nx,ny) not in g_score or tentative < g_score[(nx,ny)]:
                    came_from[(nx,ny)] = current
                    g_score[(nx,ny)] = tentative
                    f = tentative + heuristic((nx,ny), goal)
                    heapq.heappush(open_set, (f, (nx,ny)))
    return None

# -------------------------------
# CONTROLLER + COLLECTION WITH HUMAN-LIKE BEHAVIOR
# -------------------------------

def human_move(player_tile, next_tile):
    """Move to next tile with random delay to simulate human behavior."""
    dx = next_tile[0] - player_tile[0]
    dy = next_tile[1] - player_tile[1]
    # Short micro-pause before movement (faster)
    # Random micro-pause before movement (original timing preserved)
    time.sleep(0.05 + random.uniform(0, 0.05))

    # Occasionally skip a move to simulate hesitation (2% chance)
    # Occasionally skip a move to simulate hesitation (5% chance) — keep original behavior
    if random.random() < 0.05:
        return

    # Map tile deltas to WASD keys
    key = None
    if dx > 0:
        key = 'd'
    elif dx < 0:
        key = 'a'
    elif dy > 0:
        key = 's'
    elif dy < 0:
        key = 'w'
    else:
        # Already on target tile
        return

    # Hold the key briefly to simulate a real press (helps ensure game receives the input)
    hold = 0.08 + random.uniform(0, 0.04)
    try:
        pyautogui.keyDown(key)
        time.sleep(hold)
        pyautogui.keyUp(key)
    except Exception:
        # Fallback to press if keyDown/up aren't supported in this environment
        pyautogui.press(key)

def human_collect(player_tile, feather_tiles):
    """Collect feather with random delay to simulate human reaction."""
    # Allow a small tolerance so we collect when standing on or very near the feather tile
    for f_tile in feather_tiles:
        dx = abs(player_tile[0] - f_tile[0])
        dy = abs(player_tile[1] - f_tile[1])
        if dx <= 1 and dy <= 1:
            # Random small delay before collect
            time.sleep(random.uniform(0.03, 0.12))
            # Try holding Shift a couple times to ensure the game registers the collect
            attempts = 2
            for attempt in range(attempts):
                try:
                    pyautogui.keyDown('shift')
                    time.sleep(0.06 + random.uniform(0, 0.06))
                    pyautogui.keyUp('shift')
                except Exception:
                    pyautogui.press('shift')
                print(f"Tried collect at {f_tile} (attempt {attempt+1})")
                time.sleep(0.05)
            return

def add_path_jitter(path):
    """Add jitter to the path by occasionally choosing a random neighboring tile."""
    if random.random() < 0.1:  # 10% chance to add jitter
        jitter_index = random.randint(0, len(path) - 2)  # Select a random tile in path
        x, y = path[jitter_index]
        # Randomly select a neighbor tile to swap with
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        neighbors = [n for n in neighbors if 0 <= n[0] < GRID_WIDTH and 0 <= n[1] < GRID_HEIGHT]
        if neighbors:
            new_tile = random.choice(neighbors)
            path[jitter_index] = new_tile
            print(f"Added jitter: {path[jitter_index]}")

    return path

# -------------------------------
# MAIN LOOP
# -------------------------------

# Function to get the browser window region by title
def get_browser_window_region(title_keyword):
    # Try a case-insensitive search through all windows to be more robust
    keyword = title_keyword.lower()
    all_windows = gw.getAllWindows()
    for w in all_windows:
        try:
            title = w.title or ""
        except Exception:
            title = ""
        if keyword in title.lower():
            region = {"top": w.top, "left": w.left, "width": w.width, "height": w.height}
            print(f"Found browser window by title match: '{title}' -> {region}")
            return region

    # If not found, print available windows to help debug title mismatches
    print(f"No window found containing '{title_keyword}'. Available windows:")
    for w in all_windows:
        try:
            print(f" hwnd={getattr(w,'_hWnd',None)} title='{w.title}'")
        except Exception:
            pass
    return None


def grab_window_printwindow(hwnd):
    # Capture a window's client area using PrintWindow (works even if occluded)
    # Returns an BGRA numpy array
    user32 = windll.user32
    gdi32 = windll.gdi32

    # Get client rect (width/height)
    rect = RECT()
    user32.GetClientRect(hwnd, byref(rect))
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width == 0 or height == 0:
        return None

    # Convert client coords to screen coords (not strictly required for PrintWindow but kept for clarity)
    pt = POINT()
    pt.x = rect.left
    pt.y = rect.top
    user32.ClientToScreen(hwnd, byref(pt))

    # Get window/device contexts
    hwindc = user32.GetWindowDC(hwnd)
    srcdc = gdi32.CreateCompatibleDC(hwindc)
    hbmp = gdi32.CreateCompatibleBitmap(hwindc, width, height)
    # Select bitmap into our compatible DC
    oldbmp = gdi32.SelectObject(srcdc, hbmp)

    # Print the window into the compatible DC. Try to request full content rendering.
    PW_RENDERFULLCONTENT = 0x00000002
    try:
        result = user32.PrintWindow(hwnd, srcdc, PW_RENDERFULLCONTENT)
    except Exception:
        result = user32.PrintWindow(hwnd, srcdc, 0)

    # Prepare BITMAPINFO (for a 32-bit image)
    class BITMAPINFOHEADER(Structure):
        _fields_ = [
            ("biSize", DWORD),
            ("biWidth", LONG),
            ("biHeight", LONG),
            ("biPlanes", WORD),
            ("biBitCount", WORD),
            ("biCompression", DWORD),
            ("biSizeImage", DWORD),
            ("biXPelsPerMeter", LONG),
            ("biYPelsPerMeter", LONG),
            ("biClrUsed", DWORD),
            ("biClrImportant", DWORD),
        ]

    class BITMAPINFO(Structure):
        _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", DWORD * 3)]

    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = width
    bmi.bmiHeader.biHeight = -height  # negative for top-down DIB
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 0

    # Buffer to receive pixels
    buf_size = width * height * 4
    buffer = create_string_buffer(buf_size)

    # GetDIBits from the compatible DC/bitmap
    lines = gdi32.GetDIBits(srcdc, hbmp, 0, height, buffer, byref(bmi), 0)
    if lines != height:
        # Try fallback or return None
        pass

    # Convert buffer to numpy array (BGRA)
    arr = np.frombuffer(buffer, dtype=np.uint8)
    try:
        arr = arr.reshape((height, width, 4))
    except Exception:
        arr = None

    # If image is extremely dark/blank, treat as failure so caller can fallback to mss
    if arr is not None:
        try:
            if np.mean(arr[..., :3]) < 6:  # nearly black
                arr = None
        except Exception:
            pass

    # Cleanup GDI objects
    gdi32.SelectObject(srcdc, oldbmp)
    gdi32.DeleteObject(hbmp)
    gdi32.DeleteDC(srcdc)
    user32.ReleaseDC(hwnd, hwindc)

    return arr


def capture_window_by_title(title_keyword):
    windows = gw.getWindowsWithTitle(title_keyword)
    if not windows:
        print(f"No window found with title containing '{title_keyword}'")
        return None
    hwnd = windows[0]._hWnd
    return grab_window_printwindow(hwnd)

# Example usage:
game_window_title = "mysteralegacy"  # Update this to the exact title or part of the title of your browser window
GAME_REGION = get_browser_window_region(game_window_title)
if GAME_REGION is None:
    raise Exception("Game window not found, cannot continue")

# Create the window only once at the start
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detection", 800, 600)  # Resize the window to a manageable size
# Move detection window to a visible area so it doesn't overlap the game window
DET_W, DET_H = 800, 600
user32 = windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

def move_detection_window_outside_game(game_region):
    # Try to place to the right of the game, else left, else below, else top-left
    gw_left = game_region["left"]
    gw_top = game_region["top"]
    gw_w = game_region["width"]
    gw_h = game_region["height"]

    # Candidate positions
    right_x = gw_left + gw_w + 10
    left_x = gw_left - DET_W - 10
    below_y = gw_top + gw_h + 10

    if right_x + DET_W <= screen_width:
        x = right_x
        y = gw_top
    elif left_x >= 0:
        x = left_x
        y = gw_top
    elif below_y + DET_H <= screen_height:
        x = gw_left
        y = below_y
    else:
        x = 10
        y = 10

    # Wait briefly for the OpenCV window to be created and then move it using Win32
    hwnd_det = 0
    for _ in range(10):
        hwnd_det = user32.FindWindowW(None, "Detection")
        if hwnd_det:
            break
        time.sleep(0.05)

    if hwnd_det:
        HWND_TOPMOST = -1
        SWP_SHOWWINDOW = 0x0040
        windll.user32.SetWindowPos(hwnd_det, HWND_TOPMOST, int(x), int(y), DET_W, DET_H, SWP_SHOWWINDOW)
    else:
        # Fallback to OpenCV move if HWND couldn't be found
        cv2.moveWindow("Detection", int(x), int(y))

# Place detection window now
move_detection_window_outside_game(GAME_REGION)

# Create an mss instance to use as a fallback when PrintWindow fails
sct = mss.mss()

# Add an immediate quit hotkey (pressing 'q' will exit the process immediately)
try:
    keyboard.add_hotkey('q', lambda: os._exit(0))
    print("Hotkey: press 'q' to exit immediately.")
except Exception:
    # If hotkey registration fails, we'll still check in-loop
    pass

while True:
    # Try PrintWindow first (captures the target window directly and won't include overlays)
    screenshot = capture_window_by_title(game_window_title)
    if screenshot is None:
        # PrintWindow failed; move detection window outside the game region and fallback to mss
        print("PrintWindow failed — using mss fallback and moving detection window outside game")
        move_detection_window_outside_game(GAME_REGION)
        grab = sct.grab({"top": GAME_REGION['top'], "left": GAME_REGION['left'], "width": GAME_REGION['width'], "height": GAME_REGION['height']})
        screenshot = np.array(grab)

    screen_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
    screen_edges = cv2.Canny(screen_gray, 50, 150)

    # --- Detect Player ---
    player_matches = find_templates(screen_edges, player_templates)
    if not player_matches:
        print("Player not detected")
        cv2.imshow("Detection", screenshot)
        cv2.waitKey(1)
        # Check for 'q' key press to quit
        if keyboard.is_pressed('q'):
            print("Exiting script...")
            break
        time.sleep(0.05 + random.uniform(0,0.05))
        continue

    px, py, ptemplate = player_matches[0]
    player_tile = pixel_to_tile(px, py)
    h, w = ptemplate.shape
    cv2.rectangle(screenshot, (px, py), (px+w, py+h), (0,255,0), 2)

    # --- Detect Feathers ---
    result = cv2.matchTemplate(screen_edges, feather_template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= THRESHOLD)
    feather_tiles = []

    for fx, fy in zip(*locations[::-1]):
        feather_tiles.append(pixel_to_tile(fx, fy))
        cv2.rectangle(screenshot,
                      (fx, fy),
                      (fx+feather_template.shape[1],
                       fy+feather_template.shape[0]),
                      (0,255,255), 2)

    # --- Detect Chickens ---
    chicken_matches = find_templates(screen_edges, chicken_templates)
    blocked_tiles = []

    for cx, cy, ctemplate in chicken_matches:
        tile = pixel_to_tile(cx, cy)
        blocked_tiles.append(tile)
        h, w = ctemplate.shape
        cv2.rectangle(screenshot, (cx, cy), (cx+w, cy+h), (0,0,255), 2)

    # --- Pathfinding ---
    grid = build_grid(blocked_tiles)
    path = None
    if feather_tiles:
        distances = [abs(f[0]-player_tile[0]) + abs(f[1]-player_tile[1]) for f in feather_tiles]
        target = feather_tiles[distances.index(min(distances))]
        path = astar(grid, player_tile, target)

        if path:
            print(f"Path found: {path}")  # Debug: Print out the path
            # Add path jitter for more natural behavior
            path = add_path_jitter(path)

            for tile in path:
                px = tile[0] * TILE_SIZE
                py = tile[1] * TILE_SIZE
                cv2.rectangle(screenshot,
                              (px, py),
                              (px+TILE_SIZE, py+TILE_SIZE),
                              (255,0,0), 1)

    # --- Controller + Collection (Human-like) ---
    if path:
        next_tile = path[0]
        human_move(player_tile, next_tile)

    human_collect(player_tile, feather_tiles)

    # --- Display & throttle ---
    cv2.imshow("Detection", screenshot)
    cv2.waitKey(1)

    # Check for 'q' key press to quit
    if keyboard.is_pressed('q'):  # Listen for the 'q' key
        print("Exiting script...")
        break  # This will break the loop and end the script

    time.sleep(0.05 + random.uniform(0,0.05))

# Destroy the OpenCV window once the loop ends
cv2.destroyAllWindows()
