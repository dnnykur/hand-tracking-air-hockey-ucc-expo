import json
import math
import os
import random
import sys
import threading
import time
import cv2
import mediapipe as mp
import pygame
from config import * # Ambil semua dari config.py

# ---- Header layout, computed from REAL font metrics (cross-platform safe) ----
_HEADER_TOP_PAD = 16
TITLE_Y = _HEADER_TOP_PAD
_title_bottom = TITLE_Y + FONT_MED.get_height()
_header_rule_y = _title_bottom + 8
TARGET_Y = _header_rule_y + 8
SCORE_Y = TARGET_Y + FONT_SMALL.get_height() + 16
NAME_Y = SCORE_Y + FONT_HUGE.get_height() + 4
_header_bottom = NAME_Y + FONT_SMALL.get_height() + 18

# ---- Camera preview: pojok KANAN ATAS layar
PREVIEW_W, PREVIEW_H = 260, 190
PREVIEW_X = WIDTH - PREVIEW_W - 20
PREVIEW_LABEL_Y = TITLE_Y + FONT_MED.get_height() + 8
PREVIEW_Y = PREVIEW_LABEL_Y + FONT_SMALL.get_height() + 6

# Meja mulai di bawah SEMUA elemen header
TABLE_TOP = max(_header_bottom, PREVIEW_Y + PREVIEW_H + 24)
TABLE_BOTTOM = HEIGHT - 40  # (re-affirmed here, independent of TABLE_TOP)

# WINDOW = actual OS window (freely resizable / can go fullscreen).
# RENDER_SURFACE = fixed-size internal canvas all drawing code targets.
# present() scales RENDER_SURFACE to fit WINDOW each frame (letterboxed),
# so resizing/fullscreen never distorts layout or cuts anything off.
try:
    _disp_info = pygame.display.Info()
    _init_w = max(MIN_WIN_WIDTH, min(TARGET_WIDTH, _disp_info.current_w - 100))
    _init_h = max(MIN_WIN_HEIGHT, min(TARGET_HEIGHT, _disp_info.current_h - 120))
except pygame.error:
    _init_w, _init_h = TARGET_WIDTH, TARGET_HEIGHT

WINDOW = pygame.display.set_mode((_init_w, _init_h), pygame.RESIZABLE)
RENDER_SURFACE = pygame.Surface((WIDTH, HEIGHT))
SCREEN = RENDER_SURFACE  # name used by all the _draw_* methods below
CLOCK = pygame.time.Clock()
IS_FULLSCREEN = False

# Bikin RENDER_SURFACE agar pas ama WINDOW sekarang
def present():
    window = pygame.display.get_surface()
    win_w, win_h = window.get_size()
    scale = max(min(win_w / WIDTH, win_h / HEIGHT), 0.01)
    scaled_w, scaled_h = max(1, int(WIDTH * scale)), max(1, int(HEIGHT * scale))
    scaled = pygame.transform.smoothscale(RENDER_SURFACE, (scaled_w, scaled_h))
    window.fill((0, 0, 0))
    window.blit(scaled, ((win_w - scaled_w) // 2, (win_h - scaled_h) // 2))
    pygame.display.flip()

def toggle_fullscreen():
    global IS_FULLSCREEN
    if IS_FULLSCREEN:
        pygame.display.set_mode((_init_w, _init_h), pygame.RESIZABLE)
        IS_FULLSCREEN = False
    else:
        info = pygame.display.Info()
        pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
        IS_FULLSCREEN = True

# ============================================================
# ONE EURO FILTER (adaptive smoothing: halus saat diam, responsif saat cepat)
# ============================================================
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t):
        if self.t_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-6)

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat

# ============================================================
# HAND TRACKER (MediaPipe wrapper) Thread jalan sendiri sesanggup CPU
# ============================================================
class HandTracker:
    def __init__(self, cam_index=CAM_INDEX):
        self.cap = cv2.VideoCapture(cam_index)
        # MJPG: di banyak webcam Linux/V4L2 ini jauh lebih cepat daripada raw
        # YUYV di resolusi lebih tinggi -> mengurangi lag tracking.
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except AttributeError:
            pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_REQUEST_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_REQUEST_H)
        self.cap.set(cv2.CAP_PROP_FPS, CAM_REQUEST_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # jangan proses frame basi/nge-queue
        self.available = self.cap.isOpened()

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            model_complexity=HAND_MODEL_COMPLEXITY,
            max_num_hands=2,
            min_detection_confidence=HAND_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_MIN_TRACKING_CONFIDENCE,
        )
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        self._lock = threading.Lock()
        self.last_frame = None            # background terakhir buat preview
        self.last_hand_landmarks = []     # kerangka tracking
        self.p1_pos = None      # (x, y) normalized 0..1, or None if not seen this frame
        self.p2_pos = None
        self.frame_id = 0       # naik setiap ada frame baru terselesaikan
        self.last_process_ms = 0.0  # buat debug: berapa lama 1 iterasi tracking

        self._running = self.available
        self._thread = None
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    # Bantu MediaPipe deteksi di ruangan redup tanpa over expose
    def _enhance_low_light(self, bgr_frame):
        lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _loop(self):
        while self._running: # Jalan terus di background biar proses cepat
            t0 = time.perf_counter()
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)  # mirror biar kerasa natural

            if ENHANCE_LOW_LIGHT:
                frame = self._enhance_low_light(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False  # perf hint for mediapipe
            result = self.hands.process(rgb)

            hands_found = result.multi_hand_landmarks or []
            p1_pos, p2_pos = None, None

            # Kalau tangan ada 2, urutin sesuai posisi relatif P1 paling kiri, P2 paling kanan
            if len(hands_found) >= 2:
                tips = sorted(
                    (hl.landmark[CONTROL_LANDMARK] for hl in hands_found),
                    key=lambda w: w.x,
                )
                p1_pos = (tips[0].x, tips[0].y)
                p2_pos = (tips[-1].x, tips[-1].y)
            elif len(hands_found) == 1:
                tip = hands_found[0].landmark[CONTROL_LANDMARK]
                if tip.x < 0.5:
                    p1_pos = (tip.x, tip.y)
                else:
                    p2_pos = (tip.x, tip.y)

            with self._lock:
                self.last_frame = frame
                self.last_hand_landmarks = hands_found
                self.p1_pos = p1_pos
                self.p2_pos = p2_pos
                self.frame_id += 1
                self.last_process_ms = (time.perf_counter() - t0) * 1000

    #Baca p1_pos/p2_pos secara thread-safe (dipanggil dari game loop)
    def get_positions(self):
        with self._lock:
            return self.p1_pos, self.p2_pos

    # Buat debugging
    def get_preview_surface(self, size=(220, 165)):
        with self._lock:
            frame = self.last_frame
            hand_landmarks_list = self.last_hand_landmarks
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if SHOW_HAND_SKELETON and hand_landmarks_list:
            frame = frame.copy()
            for hand_landmarks in hand_landmarks_list:
                control_pt = hand_landmarks.landmark[CONTROL_LANDMARK]
                color_rgb = COLOR_P1 if control_pt.x < 0.5 else COLOR_P2
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=color_bgr, thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=color_bgr, thickness=2),
                )
                # Tandai titik yang BENERAN dipakai buat kontrol paddle (biar kelihatan jelas kalau kontrolnya dari ujung jari, bukan wrist).
                cx, cy = int(control_pt.x * w), int(control_pt.y * h)
                cv2.circle(frame, (cx, cy), 10, color_bgr, 2)
                cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
        small = cv2.resize(frame, size)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        surf = pygame.image.frombuffer(rgb.tobytes(), size, "RGB")
        return surf

    def release(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.available:
            self.cap.release()
        self.hands.close()

# ============================================================
# GAME ENTITIES
# ============================================================
class Paddle:
    def __init__(self, x, y, side, color):
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y
        self.side = side  # "left" or "right"
        self.color = color
        self.vx = 0.0
        self.vy = 0.0
        # Adaptive smoothing: halus saat tangan nyaris diam, tapi tetap ngikutin cepat saat tangan gerak cepat.
        self.filter_x = OneEuroFilter(PADDLE_FILTER_MIN_CUTOFF, PADDLE_FILTER_BETA, PADDLE_FILTER_D_CUTOFF)
        self.filter_y = OneEuroFilter(PADDLE_FILTER_MIN_CUTOFF, PADDLE_FILTER_BETA, PADDLE_FILTER_D_CUTOFF)

    def set_target(self, nx, ny):
        """nx, ny: normalized 0..1 hand position within its own camera half."""
        self.prev_x, self.prev_y = self.x, self.y

        if self.side == "left":
            min_x, max_x = TABLE_LEFT + PADDLE_RADIUS, TABLE_MID_X - PADDLE_RADIUS
            local_x = nx * 2.0  # map 0..0.5 -> 0..1
        else:
            min_x, max_x = TABLE_MID_X + PADDLE_RADIUS, TABLE_RIGHT - PADDLE_RADIUS
            local_x = (nx - 0.5) * 2.0  # map 0.5..1 -> 0..1

        # Gain di sekitar titik tengah: gerakan tangan yang kecil jadi tetap bisa nyampe ujung meja, gak perlu gerak tangan sejauh setengah frame.
        local_x = 0.5 + (local_x - 0.5) * PADDLE_SENSITIVITY
        ny = 0.5 + (ny - 0.5) * PADDLE_SENSITIVITY

        local_x = min(max(local_x, 0.0), 1.0)
        ny = min(max(ny, 0.0), 1.0)

        target_x = min_x + local_x * (max_x - min_x)
        target_y = TABLE_TOP + PADDLE_RADIUS + ny * (
            (TABLE_BOTTOM - PADDLE_RADIUS) - (TABLE_TOP + PADDLE_RADIUS)
        )

        now = time.perf_counter()
        self.x = self.filter_x.filter(target_x, now)
        self.y = self.filter_y.filter(target_y, now)

        self.vx = self.x - self.prev_x
        self.vy = self.y - self.prev_y

    def hold_position(self):
        """Kalau tangan hilang dari frame, paddle tetap diam (sesuai PRD scenario 3)."""
        self.prev_x, self.prev_y = self.x, self.y
        self.vx = 0.0
        self.vy = 0.0
        # NOTE: filter state sengaja TIDAK direset di sini -- begitu tangan terdeteksi lagi, One Euro Filter otomatis "snap" ke posisi baru (karena delta waktu besar), jadi paddle tidak nyangkut di posisi lama.

    def draw(self, surf):
        pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), PADDLE_RADIUS)
        pygame.draw.circle(surf, (255, 255, 255), (int(self.x), int(self.y)), PADDLE_RADIUS, 3)
        pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), 10)

class Puck:
    def __init__(self):
        self.reset()

    def reset(self, direction=None):
        self.x = WIDTH / 2
        self.y = (TABLE_TOP + TABLE_BOTTOM) / 2
        angle = random.uniform(-0.5, 0.5)
        if direction is None:
            direction = random.choice([-1, 1])
        speed = PUCK_RESET_SPEED
        self.vx = math.cos(angle) * speed * direction
        self.vy = math.sin(angle) * speed

    def update(self, paddles):
        self.x += self.vx
        self.y += self.vy
        self.vx *= PUCK_FRICTION
        self.vy *= PUCK_FRICTION

        # Pantulan atas/bawah -- PUCK_RESTITUTION > 1 bikin lebih memantul
        if self.y - PUCK_RADIUS <= TABLE_TOP:
            self.y = TABLE_TOP + PUCK_RADIUS
            self.vy *= -PUCK_RESTITUTION
        elif self.y + PUCK_RADIUS >= TABLE_BOTTOM:
            self.y = TABLE_BOTTOM - PUCK_RADIUS
            self.vy *= -PUCK_RESTITUTION

        # Pantulan kiri/kanan kecuali di gawang
        goal_top = (TABLE_TOP + TABLE_BOTTOM) / 2 - GOAL_HALF_HEIGHT
        goal_bottom = (TABLE_TOP + TABLE_BOTTOM) / 2 + GOAL_HALF_HEIGHT
        in_goal_range = goal_top <= self.y <= goal_bottom

        if self.x - PUCK_RADIUS <= TABLE_LEFT and not in_goal_range:
            self.x = TABLE_LEFT + PUCK_RADIUS
            self.vx *= -PUCK_RESTITUTION
        if self.x + PUCK_RADIUS >= TABLE_RIGHT and not in_goal_range:
            self.x = TABLE_RIGHT - PUCK_RADIUS
            self.vx *= -PUCK_RESTITUTION

        # Pas kena paddles
        for p in paddles:
            dx = self.x - p.x
            dy = self.y - p.y
            dist = math.hypot(dx, dy)
            min_dist = PUCK_RADIUS + PADDLE_RADIUS
            if dist < min_dist and dist > 0:
                nx, ny = dx / dist, dy / dist
                overlap = min_dist - dist
                self.x += nx * overlap
                self.y += ny * overlap

                # Reflect velocity + transfer some paddle velocity (feels responsive)
                speed_in = math.hypot(self.vx, self.vy)
                dot = self.vx * nx + self.vy * ny
                self.vx -= 2 * dot * nx
                self.vy -= 2 * dot * ny
                self.vx += p.vx * 0.6
                self.vy += p.vy * 0.6

                # *1.2 (bukan cuma *1.05) -> pukulan paddle kerasa lebih "nampol"
                speed = max(math.hypot(self.vx, self.vy), speed_in * 1.12, 6)
                speed = min(speed, MAX_PUCK_SPEED)
                angle = math.atan2(self.vy, self.vx)
                self.vx = math.cos(angle) * speed
                self.vy = math.sin(angle) * speed

        # top speed bola
        speed = math.hypot(self.vx, self.vy)
        if speed > MAX_PUCK_SPEED:
            scale = MAX_PUCK_SPEED / speed
            self.vx *= scale
            self.vy *= scale

    def check_goal(self):
        if self.x + PUCK_RADIUS < TABLE_LEFT:
            return "left"
        if self.x - PUCK_RADIUS > TABLE_RIGHT:
            return "right"
        return None

    def draw(self, surf):
        pygame.draw.circle(surf, (0, 0, 0, 0), (int(self.x), int(self.y)), PUCK_RADIUS + 4)
        pygame.draw.circle(surf, COLOR_PUCK, (int(self.x), int(self.y)), PUCK_RADIUS)
        pygame.draw.circle(surf, (120, 130, 150), (int(self.x), int(self.y)), PUCK_RADIUS, 2)

# ============================================================
# LEADERBOARD
# ============================================================
class Leaderboard:
    def __init__(self, path=LEADERBOARD_FILE):
        self.path = path
        self.entries = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=2)
        except OSError:
            pass

    def submit(self, name, score):
        name = (name.strip() or "Player")[:MAX_NAME_LEN]
        score = max(0, min(score, 999))
        self.entries.append({"playerName": name, "score": score, "createdAt": time.time()})
        self._save()

    def top(self, n=10):
        return sorted(self.entries, key=lambda e: (-e["score"], e["createdAt"]))[:n]

# ============================================================
# GAME STATES
# ============================================================
STATE_NAME_ENTRY = "name_entry"
STATE_COUNTDOWN = "countdown"
STATE_PLAYING = "playing"
STATE_GOAL_PAUSE = "goal_pause"
STATE_GAME_OVER = "game_over"
STATE_LEADERBOARD = "leaderboard"

class Game:
    def __init__(self):
        self.tracker = HandTracker()
        self.leaderboard = Leaderboard()

        self.state = STATE_NAME_ENTRY
        self.active_field = 0  # 0 = player1 name, 1 = player2 name
        self.p1_name = ""
        self.p2_name = ""
        self.name_error = ""

        self.reset_match()

        self.countdown_start = 0
        self.goal_pause_start = 0
        self.last_scorer = None

        self._preview_surface = None
        self._preview_frame_id = -1

    def reset_match(self):
        self.paddle1 = Paddle(TABLE_LEFT + 100, HEIGHT / 2, "left", COLOR_P1)
        self.paddle2 = Paddle(TABLE_RIGHT - 100, HEIGHT / 2, "right", COLOR_P2)
        self.puck = Puck()
        self.score1 = 0
        self.score2 = 0

    # ---------------- input handling ----------------
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.quit()

        if event.type == pygame.VIDEORESIZE and not IS_FULLSCREEN:
            # Accept the real new size; RENDER_SURFACE stays fixed internally,
            # present() scales it to fit -> true adaptive resize.
            pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F11:
                toggle_fullscreen()
                return
            if self.state == STATE_NAME_ENTRY:
                self._handle_name_entry_key(event)
            elif self.state == STATE_GAME_OVER:
                if event.key == pygame.K_RETURN:
                    self.leaderboard.submit(
                        self.p1_name if self.score1 > self.score2 else self.p2_name,
                        max(self.score1, self.score2),
                    )
                    self.state = STATE_LEADERBOARD
                elif event.key == pygame.K_r:
                    self.reset_match()
                    self.state = STATE_COUNTDOWN
                    self.countdown_start = time.time()
            elif self.state == STATE_LEADERBOARD:
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    self.reset_match()
                    self.p1_name, self.p2_name = "", ""
                    self.active_field = 0
                    self.name_error = ""
                    self.state = STATE_NAME_ENTRY
            if event.key == pygame.K_ESCAPE:
                if IS_FULLSCREEN:
                    toggle_fullscreen()  # exit fullscreen first, don't quit yet
                else:
                    self.quit()

    def _handle_name_entry_key(self, event):
        field = "p1_name" if self.active_field == 0 else "p2_name"
        current = getattr(self, field)
        if event.key == pygame.K_TAB or event.key == pygame.K_RETURN:
            if self.active_field == 0:
                if not self.p1_name.strip():
                    self.name_error = "Nama Player 1 wajib diisi."
                    return
                self.active_field = 1
            else:
                if not self.p1_name.strip():
                    self.name_error = "Nama Player 1 wajib diisi."
                    self.active_field = 0
                    return
                if not self.p2_name.strip():
                    self.name_error = "Nama Player 2 wajib diisi."
                    return
                self.state = STATE_COUNTDOWN
                self.countdown_start = time.time()
        elif event.key == pygame.K_BACKSPACE:
            setattr(self, field, current[:-1])
        else:
            ch = event.unicode
            if ch and ch.isprintable() and len(current) < MAX_NAME_LEN:
                setattr(self, field, current + ch)
                self.name_error = ""

    def quit(self):
        self.tracker.release()
        pygame.quit()
        sys.exit(0)

    # ---------------- update ----------------
    def update(self):
        # NOTE: tidak ada lagi tracker.update() di sini -- kamera + MediaPipe jalan sendiri di background thread (lihat HandTracker._loop), jadi game loop gak pernah nunggu mereka. Cukup baca posisi terakhirnya.
        if self.state in (STATE_PLAYING, STATE_COUNTDOWN, STATE_GOAL_PAUSE):
            p1_pos, p2_pos = self.tracker.get_positions()
            if p1_pos:
                self.paddle1.set_target(*p1_pos)
            else:
                self.paddle1.hold_position()
            if p2_pos:
                self.paddle2.set_target(*p2_pos)
            else:
                self.paddle2.hold_position()

        if self.state == STATE_COUNTDOWN:
            if time.time() - self.countdown_start >= 3.4:
                self.state = STATE_PLAYING

        elif self.state == STATE_PLAYING:
            self.puck.update([self.paddle1, self.paddle2])
            goal = self.puck.check_goal()
            if goal == "left":
                self.score2 += 1
                self.last_scorer = self.p2_name
                self._enter_goal_pause()
            elif goal == "right":
                self.score1 += 1
                self.last_scorer = self.p1_name
                self._enter_goal_pause()

        elif self.state == STATE_GOAL_PAUSE:
            if time.time() - self.goal_pause_start >= 1.6:
                if max(self.score1, self.score2) >= WIN_SCORE:
                    self.state = STATE_GAME_OVER
                else:
                    self.puck.reset()
                    self.state = STATE_COUNTDOWN
                    self.countdown_start = time.time()

    def _enter_goal_pause(self):
        self.puck.vx, self.puck.vy = 0, 0
        self.goal_pause_start = time.time()
        self.state = STATE_GOAL_PAUSE

    # ---------------- draw ----------------
    def draw(self):
        SCREEN.fill(COLOR_BG)
        self._draw_background()
        self._draw_header()

        if self.state == STATE_NAME_ENTRY:
            self._draw_name_entry()
        else:
            self._draw_table()
            self._draw_paddles_and_puck()
            self._draw_scoreboard()
            self._draw_camera_preview()

            if self.state == STATE_COUNTDOWN:
                self._draw_countdown()
            elif self.state == STATE_GOAL_PAUSE:
                self._draw_goal_banner()
            elif self.state == STATE_GAME_OVER:
                self._draw_game_over()
            elif self.state == STATE_LEADERBOARD:
                self._draw_leaderboard()

        present()

    def _draw_background(self):
        for x in range(0, WIDTH, 44):
            pygame.draw.line(SCREEN, COLOR_GRID, (x, 0), (x, HEIGHT), 1)
        for y in range(0, HEIGHT, 44):
            pygame.draw.line(SCREEN, COLOR_GRID, (0, y), (WIDTH, y), 1)

    def _draw_panel(self, rect, border_color=COLOR_PANEL_BORDER, radius=12):
        pygame.draw.rect(SCREEN, COLOR_PANEL, rect, border_radius=radius)
        pygame.draw.rect(SCREEN, border_color, rect, 2, border_radius=radius)

    def _draw_header(self):
        title = FONT_MED.render("UCC HAND TRACKING AIR HOCKEY", True, COLOR_ACCENT)
        SCREEN.blit(title, (WIDTH / 2 - title.get_width() / 2, TITLE_Y))
        pygame.draw.line(SCREEN, COLOR_ACCENT, (WIDTH / 2 - 190, _header_rule_y),
                         (WIDTH / 2 + 190, _header_rule_y), 2)

    def _draw_name_entry(self):
        self._draw_panel(pygame.Rect(WIDTH / 2 - 315, 95, 630, 475), COLOR_PANEL_BORDER, 18)
        prompt = FONT_BIG.render("Masukkan Nama Pemain", True, COLOR_TEXT)
        SCREEN.blit(prompt, (WIDTH / 2 - prompt.get_width() / 2, 140))

        for i, (label, value, color) in enumerate(
            [("Player 1 (kiri)", self.p1_name, COLOR_P1), ("Player 2 (kanan)", self.p2_name, COLOR_P2)]
        ):
            y = 260 + i * 110
            box_rect = pygame.Rect(WIDTH / 2 - 220, y, 440, 60)
            border = COLOR_ACCENT if self.active_field == i else COLOR_MUTED
            pygame.draw.rect(SCREEN, (20, 26, 40), box_rect, border_radius=10)
            pygame.draw.rect(SCREEN, border, box_rect, 3, border_radius=10)

            lab = FONT_SMALL.render(label, True, color)
            SCREEN.blit(lab, (box_rect.x, box_rect.y - 24))

            txt = FONT_MED.render(value or "", True, COLOR_TEXT)
            SCREEN.blit(txt, (box_rect.x + 14, box_rect.y + 15))

        hint = FONT_SMALL.render(
            "Ketik nama, tekan TAB/ENTER untuk pindah, ENTER lagi untuk mulai",
            True, COLOR_MUTED,
        )
        SCREEN.blit(hint, (WIDTH / 2 - hint.get_width() / 2, 500))

        if self.name_error:
            error = FONT_SMALL.render(self.name_error, True, (255, 140, 140))
            SCREEN.blit(error, (WIDTH / 2 - error.get_width() / 2, 540))

        hint2 = FONT_SMALL.render(
            "F11 = fullscreen  •  tarik pinggir window untuk resize  •  ESC = keluar",
            True, COLOR_MUTED,
        )
        SCREEN.blit(hint2, (WIDTH / 2 - hint2.get_width() / 2, 522))

        if not self.tracker.available:
            warn = FONT_SMALL.render(
                "⚠ Webcam tidak terdeteksi — cek izin kamera / device index.", True, (255, 140, 140)
            )
            SCREEN.blit(warn, (WIDTH / 2 - warn.get_width() / 2, 570))

    def _draw_table(self):
        table_rect = pygame.Rect(TABLE_LEFT, TABLE_TOP, TABLE_RIGHT - TABLE_LEFT, TABLE_BOTTOM - TABLE_TOP)
        pygame.draw.rect(SCREEN, COLOR_TABLE, table_rect, border_radius=14)
        pygame.draw.rect(SCREEN, COLOR_TABLE_LINE, table_rect, 3, border_radius=14)
        inner_rect = table_rect.inflate(-18, -18)
        pygame.draw.rect(SCREEN, (18, 27, 47), inner_rect, 1, border_radius=10)

        # center line + circle
        pygame.draw.line(SCREEN, COLOR_TABLE_LINE, (TABLE_MID_X, TABLE_TOP), (TABLE_MID_X, TABLE_BOTTOM), 2)
        pygame.draw.circle(SCREEN, COLOR_TABLE_LINE, (TABLE_MID_X, (TABLE_TOP + TABLE_BOTTOM) // 2), 60, 2)
        pygame.draw.circle(SCREEN, COLOR_TABLE_LINE, (TABLE_MID_X, (TABLE_TOP + TABLE_BOTTOM) // 2), 4)

        # goals
        mid_y = (TABLE_TOP + TABLE_BOTTOM) / 2
        pygame.draw.line(SCREEN, COLOR_GOAL, (TABLE_LEFT, mid_y - GOAL_HALF_HEIGHT), (TABLE_LEFT, mid_y + GOAL_HALF_HEIGHT), 6)
        pygame.draw.line(SCREEN, COLOR_GOAL, (TABLE_RIGHT, mid_y - GOAL_HALF_HEIGHT), (TABLE_RIGHT, mid_y + GOAL_HALF_HEIGHT), 6)

    def _draw_paddles_and_puck(self):
        self.paddle1.draw(SCREEN)
        self.paddle2.draw(SCREEN)
        self.puck.draw(SCREEN)

    def _draw_scoreboard(self):
        left_panel = pygame.Rect(TABLE_MID_X - 174, SCORE_Y - 9, 148, 128)
        right_panel = pygame.Rect(TABLE_MID_X + 26, SCORE_Y - 9, 148, 128)
        self._draw_panel(left_panel, COLOR_P1, 12)
        self._draw_panel(right_panel, COLOR_P2, 12)

        s1 = FONT_HUGE.render(str(self.score1), True, COLOR_P1)
        s2 = FONT_HUGE.render(str(self.score2), True, COLOR_P2)
        SCREEN.blit(s1, (TABLE_MID_X - 100 - s1.get_width() / 2, SCORE_Y))
        SCREEN.blit(s2, (TABLE_MID_X + 100 - s2.get_width() / 2, SCORE_Y))

        n1 = FONT_SMALL.render(self.p1_name, True, COLOR_P1)
        n2 = FONT_SMALL.render(self.p2_name, True, COLOR_P2)
        SCREEN.blit(n1, (TABLE_MID_X - 100 - n1.get_width() / 2, NAME_Y))
        SCREEN.blit(n2, (TABLE_MID_X + 100 - n2.get_width() / 2, NAME_Y))

        target = FONT_SMALL.render(f"First to {WIN_SCORE} wins", True, COLOR_MUTED)
        SCREEN.blit(target, (WIDTH / 2 - target.get_width() / 2, TARGET_Y))

    def _draw_camera_preview(self):
        w, h = PREVIEW_W, PREVIEW_H
        # Cache: cuma generate ulang surface preview kalau background thread sudah punya frame BARU (frame_id berubah). Kalau belum, dan game loop lagi render di 60fps sementara tracking cuma ~15-20fps, ini menghindari kerja ulang (resize+skeleton draw) yang sia-sia.
        current_id = self.tracker.frame_id
        if current_id != self._preview_frame_id:
            self._preview_surface = self.tracker.get_preview_surface(size=(w, h))
            self._preview_frame_id = current_id

        x, y = PREVIEW_X, PREVIEW_Y
        self._draw_panel(pygame.Rect(x - 8, y - 30, w + 16, h + 38), COLOR_PANEL_BORDER, 10)
        if self._preview_surface:
            SCREEN.blit(self._preview_surface, (x, y))
        pygame.draw.rect(SCREEN, COLOR_ACCENT, (x, y, w, h), 2, border_radius=6)
        label = FONT_SMALL.render("Camera (hand tracking)", True, COLOR_MUTED)
        SCREEN.blit(label, (x, PREVIEW_LABEL_Y))

    def _draw_countdown(self):
        elapsed = time.time() - self.countdown_start
        n = 3 - int(elapsed)
        text = "GO!" if n <= 0 else str(n)
        shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        shade.fill((5, 8, 16, 70))
        SCREEN.blit(shade, (0, 0))
        surf = FONT_HUGE.render(text, True, COLOR_ACCENT)
        SCREEN.blit(surf, (WIDTH / 2 - surf.get_width() / 2, HEIGHT / 2 - surf.get_height() / 2))

    def _draw_goal_banner(self):
        banner = pygame.Rect(WIDTH / 2 - 250, HEIGHT / 2 - 45, 500, 90)
        self._draw_panel(banner, COLOR_GOAL, 16)
        surf = FONT_BIG.render(f"GOAL! {self.last_scorer} scores", True, COLOR_GOAL)
        SCREEN.blit(surf, (WIDTH / 2 - surf.get_width() / 2, HEIGHT / 2 - 20))

    def _draw_game_over(self):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((5, 6, 10, 210))
        SCREEN.blit(overlay, (0, 0))
        self._draw_panel(pygame.Rect(WIDTH / 2 - 330, 105, 660, 390), COLOR_ACCENT, 18)

        winner = self.p1_name if self.score1 > self.score2 else self.p2_name
        title = FONT_HUGE.render("GAME OVER", True, COLOR_ACCENT)
        SCREEN.blit(title, (WIDTH / 2 - title.get_width() / 2, 150))

        sub = FONT_BIG.render(f"{winner} menang!  {self.score1} - {self.score2}", True, COLOR_TEXT)
        SCREEN.blit(sub, (WIDTH / 2 - sub.get_width() / 2, 250))

        hint1 = FONT_MED.render("ENTER — simpan skor ke leaderboard", True, COLOR_MUTED)
        hint2 = FONT_MED.render("R — main lagi tanpa simpan", True, COLOR_MUTED)
        SCREEN.blit(hint1, (WIDTH / 2 - hint1.get_width() / 2, 360))
        SCREEN.blit(hint2, (WIDTH / 2 - hint2.get_width() / 2, 405))

    def _draw_leaderboard(self):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((5, 6, 10, 230))
        SCREEN.blit(overlay, (0, 0))
        self._draw_panel(pygame.Rect(WIDTH / 2 - 330, 55, 660, 590), COLOR_ACCENT, 18)

        title = FONT_BIG.render("LIVE LEADERBOARD — TOP 10", True, COLOR_ACCENT)
        SCREEN.blit(title, (WIDTH / 2 - title.get_width() / 2, 90))

        entries = self.leaderboard.top(10)
        start_y = 170
        for i, e in enumerate(entries):
            rank = f"#{i + 1}"
            line = f"{rank:<4} {e['playerName']:<20} {e['score']}"
            color = COLOR_GOAL if i == 0 else COLOR_TEXT
            surf = FONT_MED.render(line, True, color)
            SCREEN.blit(surf, (WIDTH / 2 - 220, start_y + i * 36))

        hint = FONT_SMALL.render("ENTER / SPACE — main lagi", True, COLOR_MUTED)
        SCREEN.blit(hint, (WIDTH / 2 - hint.get_width() / 2, HEIGHT - 60))

    # ---------------- main loop ----------------
    def run(self):
        while True:
            for event in pygame.event.get():
                self.handle_event(event)
            self.update()
            self.draw()
            CLOCK.tick(FPS)

def main():
    Game().run()

if __name__ == "__main__":
    main()
