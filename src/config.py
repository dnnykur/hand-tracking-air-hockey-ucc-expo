import os
import pygame
# ============================================================
# CONFIG
# ============================================================
# WIDTH/HEIGHT adalah kanvas "logis" tempat semua elemen game digambar.
# Ukurannya TETAP -- window fisiknya (WINDOW) boleh di-resize/fullscreen
# bebas oleh user, lalu kanvas ini di-scale otomatis supaya pas (letterbox),
# jadi tidak ada elemen yang gepeng atau proporsinya berubah.
TARGET_WIDTH, TARGET_HEIGHT = 1100, 700
MIN_WIN_WIDTH, MIN_WIN_HEIGHT = 480, 320  # batas bawah ukuran window fisik
WIDTH, HEIGHT = TARGET_WIDTH, TARGET_HEIGHT

CAM_INDEX = 0  # ganti sesuai /dev/videoN dari v4l2loopback (scrcpy webcam HP)
# Balik ke 640x480 -- sekarang tracking jalan di thread terpisah (lihat
# HandTracker di bawah), jadi resolusi besar gak lagi bikin GAME-nya nge-lag,
# tapi tetap bikin loop tracking-nya sendiri lebih lambat per-iterasi ->
# posisi tangan jadi kurang "fresh". Naikkan lagi ke 960x540 kalau CPU kamu
# kuat dan mau akurasi lebih (jarak jauh/redup) daripada kecepatan.
CAM_REQUEST_W, CAM_REQUEST_H = 640, 480
CAM_REQUEST_FPS = 30

# Model MediaPipe: 0 = lite (cepat, delay minim), 1 = full (lebih akurat
# dari jarak jauh, tapi tiap frame lebih lambat diproses -> posisi tangan
# yang sampai ke game jadi kurang up-to-date). Default balik ke 0 (lite)
# karena tracking sekarang jalan real-time di background thread -- makin
# cepat loop-nya, makin "nempel" paddle ngikutin tangan.
HAND_MODEL_COMPLEXITY = 0
HAND_MIN_DETECTION_CONFIDENCE = 0.5
HAND_MIN_TRACKING_CONFIDENCE = 0.4

SHOW_HAND_SKELETON = True     # gambar skeleton tangan di preview kamera

# Titik landmark MediaPipe yang dipakai buat kontrol paddle.
#   0  = pergelangan tangan (wrist)      -> perlu seluruh telapak kelihatan
#   8  = ujung jari telunjuk (index tip) -> kontrol pakai JARI SAJA, lebih presisi
# (Landmark lain: 4=ibu jari, 12=tengah, 16=manis, 20=kelingking)
CONTROL_LANDMARK = 8
ENHANCE_LOW_LIGHT = True      # CLAHE contrast boost, bantu deteksi di cahaya redup

# One-Euro-Filter (Casiez et al.) buat smoothing paddle: adaptif, jadi tetap
# halus saat tangan diam (mengurangi jitter) TAPI tetap responsif/rendah delay
# saat tangan gerak cepat -- lebih baik dari smoothing linier biasa yang
# selalu punya trade-off tetap antara halus vs lag.
PADDLE_FILTER_MIN_CUTOFF = 4.0   # dinaikkan dari 1.4 -> baseline jauh lebih responsif
PADDLE_FILTER_BETA = 0.06        # dinaikkan dari 0.02 -> makin "nempel" saat gerak cepat
PADDLE_FILTER_D_CUTOFF = 1.0

# Sensitivitas gerakan: >1.0 = paddle "digandakan" dari gerakan tangan asli,
# jadi gak perlu gerak tangan sejauh setengah frame kamera buat nyampe ujung
# meja -- gerakan tangan kecil aja udah cukup buat gerakin paddle jauh.
# 1.0 = 1:1 apa adanya. Naikkan lagi kalau masih kerasa kurang gesit.
PADDLE_SENSITIVITY = 1.8

TABLE_LEFT = 40
TABLE_RIGHT = WIDTH - 40
TABLE_BOTTOM = HEIGHT - 40
TABLE_MID_X = (TABLE_LEFT + TABLE_RIGHT) // 2
# TABLE_TOP dihitung otomatis di bawah, setelah font dimuat, supaya header
# (judul + skor + nama pemain) selalu punya cukup ruang dan tidak pernah
# "numpuk"/overflow ke area meja -- berapapun tinggi baris font di OS user.

GOAL_HALF_HEIGHT = 90  # goal opening = 2x this, centered vertically

PADDLE_RADIUS = 34
PUCK_RADIUS = 16
PUCK_FRICTION = 0.996        # sedikit diturunkan dari 0.998 biar gak makin ngebut abis-abisan
MAX_PUCK_SPEED = 26          # diturunin dari 32 -> tetap kenceng tapi lebih kekontrol
PUCK_RESTITUTION = 1.015     # dikurangi dari 1.03 -> tetap bouncy tapi gak numpuk energi cepet
PUCK_RESET_SPEED = 7.5       # sedikit diturunin dari 9 -> kickoff gak langsung ngebut banget
WIN_SCORE = 5
MAX_NAME_LEN = 20

LEADERBOARD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard.json")
ICON_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "UCC.png")
FPS = 60

COLOR_BG = (8, 10, 18)
COLOR_TABLE = (14, 20, 36)
COLOR_TABLE_LINE = (45, 70, 120)
COLOR_P1 = (60, 200, 255)
COLOR_P2 = (255, 90, 110)
COLOR_PUCK = (235, 240, 250)
COLOR_TEXT = (225, 232, 245)
COLOR_MUTED = (140, 150, 170)
COLOR_ACCENT = (120, 210, 255)
COLOR_GOAL = (255, 215, 90)

os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "centered")
pygame.init()
try:
	pygame.display.set_icon(pygame.image.load(ICON_FILE))
except (pygame.error, OSError):
	pass
pygame.display.set_caption("UCC Hand Tracking Air Hockey")

FONT_HUGE = pygame.font.SysFont("arial", 72, bold=True)
FONT_BIG = pygame.font.SysFont("arial", 40, bold=True)
FONT_MED = pygame.font.SysFont("arial", 26, bold=True)
FONT_SMALL = pygame.font.SysFont("arial", 18)

