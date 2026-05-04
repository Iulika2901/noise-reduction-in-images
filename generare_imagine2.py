import cv2
import numpy as np
import random

def create_base_image(text="TEST", bg_color=(100, 150, 200)):
    # Creăm o imagine de bază clară cu forme geometrice și text
    img = np.full((400, 400, 3), bg_color, dtype=np.uint8)
    cv2.circle(img, (200, 200), 80, (0, 0, 255), -1)
    cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), -1)
    cv2.putText(img, text, (60, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return img

def add_salt_and_pepper(img, prob=0.03):
    noisy = np.copy(img)
    matrix = np.random.rand(img.shape[0], img.shape[1])
    noisy[matrix < (prob / 2)] = [0, 0, 0]          # Piper (Negru)
    noisy[matrix > 1 - (prob / 2)] = [255, 255, 255] # Sare (Alb)
    return noisy

def add_gaussian_noise(img, mean=0, std=25):
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = cv2.add(img.astype(np.float32), noise)
    return np.clip(noisy, 0, 255).astype(np.uint8)

# --- Generarea celor 6 imagini ---

# 1. Pentru Filtru Median Global (Zgomot Sare și Piper standard)
img1 = create_base_image("1. MEDIAN", (50, 50, 50))
img1_noisy = add_salt_and_pepper(img1, 0.05)
cv2.imwrite("test_1_median.bmp", img1_noisy)

# 2. Pentru Filtru de Mediere (Zgomot Gaussian / granulație fină)
img2 = create_base_image("2. MEDIERE", (80, 120, 80))
img2_noisy = add_gaussian_noise(img2, std=40)
cv2.imwrite("test_2_mediere.bmp", img2_noisy)

# 3. Pentru Filtru Gaussian (Netezire, păstrând marginile)
# Folosim aceeași imagine ca la 2 pentru a compara medierea cu gaussianul
cv2.imwrite("test_3_gaussian.bmp", img2_noisy)

# 4. Pentru Filtru de Minim (Puncte albe pe fundal negru)
img4 = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.putText(img4, "4. MINIM", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
# Adăugăm doar puncte albe ("sare")
matrix = np.random.rand(400, 400)
img4[matrix > 0.98] = [255, 255, 255]
cv2.imwrite("test_4_minim.bmp", img4)

# 5. Pentru Filtru Median Selectiv (Zgomot pe text subțire)
img5 = create_base_image("5. SELECTIV", (100, 50, 100))
# Adăugăm un text suplimentar mai subțire pentru a demonstra că nu e blurat
cv2.putText(img5, "Text fin intact", (80, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1)
img5_noisy = add_salt_and_pepper(img5, 0.04)
cv2.imwrite("test_5_selectiv.bmp", img5_noisy)

# 6. Pentru Inpainting (Zgârieturi albe vizibile)
img6 = create_base_image("6. INPAINT", (60, 100, 150))
cv2.line(img6, (40, 40), (360, 360), (255, 255, 255), 3)  # Zgârietură lungă
cv2.line(img6, (200, 50), (100, 300), (245, 245, 245), 2) # Zgârietură scurtă
cv2.line(img6, (300, 100), (350, 200), (250, 250, 250), 4) # Zgârietură groasă
cv2.imwrite("test_6_inpaint.bmp", img6)

print("Toate cele 6 imagini de test au fost generate cu succes!")