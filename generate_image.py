import cv2
import numpy as np
import random

# 1. Creăm o imagine de bază curată (un gradient cu un cerc și text)
# Dimensiune 400x400 pixeli
img = np.zeros((400, 400, 3), dtype=np.uint8)
for i in range(400):
    img[:, i] = (i//2, 100, 200 - i//2) # Un fundal colorat în gradient

cv2.circle(img, (200, 200), 100, (0, 200, 0), -1) # Un cerc verde
cv2.putText(img, "TEST RESTAURARE", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# 2. Adăugăm Zgomot "Sare și Piper" (afectează 5% din imagine)
noise_matrix = np.random.rand(400, 400)
img[noise_matrix < 0.025] = [0, 0, 0]       # Zgomot negru (piper)
img[noise_matrix > 0.975] = [255, 255, 255] # Zgomot alb (sare)

# 3. Adăugăm Zgârieturi Albe (pentru testul de Inpainting)
# Acestea vor fi detectate de masca ta (valori > 200)
cv2.line(img, (50, 50), (350, 350), (255, 255, 255), 4) # Zgârietură groasă
cv2.line(img, (300, 50), (100, 350), (240, 240, 240), 2) # Zgârietură subțire

# 4. Salvăm imaginea
cv2.imwrite("degradat.bmp", img)
print("Imaginea de test 'degradat.bmp' a fost creată cu succes în folderul curent!")