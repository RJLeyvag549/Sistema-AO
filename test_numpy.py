import numpy as np

height, width = 1080, 1920
radius_pixels = height // 2
y, x = np.ogrid[-height//2 : height//2, -width//2 : width//2]
r2 = (x**2 + y**2) / (radius_pixels**2)
pupil_mask = r2 <= 1.0

fase_continua = np.zeros((height, width), dtype=np.float32)
piston_val = 0.0
fase_continua[pupil_mask] += piston_val * np.pi

fase_envuelta = np.mod(fase_continua, 2 * np.pi)

fase_slm_8bit = np.full((height, width), 255, dtype=np.uint8)
fase_slm_8bit[pupil_mask] = 255 - (fase_envuelta[pupil_mask] / (2 * np.pi) * 255).astype(np.uint8)

print("Background value:", fase_slm_8bit[0, 0])
# Get a pixel inside the pupil
cy, cx = height // 2, width // 2
print("Pupil value:", fase_slm_8bit[cy, cx])
