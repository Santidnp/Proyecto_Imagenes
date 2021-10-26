from Procesamiento_imagenes import *
from datetime import datetime

imag1=cv2.imread('nado1.jpeg')

imagen = Reconocimiento(path=None,imagen=imag1).Transformacio()
cv2.imshow("Imagen_final",imagen)
cv2.waitKey(0)

