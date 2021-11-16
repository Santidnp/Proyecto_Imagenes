from Procesamiento_imagenes import *
from datetime import datetime
from Segmentos_Angulo import *

imag1=cv2.imread('Nados_7.jpg')

#imagen = Reconocimiento(path=None,imagen=imag1).Transformacio()
#cv2.imshow("Imagen_final",imagen)
#cv2.waitKey(0)

seg ,angulo = Segmentacion(path=None,imagen=imag1).segmentacion_angulo()

print(angulo)