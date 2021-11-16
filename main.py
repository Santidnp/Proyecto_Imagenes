from Procesamiento_imagenes import *
from datetime import datetime
from Segmentos_Angulo import *
import sys
import os
#imag1=cv2.imread('Nado_1.jpg')

#imagen = Reconocimiento(path=None,imagen=imag1).Transformacio()
#cv2.imshow("Imagen_final",imagen)
#cv2.waitKey(0)

#seg ,angulo = Segmentacion(path=None,imagen=imag1).segmentacion_angulo()

"""
python path_image image
"""

if __name__ == '__main__':
    path = sys.argv[1]
    img = sys.argv[2]
    imagen = os.path.join(path, img)
    print(imagen)

    seg, angulo = Segmentacion(path=imagen, imagen=None).segmentacion_angulo()
    cv2.imshow("Imagen con angulo", seg)
    cv2.waitKey(0)
    print(angulo)