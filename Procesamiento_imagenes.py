import cv2
import numpy as np


class Reconocimiento:

    def __init__(self,path=None,imagen=None):
        if path is not None:
            self.img = cv2.imread(path)
            assert isinstance(self.img,(np.ndarray)), "La imagen no se cargo"
        else:
            self.img = imagen
            assert isinstance(self.img, (np.ndarray)), "La imagen no se cargo"


    def Transformacio(self):
        imag = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        colorbgr= cv2.cvtColor(imag,cv2.COLOR_RGB2BGR)
        gris = cv2.cvtColor(colorbgr, cv2.COLOR_BGR2GRAY)
        colorlab = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)
        colorf= colorlab[:,:,0]
        colora = colorlab[:, :, 1]
        colorb = colorlab[:, :, 2]
        colorsize = colorf.shape
        color = list(colorsize)  # crear matriz
        valid = np.zeros((color[0], color[1]))  # creon una matriz de ceros del tamaño de la imagen
        a = np.zeros((color[0], color[1]))
        # Guardar la imagen como binaria (Binarizacion OTSU)
        for x in range(color[0]):  # recorrer en el eje x
            for y in range(color[1]):  # recorrer en el eje y para finalmente recorrer pixel por pixel

                if colorf[x, y] < 180 and colorf[x, y] > 100:  # para que se quede con los tonos blancos de la imagen
                    valid[x, y] = 1

                else:
                    valid[x, y] = 0
        imagenfinal = np.array(imag)

        # Multiplico la mascara por la imagen original
        imagenfinal[:, :, 0] = valid * imag[:, :, 0]
        imagenfinal[:, :, 1] = valid * imag[:, :, 1]
        imagenfinal[:, :, 2] = valid * imag[:, :, 2]
        #cv2.imshow("Imagen_final",imagenfinal)
        #cv2.waitKey(0)
        return imagenfinal
