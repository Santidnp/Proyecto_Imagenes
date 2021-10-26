from cv2 import VideoWriter,VideoWriter_fourcc
import cv2
import os
from PIL import Image
def reconstruir1():
    path1 = r'C:\Users\sngh9\OneDrive\Escritorio\Maestria_Semestre_2\Procesamiento_de_imagenes\Proyecto\Frames'
    width = 480
    height = 848
    FPS = 30
    seconds = 18
    nombre = r'C:\Users\sngh9\OneDrive\Escritorio\Maestria_Semestre_2\Procesamiento_de_imagenes\Proyecto'
    gv1 = (nombre + './recof.avi')

    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(gv1, fourcc, float(FPS), (width, height))
    listing = os.listdir(path1)
    print(listing)

    for _ in range(FPS * seconds):
        for file in listing:
            #print(path1 + '/' + file)
            #im = Image.open(path1 + '/' + file)
            im2 = cv2.imread(path1 + '/' + file)
            fram = im2
            #print(type(im2))
            video.write(fram)
    video.release()

    ref = cv2.VideoCapture('recof.avi')

    for y in range(350):
        ret, frame = ref.read()
        #print(type(frame))

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ref.release()
    cv2.destroyAllWindows()


reconstruir1()