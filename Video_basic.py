import cv2
import sys
import os
from Procesamiento_imagenes import *
from Segmentos_Angulo import *

if __name__ == '__main__':
    path = sys.argv[1]
    video_name = sys.argv[2]
    path_file = os.path.join(path, video_name)

    # load video
    camera = cv2.VideoCapture(path_file)
    #camera.set(cv2.CAP_PROP_POS_FRAMES, 500)

    # properties
    # n_frames = camera.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(n_frames)
    # fps = camera.get(cv2.CAP_PROP_FPS)
    # print(fps)
    # width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # pos = camera.get(cv2.CAP_PROP_POS_FRAMES)
    # bitrate = camera.get(cv2.CAP_PROP_BITRATE)
    carp_videos = r'C:\Users\sngh9\OneDrive\Escritorio\Maestria_Semestre_2\Procesamiento_de_imagenes\Proyecto\Frames'
    # visualization
    ret = True
    i = 0
    Angulos = []
    while ret:
        ret, image = camera.read()
        #cv2.imshow("Image", image)
        #imagen = Reconocimiento(path=None, imagen=image).Transformacio()
        try:
            seg, angulo = Segmentacion(path=None, imagen=image).segmentacion_angulo()
            cv2.imwrite(carp_videos + '/Frame' +str(i)+'.jpg', seg)
            Angulos.append(angulo)
        except UnboundLocalError:
            print('seguir')


        i +=1

        #if ret:
            #cv2.imshow("Image", image)
            #cv2.waitKey(int(1000 / fps))
            #if cv2.waitKey(25) & 0xFF == ord('q'):
                #break

