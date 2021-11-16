# MARCADORES VERDES
import cv2
import matplotlib.pyplot as plt  # para hacer gráficas
import numpy as np
import matplotlib.image as mpimg
import mediapipe as mp
import math
from PIL import Image, ImageDraw, ImageFont

# MARCADORES VERDES

imag1 = cv2.imread(r"C:\Users\sngh9\OneDrive\Escritorio\Maestria_Semestre_2\Procesamiento_de_imagenes\Proyecto\Nados_5.jpg")


plt.imshow(imag1)
plt.show()
imagen = cv2.cvtColor(imag1, cv2.COLOR_BGR2RGB)
print("imagen original")
plt.imshow(imagen)
plt.show()

imagenr = imagen
ancho = imagenr.shape[1]  # columnas
alto = imagenr.shape[0]  # filas

# Rotación de la imagen
M = cv2.getRotationMatrix2D((ancho // 2, alto // 2), 90, 1)
imagr = cv2.warpAffine(imagenr, M, (ancho, alto))
print("imagen")
plt.imshow(imagr)
plt.show()

#imag = imagr[0:800, 550:700]
#plt.imshow(imag)
#plt.show()
imag=imagr
colorbgr = cv2.cvtColor(imag, cv2.COLOR_RGB2BGR)  # cambiar espacio de color  a BGR
print(np.shape(colorbgr))  # tamaño de la imagen
bins = 25
plt.hist(colorbgr.ravel(), bins)
plt.show()

gris = cv2.cvtColor(colorbgr, cv2.COLOR_BGR2GRAY)
print("imagen gris")
plt.imshow(gris)
plt.show()

colorlab = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)  # cambiar de BGR  a  Lab
print("imagen HSV")
plt.imshow(colorlab)
plt.show()

colorf = colorlab[:, :,
         0]  # unicamente el componente 1 de el espacio de color y en x, y todos los valores (tamaño de la imagen)
plt.imshow(colorf)
plt.show()

colora = colorlab[:, :,
         1]  # unicamente el componente 1 de el espacio de color y en x, y todos los valores (tamaño de la imagen)
plt.imshow(colora)
plt.show()

colorb = colorlab[:, :,
         2]  # unicamente el componente 1 de el espacio de color y en x, y todos los valores (tamaño de la imagen)
plt.imshow(colorb)
plt.show()

# El espacio a diferencia verdes y rojo siendo verdes los valores mas bajos
plt.imshow(colorf, cmap='Greys')
plt.show()
bins = 25
plt.hist(colorf.ravel(), bins)
plt.show()

colorsize = colorf.shape  # Tamaño de la imagen
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

print("Histograma  de la imagen:")
plt.hist(valid.ravel(), 255, [0, 255])
plt.show()
print("imagen valida")
plt.imshow(valid)
plt.show()

# Crear una nueva variable del tamaño de la imagen original
imagenfinal = np.array(imag)  # np.zeros(np.shape(imag[0]),np.shape(imag[1]),np.shape(imag[2]))
print("imagen inicial")
plt.imshow(imag)
plt.show()

# Multiplico la mascara por la imagen original
imagenfinal[:, :, 0] = valid * imag[:, :, 0]
imagenfinal[:, :, 1] = valid * imag[:, :, 1]
imagenfinal[:, :, 2] = valid * imag[:, :, 2]

print("imagen final")
plt.imshow(imagenfinal)
plt.show()

####esqueletización###
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
with mp_pose.Pose(
    static_image_mode=True) as pose:
    image = colorbgr
    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    print("Pose landmarks:", results.pose_landmarks)
    if results.pose_landmarks is not None:
        print(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width))
        x1 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)
        y1 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)
        x2 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * width)
        y2 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * height)
        x3 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width)
        y3 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)
        x4 = int(results.pose_landmarks.landmark[11].x * width)
        y4 = int(results.pose_landmarks.landmark[11].y * height)
        x5 = int(results.pose_landmarks.landmark[13].x * width)
        y5 = int(results.pose_landmarks.landmark[13].y * height)
        x6 = int(results.pose_landmarks.landmark[15].x * width)
        y6 = int(results.pose_landmarks.landmark[15].y * height)
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.line(image, (x2, y2), (x3, y3), (255, 255, 255), 3)
        cv2.circle(image, (x1, y1), 6, (128, 0, 255), -1)
        cv2.circle(image, (x2, y2), 6, (128, 0, 255), -1)
        cv2.circle(image, (x3, y3), 6, (128, 0, 255), -1)
       # cv2.line(image, (x4, y4), (x5, y5), (255, 255, 255), 3)
       # cv2.line(image, (x5, y5), (x6, y6), (255, 255, 255), 3)
        #cv2.circle(image, (x4, y4), 6, (255, 191, 0), -1)
        #cv2.circle(image, (x5, y5), 6, (255, 191, 0), -1)
       # cv2.circle(image, (x6, y6), 6, (255, 191, 0), -1)
        '''
        mp_drawing.draw_landmarks(image, results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(128, 0, 250), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
        '''
    cv2.imshow("Imagen con angulo", image)
    cv2.waitKey(0)

    imagenr =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ancho = imagenr.shape[1]  # columnas
    alto = imagenr.shape[0]  # filas

    # Rotación de la imagen
    M = cv2.getRotationMatrix2D((ancho // 2, alto // 2), 270, 1)
    imagr = cv2.warpAffine(imagenr, M, (ancho, alto))
    print("imagen")
    plt.imshow(imagr)
    plt.show()

    hombro=(x3,y3)
    codo=(x2,y2)
    mano= (x1, y1)
    #calculo de angulo
    # Antebrazo respecto al brazo
    v4 = np.array(hombro) - np.array(codo)
    v5 = np.array(mano) - np.array(codo)

    angleb = np.math.atan2(np.linalg.det([v4, v5]), np.dot(v4, v5))
    dangleb = abs(np.degrees(angleb))

    print('ángulo del antebrazo: ')
    print(dangleb)
