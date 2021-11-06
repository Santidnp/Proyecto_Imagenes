
# coding: utf-8

# In[148]:


#MARCADORES VERDES 
import cv2 
import matplotlib.pyplot as plt  # para hacer gráficas
import numpy as np
import os
cwd = os.getcwd()
import matplotlib.image as mpimg
from IPython.display import display
import ipywidgets as widgets
from IPython.display import display

#MARCADORES VERDES 
#imag1=cv2.imread("‪C:/Users/Laura/Downloads/nado1.jpg")
imag1=cv2.imread(cwd+'/Nado_Lau.jfif')
#imag1=cv2.imread("C:/Users/Laura/Documents/Tesis/kang4.jpg")

plt.imshow(imag1)
plt.show()
imag= cv2.cvtColor(imag1, cv2.COLOR_BGR2RGB)
print("imagen original")
plt.imshow(imag)
plt.show()

colorbgr= cv2.cvtColor(imag, cv2.COLOR_RGB2BGR) #cambiar espacio de color  a BGR
#print(np.shape(colorhsv)) #tamaño de la imagen
bins=25
plt.hist(colorbgr.ravel(),bins)
plt.show()


gris= cv2.cvtColor(colorbgr, cv2.COLOR_BGR2GRAY)
print("imagen gris")
plt.imshow(gris)
plt.show() 

colorlab= cv2.cvtColor(imag, cv2.COLOR_BGR2HSV) # cambiar de BGR  a  Lab
print("imagen HSV")
plt.imshow(colorlab) 
plt.show()


colorf=colorlab[:,:,0] # unicamente el componente 1 de el espacio de color y en x, y todos los valores (tamaño de la imagen)
plt.imshow(colorf)
plt.show()

colora=colorlab[:,:,1] # unicamente el componente 1 de el espacio de color y en x, y todos los valores (tamaño de la imagen)
plt.imshow(colora)
plt.show()

colorb=colorlab[:,:,2] # unicamente el componente 1 de el espacio de color y en x, y todos los valores (tamaño de la imagen)
plt.imshow(colorb)
plt.show()

#El espacio a diferencia verdes y rojo siendo verdes los valores mas bajos 
plt.imshow(colorf,cmap='Greys')
plt.show()
bins=25
plt.hist(colorf.ravel(),bins)
plt.show()


colorsize=colorf.shape # Tamaño de la imagen 
color=list(colorsize) #crear matriz 
valid=np.zeros((color[0],color[1])) #creon una matriz de ceros del tamaño de la imagen
a=np.zeros((color[0],color[1]))

#Guardar la imagen como binaria (Binarizacion OTSU)
for  x in range(color[0]):# recorrer en el eje x
    for y in range(color[1]):# recorrer en el eje y para finalmente recorrer pixel por pixel
        
        if colorf[x,y]<180 and colorf[x,y]>100 :#para que se quede con los tonos blancos de la imagen
            valid[x,y]=1
            
        else:
            valid[x,y]=0
        
print("Histograma  de la imagen:" )           
plt.hist(valid.ravel(),255,[0,255])
plt.show()
print("imagen valida")
plt.imshow(valid)
plt.show() 

#Crear una nueva variable del tamaño de la imagen original
imagenfinal=np.array(imag)#np.zeros(np.shape(imag[0]),np.shape(imag[1]),np.shape(imag[2]))
print("imagen inicial")
plt.imshow(imag)
plt.show()


#Multiplico la mascara por la imagen original 
imagenfinal[:,:,0]=valid*imag[:,:,0]
imagenfinal[:,:,1]=valid*imag[:,:,1]
imagenfinal[:,:,2]=valid*imag[:,:,2]

print("imagen final")
plt.imshow(imagenfinal)
plt.show() 

imagbgr=cv2.cvtColor(imagenfinal, cv2.COLOR_RGB2BGR)
gris= cv2.cvtColor(imagbgr, cv2.COLOR_BGR2GRAY)
print("imagen gris")
plt.imshow(gris)
plt.show() 

    
#Función para encontrar contornos 
#ret, thresh = cv2.threshold(gris, 0,255,0) # El tercer argumento es el valor maxVal


ret,thresh = cv2.threshold(gris,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#que representa el valor que debe darse si el valor de píxel es mayor que (a veces menor que) el valor de umbral.
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Dibujar los contornos en la imagen 
print("Se detectaron: {0} contornos".format(len(contours))) 
img_contours = imagenfinal.copy()
img_contours = cv2.drawContours(img_contours, contours, -1, (255, 0, 0), 3)
plt.imshow(img_contours)

kernel = np.ones((7,7),np.uint8)
dilatacion = cv2.dilate(gris,kernel,iterations = 1)
print("imagen diltada:")
plt.imshow(dilatacion)
plt.show() 

