import cv2

#camera = cv2.VideoCapture('GOPR7428.MP4')
camera = cv2.VideoCapture('Video_01.MP4')

camera.set(cv2.CAP_PROP_POS_FRAMES, 500)
#
# ret = True
# fps = camera.get(cv2.CAP_PROP_FPS)
# print(fps)
# while ret:
#     ret, image = camera.read()
#     if ret:
#         cv2.imshow("Image", image)
#         cv2.waitKey(int(1000 / fps))

i = 0
print(camera.isOpened())
while (camera.isOpened()) :
    ret, frame = camera.read()
    if ret == False:
        break

    if i == 10:
        break
    cv2.imwrite('Nados_' + str(i) + '.jpg', frame)
    i += 1

camera.release()
cv2.destroyAllWindows()