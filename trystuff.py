import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()
plt.axis([0, 1000, 0, 1])

i = 0
x = list()
y = list()

while i < 1000:
    temp_y = np.random.random()
    x.append(i)
    y.append(temp_y)
    plt.scatter(i, temp_y)
    i += 1
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

plt.axis([0, 10, 0, 1])

for i in range(10):
    y = np.random.random()
    plt.scatter(i, y)
    plt.pause(0.05)

plt.show()


import random
import matplotlib.pyplot as plt

dat=[0,1]
fig = plt.figure()
ax = fig.add_subplot(111)
Ln, = ax.plot(dat)
ax.set_xlim([0,20])
plt.ion()
plt.show()
for i in range (18):
    dat.append(random.uniform(0,1))
    Ln.set_ydata(dat)
    Ln.set_xdata(range(len(dat)))
    plt.pause(1)

    print('done with loop')

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while (1):
    # read frame
    ret, frame = cap.read()

    # convert BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])

    # Threshold the HSV to get only the blue colors
    mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # Bitwise-AND the mask and the original image
    res = cv2.bitwise_and(frame, frame, mask=mask_blue)

    # Display
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask_blue)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()



import cv2
import numpy as np

img = cv2.imread('ratimage.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,gray = cv2.threshold(gray,127,255,0)
gray2 = gray.copy()
mask = np.zeros(gray.shape,np.uint8)

contours, hier = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    #print(cv2.contourArea(cnt))
    if 5000<cv2.contourArea(cnt)<15000:
        print(cv2.contourArea(cnt))
        cv2.drawContours(img,[cnt],0,(0,255,0),2)
        cv2.imshow("img", img)
        cv2.drawContours(mask,[cnt],0,255,-1)
        cv2.imshow("mask", mask)

select_cnt = [cnt for cnt in contours if 5000 < cv2.contourArea(cnt) < 15000]
if len(select_cnt) > 2:
    print('Problem! Too many white area detected')
else:
    select_cnt = select_cnt[0][0]

x1, y1 = select_cnt[0][0], select_cnt[0][1] - 200
x2, y2 = select_cnt[0][0] + 100, select_cnt[0][1]

img = cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 3)
#img = cv2.line(img, (383,409), (511,409), (0,0,255), 5)
cv2.imshow('i', img)




# draw the figure so the animations will work
fig = plt.gcf()
fig.show()
fig.canvas.draw()

while True:
    # compute something
    plt.plot([1], [2])  # plot something

    # update canvas immediately
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    # plt.pause(0.01)  # I ain't needed!!!
    fig.canvas.draw()

plt.close('all')



f = open("sc08_083115.txt", "r")
d = f.read()
print(d)