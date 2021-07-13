import cv2
file1 = open('imposter_train_raw.txt', 'r')
Lines = file1.readlines()
count=0
for line in Lines:
    count+=1
    print(line)
    img = cv2.imread("D:/Liveness Detection/dataset/NUAA/ImposterRaw/{}".format(line.strip()))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imwrite("D:/Liveness Detection/Code/NUAA/train/spoof_hsv/{}.jpg".format(count), hsv)