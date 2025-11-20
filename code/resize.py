import os
import glob
import cv2

raw_folder = '../images/raw'
resized_folder = '../images/cleaned'

i = 0

for img in glob.glob(os.path.join(raw_folder, '*.jpg')):
    img = cv2.imread(img)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(resized_folder, str(i) + '.jpg'), img)
    i += 1

