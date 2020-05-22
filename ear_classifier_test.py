from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import os

def evaluate_ears(imgdir
                  ,mode_path='logs/ear.h5'
                  ,isDetail=True):
    img_width, img_height = 100, 100
    files = os.listdir(imgdir)
    n= len(files)
    i = 0
    model = load_model(mode_path)
    labels = ['Non-wheat ear', '1 wheat ear', '2 wheat ears', '3 wheat ears']
    nonear,oneear,twoears,threeears=0,0,0,0
    for file in files:
        img_dir = os.path.join(imgdir, file)
        if isDetail:
            print(i, '/', n, ':', file)
        i = i + 1
        img=image.load_img(os.path.join(imgdir, file),target_size=(100,100))
        img=image.img_to_array(img)
        img=img.astype(float)/255.0
        img = np.expand_dims(img, axis=0)
        # img = preprocess_input(img)
        pred=model.predict(img)
        predIndex=pred.argmax()
        if isDetail:
            print('{}: {:.2f}%'.format(labels[predIndex],pred[0][predIndex]*100))
        if predIndex==0:
            nonear+=1
        elif predIndex==1:
            oneear+=1
        elif predIndex==2:
            twoears+=1
        elif predIndex==3:
            threeears+=1
        if isDetail:
            print('*'*40)
    print('Non-wheat ear: {}, 1 wheat ear: {}, 2 wheat ears: {}, 3 wheat ears: {}'.format(nonear,oneear,twoears,threeears))
    print('ears:',oneear+twoears*2+threeears*3)

if __name__=='__main__':
    evaluate_ears(r'data\test1\testcut\standard',isDetail=False)