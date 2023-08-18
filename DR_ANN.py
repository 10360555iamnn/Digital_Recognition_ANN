import streamlit as st
import pandas as pd
import numpy as np
#import cv2
from PIL import Image

import tensorflow as tf #機器學習的核心
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops import losses


#with st.spinner('數據訓練中...'):


#上傳圖片
upload_file=st.file_uploader("請上傳一張照片",type=['jpg','png'])

#照片預處理
if upload_file is not None:
    #cv2版本沒搞懂
    #image = np.array(bytearray(upload_file.read()),dtype='uint8')
    #image = cv2.imdecode(image,cv2.IMREAD_COLOR)#字節解碼
    #RPG_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)#圖片轉檔
    #image_resize = cv2.resize(RPG_image,(28,28))
    
    img = Image.open(upload_file).convert('L')
    img = img.resize((28, 28))

    arr = []

    for i in range(28):
        for j in range(28):
            # mnist 里的颜色是0代表白色（背景），1.0代表黑色
            pixel = 1.0 - float(img.getpixel((j, i)))/255.0
            arr.append(pixel)
    arr1 = np.array(arr).reshape((1, 28, 28))
    
    #顯示圖片
    st.markdown("### 用戶上傳圖片，顯示如下")
    st.image(upload_file,channels="RGB")
    st.markdown("**點擊按鈕開始預測**")
    predict = st.button("數字預測")
    if predict :
        (x_train,y_train),(x_test,t_test)=tf.keras.datasets.mnist.load_data()

        x_train=x_train/255 #將0～255縮放到只有0～1為了方便機器學習
        x_test=x_test/255

        ANN = keras.Sequential(name="ClassificationANN")#建造網路架構
        ANN.add(layers.Flatten(input_shape=(28,28)))#吃照片(壓平)
        ANN.add(layers.Dense(128,activation='relu'))#第一層128節點,ReLu激活函數
        ANN.add(layers.Dense(64,activation='relu'))#第二層64節點,ReLu激活函數
        ANN.add(layers.Dense(32,activation='relu'))#第二層32節點,ReLu激活函數
        ANN.add(layers.Dense(10,activation='softmax'))#第三層10節點,softmax激活函數
        #keras.utils.plot_model(ANN,show_shapes=True)#印出ANN的樣子

        ANN.compile(optimizer='adam',
            loss=keras.losses.sparse_categorical_crossentropy)#設定如何更新參數,損失函數是誰

        ANN.fit(x_train,y_train,epochs=10)#訓練

        prediction = ANN.predict(arr1)
        max_prediction = prediction.argmax()
        st.title("照片的數字為:{}".format(max_prediction ))
