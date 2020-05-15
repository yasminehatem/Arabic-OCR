from untitledCharacter import *

from commonfunctions import *
import numpy as np
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import binary_erosion, binary_dilation, binary_closing,skeletonize, thin
from skimage.measure import find_contours
from skimage.draw import rectangle
from skimage.filters import threshold_otsu
from skimage.viewer import ImageViewer
from skimage.filters import gaussian
from statistics import mode
from sklearn.neural_network import MLPClassifier
import pickle
import time
import matplotlib.pyplot as plt
import skimage.morphology
from skimage import img_as_ubyte
from skimage import feature
from scipy import stats
import math
from skimage.morphology import skeletonize
import os, sys
import xlrd
import csv
from scipy import ndimage
import xlwt 
from xlwt import Workbook 
from PIL import Image
from sklearn.metrics import accuracy_score

def checkdotes2(letter):
    dim=(28,28)
    img = cv2.resize(letter, dim, interpolation = cv2.INTER_AREA)
    labeled,nr_objects=ndimage.label(letter>0)

    
    #plt.imshow(labeled)
  
    
    return nr_objects-1 ,labeled
def count_holes(letter,c):
   
    c,labeled=checkdotes2(letter)
    contours,hierarchy=cv2.findContours(letter,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)-c-1
def dotes(img):
    count_con,labeled=checkdotes2(img)      
    c2=count_holes(img,count_con)
    dotnum=0
    dotplace=0
    mylist=labeled.flatten()

    sumones=(mylist==1).sum()
    sumtwos=(mylist==2).sum()

    if(count_con==1):


        if(sumones< sumtwos):
            #dot fo2 di kda feat
            dotplace=1
            if(sumones<=30):
                dotnum=1
            elif(sumones>=31 and sumones <=72):
                dotnum=2
            else:
                
                dotnum=3
            #tab lw dot fo2 3yza a3rf 1 or 2 or 3 dots
        else:
            #dot ta7t aw fel nos zay el gim el fl a5er di feat tania , yeh
            dotplace=2
            if(sumtwos<=30):
                dotnum=1
            elif(sumtwos>=31 and sumtwos <=72):
                dotnum=2
            else:

                dotnum=3
    if(count_con==2):
        #7arf yeh
        dotnum=2

        if(sumones< sumtwos):
                #dot fo2 di kda feat
            dotplace=1


        else:
                #dot ta7t aw fel nos zay el gim el fl a5er di feat tania , yeh
            dotplace=2
    return dotplace,dotnum
def getwidth(letter): #return width of a letter can be less accurte if the letter is segmented wrong
    dim=(28,28)
    letter = cv2.resize(letter, dim, interpolation = cv2.INTER_AREA)
    start=letter.shape[1]
    #plt.imshow(letter)
    for i in range(letter.shape[1]):
        
        
        for j in range(letter.shape[0]):
            
           
            if letter[j][i]==1 and i<=start:
                start=i
               
                
    k=letter.shape[1]-1
   
    end=0
    while(k>=0):
       
        for m in range(letter.shape[0]):
            
            if letter[m][k]==1 and k >= end:
                end=k    
        k=k-1
    return end-start 
#### only call get4areafit it returns 4 numbers 
def getsinglearea(img):
    if cv2.countNonZero(img) == 0:
        return 0
    c=0
    for i in range(img.shape[0]):
        #print("hena")
        for j in range(img.shape[1]):
            if img[i][j]!=0:
                c=c+1
    return c

def get4areas(res):
    shapeImage=res.shape
    firstQ=res[0:int(shapeImage[0]/2),0:int(shapeImage[1]/2)]
    thirdQ=res[0:int(shapeImage[0]/2),int(shapeImage[1]/2):int(shapeImage[1])]
    secondQ=res[int(shapeImage[0]/2):int(shapeImage[0]+1),0:int(shapeImage[1]/2)+1]
    fourthQ=res[int(shapeImage[0]/2):int(shapeImage[0])+1,int(shapeImage[1]/2):int(shapeImage[1])+1]
    show_images([res])
    #print(res)
    #print(firstQ)
    return getsinglearea(firstQ),getsinglearea(thirdQ),getsinglearea(thirdQ),getsinglearea(fourthQ)



def fitimg(letter):
    i=0

    while((np.sum(letter[i,:]))==0 and i< letter.shape[0]):
        i=i+1
    rowtop=i

    i=letter.shape[0]-1


    while((np.sum(letter[i,:]))==0 and i>=0):
        i=i-1
    rowbottom=i

    i=0
    while((np.sum(letter[:,i]))==0 and i< letter.shape[1]):
        i=i+1
    colleft=i

    print("7araaaaaaaaaam")
    i=letter.shape[1]-1     
    while((np.sum(letter[:,i]))==0 and i>=0):
        i=i-1
    colright=i


    mycut=np.copy(letter)  
    mycut=mycut[rowtop:rowbottom+1,colleft:colright+1]
    #letter fits perfectly with no space
    dim=(28,28)
    #resize to fairly compare
    mycut = cv2.resize(mycut, dim, interpolation = cv2.INTER_AREA)
    print(mycut)  
    return mycut

def get4areafit(img):
    
    img=fitimg(img)
    return get4areas(img)
##call this
def get4areafitedg(letter): #bt5od binary 3ady na bs mksla aghyr l esm btrg3 4 arkam
    gray=np.copy(letter)
    gray=fitimg(gray)
    #print(gray)
    c,_=checkdotes2(gray)

    m=count_holes(gray,c)
    if m>=1:
        des = cv2.bitwise_not(gray)

        #print(des)
        #plt.imshow(des)

        contour,hier = cv2.findContours(gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contour:
            cv2.drawContours(des,[cnt],0,1,-1)

        gray = cv2.bitwise_not(des)
        #print("gray")
        #print(gray.shape)
        #print(gray)


        #plt.imshow(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        res = cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel)
    else:
        res=np.copy(gray)
    show_images([res])
    img=np.copy(res)
    img=sobel(img)
    show_images([img])
    return get4areas(img)
### 2 returns first is the dot place(0 no dot , 1 dot is above bl , 2 dot below base line) second number of dots 
#works only with 28*28 , resizing already done in func check dotes 2
def dotes(img):
    count_con,labeled=checkdotes2(img)     
    print(count_con) 
    c2=count_holes(img,count_con)
    print(c2)
    #print(labeled)

    print(labeled)
    ####for no dots###
    dotnum=0
    dotplace=0
    mylist=labeled.flatten()

    sumones=(mylist==1).sum()
    sumtwos=(mylist==2).sum()
    print("some 1")
    print(sumones)
    print("sum 2")
    print(sumtwos)
    print("count con")
    print(count_con)
    if(count_con==1):


        if(sumones< sumtwos):
            #dot fo2 di kda feat
            dotplace=1
            if(sumones<=30):
                print("leh")
                dotnum=1
            elif(sumones>=31 and sumones <=72):
                dotnum=2
            else:
                
                dotnum=3
            #tab lw dot fo2 3yza a3rf 1 or 2 or 3 dots

        else:
            #dot ta7t aw fel nos zay el gim el fl a5er di feat tania , yeh
            dotplace=2
            if(sumtwos<=30):
                dotnum=1
            elif(sumtwos>=31 and sumtwos <=72):
                dotnum=2
            else:

                dotnum=3
    if(count_con==2):
        #7arf yeh
        dotnum=2

        if(sumones< sumtwos):
                #dot fo2 di kda feat
            dotplace=1


        else:
                #dot ta7t aw fel nos zay el gim el fl a5er di feat tania , yeh
            dotplace=2
    print("dot num")
    print(dotnum)
    return dotplace,dotnum

 #call this
def getheight(letter):   
    (h, w) = letter.shape[:2]
    center = (w / 2, h / 2)
    scale = 1.0
    M = cv2.getRotationMatrix2D(center,90, scale)
    rotated90 = cv2.warpAffine(letter, M, (h, w))
    plt.imshow(rotated90)
     
    return getwidth(rotated90)
with open('training.csv','r') as file:
    reader=csv.reader(file)
    rows=list(reader)
    numrows=len(rows)
print(numrows)
# For row 0 and column 0 
Featurevector=[]
mylabels=[]
for row in range(numrows):

    image = (rgb2gray(io.imread(rows[row][0])))

    #resized=cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
    w = 28 - image.shape[1]
    h = 28 - image.shape[0]

    if(w>=0 and h>=0):
        new = cv2.copyMakeBorder( image, 0, h, 0, w, cv2.BORDER_CONSTANT,value=0)
        ThreshImage=new/255
        x1=getwidth(ThreshImage)
        x2,_=checkdotes2(ThreshImage)
        x3=getheight(ThreshImage)
#        x4=dotes(ThreshImage)
        MyFeatures=[]
        MyFeatures.append(x1)
        MyFeatures.append(x2)
        MyFeatures.append(x3)
#        MyFeatures.append(x4)
        for i in range(ThreshImage.shape[0]):
            for j in range(ThreshImage.shape[1]):
                MyFeatures.append(ThreshImage[i][j])
        Featurevector.append(MyFeatures)
        mylabels.append(rows[row][1])

def Train(X,Y):
    clf = MLPClassifier(solver='adam',alpha=0.0001,hidden_layer_sizes=(50, 40))
    print("Started Training")
    ts = time.perf_counter() * 1000
    X_np=np.array(X)
    y_np=np.array(Y)
    clf.fit(X,Y)
    
   
    te=time.perf_counter()*1000
    print("Ended Training in:",te-ts,"ms")
    return clf
#
#classifier=Train(Featurevector,mylabels)
#Clas = open('Classifier', 'wb')
#pickle.dump(classifier, Clas)
#Clas.close()

a=open('Classifier','rb')
c=pickle.load(a)
#print(c.predict(f))
score = accuracy_score(mylabels,c.predict(Featurevector))
print(score*100)
for scanned in os.listdir('test/scanned/'):
        Path = 'test/scanned/'+scanned
        image = (rgb2gray(io.imread(Path)))
        thresh=threshold_otsu(image)
        #thresh=127
        image[image<thresh]=0
        image[image>=thresh]=1
        finalImage=rotate(image)
        List=segment_Lines(finalImage)
        #show_images([image,rotated])

        Lines=[] 
        dilatedlines=[]  
        kernel1 = np.ones((4,4), np.uint8)
        kernel2 = np.ones((2,2), np.uint8)  
        for i in range(0,len(List),2): #start,stop,step
            line=finalImage[int(List[i]):int(List[i+1]),:]
            Lines.append(line)
            dilated=binary_erosion(line,kernel1)
            dilated=binary_dilation(dilated,kernel2)
            dilatedlines.append(dilated)
        threshimage=Lines[0]
        words= wordSegmentation(dilatedlines,Lines)
        wordPosEachLine=wordsPostions(dilatedlines,Lines)

        BI = BaselineIndex(Lines )


        MaxTransitionIndex =MaxTrans(Lines, BI)

        SeparationRegions=CutPointIdentification(Lines,words,MaxTransitionIndex)
        SeparationRegions= CutPointIdentificationFilteration(words,BI,MaxTransitionIndex,SeparationRegions,Lines)
        arraychar=getCharcter(words,SeparationRegions,wordPosEachLine)
        
        f= open("test/text/"+scanned[:-4]+".txt","a+",encoding='utf-8')
        for x in range(len(arraychar)):
            for y in range(len(arraychar[x])-1,-1,-1):
                for z in range(len(arraychar[x][y])-1,-1,-1):
                    
                    w = 28 - arraychar[x][y][z].shape[1]
                    h = 28 - arraychar[x][y][z].shape[0]
                    new = cv2.copyMakeBorder( arraychar[x][y][z], 0, h, 0, w, cv2.BORDER_CONSTANT,value=0)
                    ThreshImage=new
                    x1=getwidth(ThreshImage)
                    x2,_=checkdotes2(ThreshImage)
                    x3=getheight(ThreshImage)
#                    x4=dotes(ThreshImage)
                    MyFeatures=[]
                    MyFeatures.append(x1)
                    MyFeatures.append(x2)
                    MyFeatures.append(x3)
#                    MyFeatures.append(x4)
                    Featurevector=[]
                    for i in range(ThreshImage.shape[0]):
                        for j in range(ThreshImage.shape[1]):
                            MyFeatures.append(ThreshImage[i][j])
                    Featurevector.append(MyFeatures)
#                    print(Featurevector)
                    label=c.predict(Featurevector)
                    
                    label=int(label[0])
#                    print(label)
                    if(label==0):
                        f.write('ل')
                        f.write('ا')
                    elif(label==1):
                        f.write('ا')
                    elif(label==2):
                        f.write('ب')
                    elif(label==3):
                        f.write('ت')
                    elif(label==4):
                        f.write('ث')
                    elif(label==5):
                        f.write('ج')
                    elif(label==6):
                        f.write('ح')
                    elif(label==7):
                        f.write('خ')
                    elif(label==8):
                        f.write('د')
                    elif(label==9):
                        f.write('ذ')
                    elif(label==10):
                        f.write('ر')
                    elif(label==11):
                        f.write('ز')
                    elif(label==12):
                        f.write('س')
                    elif(label==13):
                        f.write('ش')
                    elif(label==14):
                        f.write('ص')
                    elif(label==15):
                        f.write('ض')
                    elif(label==16):
                        f.write('ط')
                    elif(label==17):
                        f.write('ظ')
                    elif(label==18):
                        f.write('ع')
                    elif(label==19):
                        f.write('غ')
                    elif(label==20):
                        f.write('ف')
                    elif(label==21):
                        f.write('ق')
                    elif(label==22):
                        f.write('ك')
                    elif(label==23):
                        f.write('ل')
                    elif(label==24):
                        f.write('م')
                    elif(label==25):
                        f.write('ن')
                    elif(label==26):
                        f.write('ه')
                    elif(label==27):
                        f.write('و')
                    elif(label==28):
                        f.write('ي')
                f.write(' ')
        f.close()        
                
                