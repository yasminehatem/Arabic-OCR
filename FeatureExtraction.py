import cv2
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 

from skimage.color import rgb2gray
from skimage.morphology import binary_erosion, binary_dilation, binary_closing,skeletonize, thin
from skimage.measure import find_contours
from skimage.draw import rectangle
from skimage.filters import threshold_otsu
from skimage.viewer import ImageViewer
from skimage.filters import gaussian
from statistics import mode
from commonfunctions import *
from PIL import Image
from skimage.filters import threshold_otsu

from scipy import ndimage

def BaselineIndex(threshimg):
    BaseLineIndex = 0
    HP = []
    PV = []
    
    HP = np.sum(threshimg, axis = 1)
    #print('HP')
    #print(HP)
    MFV= mode(HP)
    #print("mfv")
    #print(MFV)
    
 
    IND_PV = (HP > np.roll(HP,1)) & (HP > np.roll(HP,-1))
    for i in range(len(IND_PV)):
        if IND_PV[i] == True:
            PV.append(HP[i])
    #print(PV)
    MAX = max(PV)
    for i in range(len(HP)):
        if HP[i] == MAX:
            BaseLineIndex = i
    #print('BaseLineIndex')        
    #print(BaseLineIndex)
    return BaseLineIndex

    def checkdots(letter):
    
  
    for i in range(letter.shape[1]):
        
        
        for z in range(letter.shape[0]):
            if(z+1 !=letter.shape[0]):
                    if (letter[z][i]==1 and letter[z+1][i]==0):
                        #print(z)
                        #print(i)
                        #print(letter[z][i])
                        for j in range(z+1,letter.shape[0]):
                            if(j+1 !=letter.shape[0]):
                                if(letter[j+1][i]==1 ):
                                    hp=np.sum(letter[z+1:j+1,0:letter.shape[1]], axis = 1)
                                    s=set(hp)
                                    if 0 in s:
                                        print("ok")
                                        #print(i+1:j+1)
                                        #return 1,i+1:j+1,
                                        return 1
                                    
    return 0
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
  #take binary image of 1 char , char white background black return a vector with the places of 1 and zeros
def pixelfeatextrac(img):
    myfitimg=np.copy(img)
    
    myfitimg=fitimg(myfitimg)
    
    for i in range(myfitimg.shape[0]):
        for j in range(myfitimg.shape[1]):
            pixelvector.append(myfitimg[i][j])
    return pixelvector

  #all return vector
def HPaboveBL(letter,BLindex):
    
    
    #BLindex=BaselineIndex(letter)
   
    #print(letter[0:BLindex,0:letter.shape[1]])
    HPABL = np.sum(letter[0:BLindex,0:letter.shape[1]], axis = 1)
    return HPABL
    
    
    
    
def HPbelowBL(letter,BLindex):
    
    #BLindex=BaselineIndex(letter)
   
    #print(letter)
    HPBBL = np.sum(letter[BLindex:letter.shape[0],0:letter.shape[1]], axis = 1)

    return HPBBL
##############################e7tmal el etnn l gayin mykonsh lehom lzma ######

def VPaboveBL(letter, BLindex):
    
    #BLindex=BaselineIndex(letter)
    
    #print(letter[0:BLindex,0:letter.shape[1]])
    VPABL = np.sum(letter[0:BLindex,0:letter.shape[1]], axis = 0)
    return VPABL

def VPbelowBL(letter,BLindex):
    
    #BLindex=BaselineIndex(letter)
   
    VPBBL = np.sum(letter[BLindex:letter.shape[0],0:letter.shape[1]], axis = 0)
    return VPBBL

### return if there is at lesat 1 dot(because dots are connected) returns 1 , if no dots return 0
def checkdotes2(letter):
    
    labeled,nr_objects=ndimage.label(letter>0)
    print(labeled)
    
    #plt.imshow(labeled)
  
    
    return nr_objects-1 ,labeled

## get the number of holes 
def count_holes(letter,c):
   
    c,labeled=checkdotes2(letter)
    contours,hierarchy=cv2.findContours(letter,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)-c-1
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

 def getwidth(letter): #return width of a letter can be less accurte if the letter is segmented wrong
    
    start=letter.shape[1]
    #plt.imshow(letter)
    if cv2.countNonZero(image) == 0:
        return 0
                
               
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
    print(letter)
    return end-start   
    
#call this
def getheight(letter):   
    (h, w) = letter.shape[:2]
    center = (w / 2, h / 2)
    scale = 1.0
    M = cv2.getRotationMatrix2D(center,90, scale)
    rotated90 = cv2.warpAffine(letter, M, (h, w))
    plt.imshow(rotated90)
     
    return getwidth(rotated90)

##########testing pixelfeatextrac##########
fname='10.png'

img=cv2.imread(fname,cv2.IMREAD_GRAYSCALE)
#thresh,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)#inver bckgrd and writing color, bgkgrd nw black

#print(threshold)

#print(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j]==255:
            img[i][j]=1
        else:
            img[i][j]=0
vector=pixelfeatextrac(img)
print(vector)



          
    
    