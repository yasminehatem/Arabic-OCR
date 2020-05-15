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
import matplotlib.pyplot as plt
import skimage.morphology
from skimage import img_as_ubyte
from skimage import feature
from scipy import stats
import math
from skimage.morphology import skeletonize
import os
import shutil
import csv

from skimage.morphology import binary_erosion, binary_dilation, binary_closing, skeletonize, thin
def Labeling(NumbOfWords,Characters,filename,var):
    
    f = open(filename, encoding='utf-8')
    #f = open('capr1.txt', encoding='cp720')
    #f = open('capr1.txt','r', encoding='windows-1256')
              
#    try:
#        shutil.rmtree("train")
#    except:
#        print("No train folder")
#    
#    os.mkdir("train")
    wordsfile=[]
    incwords=-1
    LamAlef=False
    wordsCount=0
    for line in f:
        for word in line.split():
            wordsfile.append(word)
            wordsCount+=1
    if(wordsCount==NumbOfWords):
        print(filename)
        for i in range(len(Characters)):
            for j in range(len(Characters[i])-1,-1,-1):
                incwords+=1
                for z in range(len(wordsfile[incwords])-1):
                    if(wordsfile[incwords][z]=='ل' and wordsfile[incwords][z+1]=='ا'):


                        LamAlef=True
                if(len(Characters[i][j])==len(wordsfile[incwords])):
                    for x in range(len(wordsfile[incwords])):
                        var+=1
                        #show_images([Characters[i][j][len(wordsfile[incwords])-1-x]])
                        myarr=[]
                        cv2.imwrite(str(var)+".png",Characters[i][j][len(wordsfile[incwords])-1-x]*255)
                        myarr.append(str(var)+".png")
                        if (wordsfile[incwords][x]=='لا'):
                            alphabet=0
                        elif(wordsfile[incwords][x]=='ا'):
                            alphabet=1
                        elif(wordsfile[incwords][x]=='ب'):
                            alphabet=2
                        elif(wordsfile[incwords][x]=='ت'):
                            alphabet=3
                        elif(wordsfile[incwords][x]=='ث'):
                            alphabet=4
                        elif(wordsfile[incwords][x]=='ج'):
                            alphabet=5
                        elif(wordsfile[incwords][x]=='ح'):
                            alphabet=6
                        elif(wordsfile[incwords][x]=='خ'):
                            alphabet=7
                        elif(wordsfile[incwords][x]=='د'):
                            alphabet=8
                        elif(wordsfile[incwords][x]=='ذ'):
                            alphabet=9
                        elif(wordsfile[incwords][x]=='ر'):
                            alphabet=10
                        elif(wordsfile[incwords][x]=='ز'):
                            alphabet=11
                        elif(wordsfile[incwords][x]=='س'):
                            alphabet=12
                        elif(wordsfile[incwords][x]=='ش'):
                            alphabet=13
                        elif(wordsfile[incwords][x]=='ص'):
                            alphabet=14
                        elif(wordsfile[incwords][x]=='ض'):
                            alphabet=15
                        elif(wordsfile[incwords][x]=='ط'):
                            alphabet=16
                        elif(wordsfile[incwords][x]=='ظ'):
                            alphabet=17
                        elif(wordsfile[incwords][x]=='ع'):
                            alphabet=18
                        elif(wordsfile[incwords][x]=='غ'):
                            alphabet=19
                        elif(wordsfile[incwords][x]=='ف'):
                            alphabet=20
                        elif(wordsfile[incwords][x]=='ق'):
                            alphabet=21
                        elif(wordsfile[incwords][x]=='ك'):
                            alphabet=22
                        elif(wordsfile[incwords][x]=='ل'):
                            alphabet=23
                        elif(wordsfile[incwords][x]=='م'):
                            alphabet=24
                        elif(wordsfile[incwords][x]=='ن'):
                            alphabet=25
                        elif(wordsfile[incwords][x]=='ه'):
                            alphabet=26
                        elif(wordsfile[incwords][x]=='و'):
                            alphabet=27
                        elif(wordsfile[incwords][x]=='ي'):
                            alphabet=28                      
                        myarr.append(alphabet)
                        with open("training.csv", 'a', newline='') as file:
                                 writer = csv.writer(file)   
                                 writer.writerow(myarr)
                                 
                elif(len(Characters[i][j])==len(wordsfile[incwords])-1 and LamAlef==True):
                    for x in range(len(wordsfile[incwords])):
                        var+=1
                        #show_images([Characters[i][j][len(wordsfile[incwords])-1-x]])
                        myarr=[]
                        cv2.imwrite(str(var)+".png",Characters[i][j][len(wordsfile[incwords])-2-x]*255)
                        myarr.append(str(var)+".png")
                        if(wordsfile[incwords][x]=='ل' and x<len(wordsfile[incwords])-1):
                            if(wordsfile[incwords][x+1]=='ا' or wordsfile[incwords][x]=='لا'):
                                alphabet=0
                                x+=1
                        elif(wordsfile[incwords][x]=='ا'):
                            alphabet=1
                        elif(wordsfile[incwords][x]=='ب'):
                            alphabet=2
                        elif(wordsfile[incwords][x]=='ت'):
                            alphabet=3
                        elif(wordsfile[incwords][x]=='ث'):
                            alphabet=4
                        elif(wordsfile[incwords][x]=='ج'):
                            alphabet=5
                        elif(wordsfile[incwords][x]=='ح'):
                            alphabet=6
                        elif(wordsfile[incwords][x]=='خ'):
                            alphabet=7
                        elif(wordsfile[incwords][x]=='د'):
                            alphabet=8
                        elif(wordsfile[incwords][x]=='ذ'):
                            alphabet=9
                        elif(wordsfile[incwords][x]=='ر'):
                            alphabet=10
                        elif(wordsfile[incwords][x]=='ز'):
                            alphabet=11
                        elif(wordsfile[incwords][x]=='س'):
                            alphabet=12
                        elif(wordsfile[incwords][x]=='ش'):
                            alphabet=13
                        elif(wordsfile[incwords][x]=='ص'):
                            alphabet=14
                        elif(wordsfile[incwords][x]=='ض'):
                            alphabet=15
                        elif(wordsfile[incwords][x]=='ط'):
                            alphabet=16
                        elif(wordsfile[incwords][x]=='ظ'):
                            alphabet=17
                        elif(wordsfile[incwords][x]=='ع'):
                            alphabet=18
                        elif(wordsfile[incwords][x]=='غ'):
                            alphabet=19
                        elif(wordsfile[incwords][x]=='ف'):
                            alphabet=20
                        elif(wordsfile[incwords][x]=='ق'):
                            alphabet=21
                        elif(wordsfile[incwords][x]=='ك'):
                            alphabet=22
                        elif(wordsfile[incwords][x]=='ل'):
                            alphabet=23
                        elif(wordsfile[incwords][x]=='م'):
                            alphabet=24
                        elif(wordsfile[incwords][x]=='ن'):
                            alphabet=25
                        elif(wordsfile[incwords][x]=='ه'):
                            alphabet=26
                        elif(wordsfile[incwords][x]=='و'):
                            alphabet=27
                        elif(wordsfile[incwords][x]=='ي'):
                            alphabet=28
                        myarr.append(alphabet)
                        with open("training.csv", 'a', newline='') as file:
                                 writer = csv.writer(file)   
                                 writer.writerow(myarr)
                                 
                LamAlef=False
    print(var)
    return var


def Thresholding(img):
    thresh = threshold_otsu(img)
    img[img>=thresh]=1
    img[img<thresh]=0
    return img
def rotate(img):
    img_blur = cv2.medianBlur(img,5).astype('uint8')
    thresh = cv2.threshold(cv2.bitwise_not(img_blur), 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)   
    return rotated
#Segment lines
def segment_Lines(img):
    Line_positions=[]
    first=False
    horiz=np.sum(img,axis=1)
   
    for i in range(len(horiz)):
        if(int(horiz[i])!=img.shape[1] and first==False):
            
            first=True
            j=i
        elif(int(horiz[i])==img.shape[1] and first==True ):
            if(i-j>3):
                Line_positions.append(j)
                Line_positions.append(i)
            first=False
            
    return Line_positions
def wordsPostions(dilatedlines,Lines):
    wordPosEachLine=[]
    for i in range(len(dilatedlines)):
        first=False
        wordPos=[]
        vertical=[]
        vertical=np.sum(dilatedlines[i],axis=0)
        for j in range(len(vertical)):
            if(int(vertical[j])<dilatedlines[i].shape[0] and first==False):
                k=j
                first=True
            elif(int(vertical[j])==dilatedlines[i].shape[0] and first==True ):
                if(j-k>=2):
                    wordPos.append(k)
                    wordPos.append(j)
                first=False
        wordPosEachLine.append(wordPos) 
    return   wordPosEachLine  
def wordSegmentation(dilatedlines,Lines):
    wordPosEachLine=[]


    for i in range(len(dilatedlines)):
        first=False
        wordPos=[]
        vertical=[]
        vertical=np.sum(dilatedlines[i],axis=0)
        for j in range(len(vertical)):
            if(int(vertical[j])<dilatedlines[i].shape[0] and first==False):
                k=j
                first=True
            elif(int(vertical[j])==dilatedlines[i].shape[0] and first==True ):
                if(j-k>2):
                    wordPos.append(k)
                    wordPos.append(j)
                first=False
        wordPosEachLine.append(wordPos) 
          
    
     
    words=[]    
    for i in range(len(wordPosEachLine)):
        wordsPerLine=[]
        for j in range(0,len(wordPosEachLine[i]),2):
            start=int(wordPosEachLine[i][j])
            end=int(wordPosEachLine[i][j+1])
            oneword=Lines[i][:,start:end]

            oneword[oneword==0]=2
            oneword[oneword==1]=0
            oneword[oneword==2]=1


            onewordCopy=np.copy(oneword)
            skeleton = skeletonize(onewordCopy)
            skeleton=skeleton.astype(int)


            wordsPerLine.append(skeleton)
                                
                                
            oneword[oneword==0]=2
            oneword[oneword==1]=0
            oneword[oneword==2]=1

        words.append(wordsPerLine)
    return words




def BaselineIndex(lines):
    Baseline=[]
    for line in lines:
        
        
        line[line==1]=25
        line[line==0]=1
        line[line==25]=0
        
        xline= thin(line,160) #TESTTTTTTTTS
        hp = np.sum(xline,axis=1)
        blur=gaussian(hp)
        PeakValue= np.argmax(blur)
        line[line==1]=25
        line[line==0]=1
        line[line==25]=0
        b=0
        if(PeakValue+1<line.shape[0]):
            b=PeakValue+1
        else:
            b=PeakValue-3
#        print(b)    
        Baseline.append(b)
#        print(line.shape[0],PeakValue+1)
    return Baseline
	
	
def MaxTrans(Lines, BaseLineIndex):
    MaxTransf=[]
    for i in range(len(Lines)):
        line=Lines[i]
        MaxTrans = 0
        MaxTransIndex = BaseLineIndex[i]
        j=BaseLineIndex[i]
        while j > 0:
            Flag = 0
            Current = 0
            
            k=line.shape[1] - 1
#            if(i==38):
#                show_images([Lines[i]])
##                print(line[11][0])
#                print(k,j, BaseLineIndex[i],line.shape[0])
            while k >= 0:
                

                if (line[j, k] == 1.00 and Flag == 0):
                    Current += 1
                    Flag = 1
                    
                if (line[j, k] != 1.00 and Flag == 1):
                    Flag = 0
                k -= 1
    
            if Current >= MaxTrans:
                MaxTrans = Current
                MaxTransIndex = j
            j -= 1
#        print('max transindex is')
#        print(MaxTransIndex)
        if(i==38):
            print(BaseLineIndex[i],MaxTransIndex,Lines[i].shape[0])
        MaxTransf.append(MaxTransIndex)
    return MaxTransf


    
    
def smallestIndex(x, value):
    x = np.asarray(x)
    dist = (np.abs(x - value))
    i=dist.argmin()
    return x[i]	
class SeparationRegion:
    start=0
    cut=0
    end=0
def CutPointIdentification(line,blackWord,MaxTransIndex):
    
    SeparationRegions=[]
    for l in range(len(blackWord)):
        
        lineSR=[]
        for f in range(len(blackWord[l])):
            blackoneWord=blackWord[l][f]
            i=1
            MTI=MaxTransIndex[l]
            flag=0
            VP = np.sum(blackoneWord,axis=0)
   
            MFV=stats.mode(VP)
            MFV=MFV[0]
            oneWordSR=[]
     
            while(i<blackoneWord.shape[1]-1):
               
                    
                if(blackoneWord[MTI,i] == 1 and blackoneWord[MTI,i+1] == 0 and flag == 0):
                    SR = SeparationRegion()
                    SR.end=i
                    flag=1
                
                elif(blackoneWord[MTI,i] == 0 and blackoneWord[MTI,i+1] == 1 and flag == 1):
                    SR.start=i
#        
                    MidIndex = ( SR.end + SR.start )/2
                    MidIndex = int(MidIndex)
                    EndtoStart=[]

                    
                    for i in range(SR.end,SR.start+1):
                        if(VP[i]==0):
                            EndtoStart.append(i)
                            break

                    EndtoMid=[]        
                    for i in range(MidIndex,SR.end,-1):
                        if(VP[i]<= MFV):
                            EndtoMid.append(i)
                            break


                    MidtoStart=[]
                    for i in range(MidIndex,SR.start+1):
                        if(VP[i]<= MFV):
                            MidtoStart.append(i)
                            break
                    

                    
                    if VP[MidIndex] == MFV:
                        SR.cut = MidIndex 
#                    elif(len(EndtoMidMFV)!=0):
#                         SR.cut =EndtoMidMFV[0]
#                    elif(len(MidtoStartEMFV)!=0):
#                         SR.cut =MidtoStartEMFV[0]     
                        
                    elif len(EndtoMid) != 0: 
                        
                        SR.cut = smallestIndex(EndtoMid , MidIndex)
                    
                    elif len(EndtoStart) != 0: 
                        
                        SR.cut = EndtoStart[0]
        
                

                    
                    elif len(MidtoStart) != 0:
                        SR.cut= smallestIndex(MidtoStart , MidIndex)
                    else:
                        SR.cut = MidIndex
                    flag=0
                    oneWordSR.append(SR)
                    
                i+=1   
            lineSR.append(oneWordSR)
                       

        SeparationRegions.append(lineSR)

    return 	SeparationRegions
def DetectHoles2(BI,blackoneWord,cut,line):
    for v in range (BI):
        if(blackoneWord[v][cut]==1):
            HeightofHole=BI - v
            break
    distBItoMIT=GetHeight(line,BI)    
    countChange=0
    for v in range(BI):
        if(blackoneWord[v][cut]!=blackoneWord[v+1][cut]):
            countChange=countChange+1
    if(countChange>=3 and HeightofHole< distBItoMIT  ): #hole harf kaf we 3en 
        return True
#    elif(countChange>=3 and HeightofHole< 0.8* distBItoMIT ): #lwcut fy akher kaf
#        return True 
    else:
        return False

def GetHeight(blackoneword,BI):
    height=0
    blackonewordHP = np.sum(blackoneword, axis=1)
    for i in range(len(blackonewordHP)):
        if blackonewordHP[i] != 0:
            height = BI - i
            break    
    return height    
def lastRegionEmptyRegion(SRCurrent,SRprevCut,blackoneWord,MTI,BI,k,VP,line):

    distBItoMIT=GetHeight(line,BI) 

    HeightLeftPixel=0
    flag=0

    HP=np.sum(blackoneWord[:,SRprevCut:SRCurrent.cut],axis=1)
    for i in range(BI):
        if(HP[i]!=0):
             HeightLeftPixel=i 
             flag=1
             break
    distLefttoBI= np.abs(BI - HeightLeftPixel) 

    if(( k==0 )and (distLefttoBI<(0.5*distBItoMIT))and flag==1 and  SRCurrent.cut<6): #last Region (h)
      #condition <6 3shan elreh matkoshesh hena kont 3amlha 8 3shan yeh be no2teten 
        return True
    if(k==0):
        return False
       
    if((VP[SRprevCut]==0)and (distLefttoBI<(0.5*distBItoMIT))and flag==1):#nos kelma (g)

        return True
    return False 
def DALALFNOSKELMA(SRCurrent,SRprevCut,blackoneWord,MTI,BI,k,VP,line):
    
    distBItoMIT=GetHeight(line,BI) 

    HeightLeftPixel=0
    flag=0

    HP=np.sum(blackoneWord[:,SRprevCut:SRCurrent.cut],axis=1)
    for i in range(BI):
        if(HP[i]!=0):
             HeightLeftPixel=i 
             flag=1
             break
    distLefttoBI= np.abs(BI - HeightLeftPixel) 
    HeightRightPixel=BI
    for i in range(BI):
        if(HP[i]!=0):
             HeightRightPixel=i 
             flag=1
             break
#    distLefttoBI= np.abs(BI - HeightRightPixel) 
       
    if((VP[SRprevCut]==0)and BI -HeightRightPixel>=5 and VP[SRCurrent.end+1]<=1  and (distLefttoBI<=(distBItoMIT))and flag==1):#nos kelma (g)

        return True

    return False 
def IsStroke(SR,BI,blackoneword,SRprevCut,line):
#    c=count_connected(img)
#    if(c==1):
#        return True
    SHPA=[]
    SHPA=np.sum(blackoneword[0:BI,SRprevCut:SR.cut],axis=1)
    SHPB=np.sum(blackoneword[BI+1:,SRprevCut:SR.cut],axis=1)
    SHPA=np.sum(SHPA)
    SHPB=np.sum(SHPB)

    HeightSegmented=GetHeight(blackoneword[:,SRprevCut:SR.cut],BI)
    HLine=GetHeight(line,BI)
    if(DetectDot(BI,blackoneword,SR.cut,SRprevCut,SR) and SHPA>SHPB and not DetectHoles2(BI,blackoneword,SR.cut,line) ):#harf sheen
        return True
    if (SHPA>SHPB and not DetectHoles2(BI,blackoneword,SR.cut,line) and HeightSegmented <=math.ceil( HLine)):#kant 0.3
        return True
    return False
        
def DetectDot(BI,blackoneWord,cut,prevCut,SR):

    
    
    countChangeAbove=0
    countChangeBelow=0
    hpABI = np.sum(blackoneWord[0:BI,prevCut:cut],axis=1)
    hpBBI = np.sum(blackoneWord[BI+1:,prevCut:cut],axis=1)
    
    firstHpBBI=hpBBI[0]
    if( np.sum( blackoneWord[BI,SR.end+1:SR.start] ) == 0 and firstHpBBI==0): #harf non
        countChangeBelow=0
        flag=1
    elif( firstHpBBI==0)  :
        countChangeBelow=1 ####BEH YEEH
        flag=1
    else:
        flag=0
    firstHpABI=hpABI[0]
    if(firstHpABI==0):
        countChangeAbove=1
        flag2=1
    else:
        flag2=0              
    for v in range(1,len(hpBBI)):
        if(hpBBI[v]!=0 and flag==1):
            countChangeBelow=countChangeBelow+1
            flag=0
        elif(hpBBI[v]==0 and flag==0):
            countChangeBelow=countChangeBelow+1
            flag=1


    for v in range(1,len(hpABI)):
        if(hpABI[v]!=0 and flag2==1):
            countChangeAbove=countChangeAbove+1
            flag2=0
        elif(hpABI[v]==0 and flag2==0):
            countChangeAbove=countChangeAbove+1
            flag2=1
            
            
    if(countChangeBelow>=3 or countChangeAbove>=3):
        return True
    else:
        return False
    
def CutPointIdentificationFilteration(blackWord,Baseline,MaxTransIndex,SRArray,line):
    validSeparationRegions=[]
    for l in range(len(blackWord)):
        validlineSR=[]
        for f in range(len(blackWord[l])):
            i=0
            SRWord=SRArray[l][f]
            blackoneWord=blackWord[l][f]
            MTI=MaxTransIndex[l]
            BI=Baseline[l]
           
            VP = np.sum(blackoneWord,axis=0)

            
            
            
            MFV=stats.mode(VP)
            MFV = MFV[0]
            validoneWordSR=[]
            SRNext=SeparationRegion()
            SRPre=SeparationRegion()
    
            while(i<len(SRWord)):
                SR=SRWord[i]
               
                PathFromStartEnd= blackoneWord[BI,SR.end+1:SR.start]
                if(i>0):
                    SRprevCut=SRArray[l][f][i-1].cut 
                else:
                    SRprevCut=0
                if(i+1<len(SRWord)):
                    SRNext=SRWord[i+1]  
                else:
                    SRNext=SR  #DUMMY DATA
                if(i-1>=0):
                    SRPre=SRWord[i-1]  
                else:
                    SRPre=SR   #DUMMY DATA
                    
            
#                if(l==5 and f==6 and i==1):
#                    HP=np.sum(blackoneWord[:,SR.cut:SR.start],axis=1)
#                    HeightRightPixel=0
#                    for i in range(BI):
#                        if(HP[i]!=0):
#                             HeightRightPixel=i 
#                             break
#                    print(BI,HeightRightPixel,lastRegionEmptyRegion(SR,SRprevCut,blackoneWord,MTI,BI,i,VP,line[l],),DALALFNOSKELMA(SR,SRprevCut,blackoneWord,MTI,BI,i,VP,line[l]))    
                    
                
                if(VP[SR.cut]==0 ):
                    validoneWordSR.append(SR)
                    
                    
                    
                    i=i+1
                

                elif(DetectHoles2(BI,blackoneWord,SR.cut,line[l])): #hole harf kaf we 3en (SAH TESTED)
                    
                    
                     i=i+1 #invalid
                    
                elif (not(1 in PathFromStartEnd) and not  lastRegionEmptyRegion(SR,SRprevCut,blackoneWord,MTI,BI,i,VP,line[l]) and not(IsStroke(SRNext,BI,blackoneWord,SR.cut,line[l]) ) and not ( DetectHoles2(BI,blackoneWord,SR.cut,line[l]) )): #no path harf reh 
                     #notlastRegionEmptyRegion mtkhoshesh HARF LAM
                     #STROKE 
                     
                    validoneWordSR.append(SR)
                    i=i+1
              
                
                elif( np.sum( blackoneWord[BI,SR.end:SR.start] ) == 0):  #harf non (SAH TESTED)                     
                    SHPB=np.sum(blackoneWord[BI+1:,SR.end:SR.start+1],axis=1)
                    SHPA=np.sum(blackoneWord[0:BI,SR.end:SR.start+1],axis=1)
                    SHPB = np.sum(SHPB)
                    SHPA = np.sum(SHPA)
                    
                    
                    
                    
                    if(SHPB>SHPA):

                        i=i+1
                    elif(VP[SR.cut]<=MFV):
                        validoneWordSR.append(SR)
                        i=i+1
                    else:
                        i=i+1
                elif(DALALFNOSKELMA(SR,SRprevCut,blackoneWord,MTI,BI,i,VP,line[l])):
                    validoneWordSR.append(SR)
#                    if(l==5 and f==6 and i==1):
#                        print(SR.start,SR.end)
                    i=i+1       
                elif(lastRegionEmptyRegion(SR,SRprevCut,blackoneWord,MTI,BI,i,VP,line[l])):
#                    if(i==0):
#                        print(l,f)

                    
                    i=i+1   
                
                    
                elif ( i+2<len(SRWord) and not(IsStroke(SRNext,BI,blackoneWord,SR.cut,line[l]) )):
                     startToEndBaselineNextRegion= blackoneWord[BI,SRWord[i+2].end:SRWord[i+2].start+1]
                     ###TEHH BTKHOSH HENA 3'alat
                     
                     
          
                     
                     if(np.sum(startToEndBaselineNextRegion)==0 and SRWord[i+2].cut<=MFV ):
                   
                         i=i+1
                            
                     else:
                         validoneWordSR.append(SR)
                         i=i+1
                elif(i+1<len(SRWord) and IsStroke(SRNext,BI,blackoneWord,SR.cut,line[l])and DetectDot(BI,blackoneWord,SRNext.cut,SR.cut,SRNext)): #harf beh
  
                    validoneWordSR.append(SR)
                    
#                    print(l,f,i)
                    i=i+1
                    
                    
                elif(i+1<len(SRWord)):
                    
                    
                    
                    
#                    if(l==6 and f==3 and i==3):
#                            HeightSegmented=GetHeight(blackoneWord[:,SR.cut:SRWord[i+1].cut],BI)
#                            HeightSegmented2=GetHeight(blackoneWord[:,SRWord[i+2].cut:SRWord[i+3].cut],BI)
#                            HLine=GetHeight(line[l],BI)
#                    
                   
                    if (IsStroke(SRWord[i+1],BI,blackoneWord,SR.cut,line[l])and not DetectDot(BI,blackoneWord,SRWord[i+1].cut,SR.cut,SRWord[i+1]) ):
#                        if(i==2 and l==3 and f==6):
#                            print(  DetectDot(BI,blackoneWord,SRWord[i+1].cut,SR.cut,SRWord[i+1]))
                            
                        if(i+2<len(SRWord) and IsStroke(SRWord[i+1],BI,blackoneWord,SR.cut,line[l]) and not DetectDot(BI,blackoneWord,SRWord[i+1].cut,SR.cut,SRWord[i+1])):#harf seen
                            SHPB=np.sum(blackoneWord[BI+1:,SRPre.end:SRPre.start+1],axis=1)
                            SHPA=np.sum(blackoneWord[0:BI,SRPre.end:SRPre.start+1],axis=1)
                            SHPB = np.sum(SHPB)
                            SHPA = np.sum(SHPA)

#                            if(i==1 and l==14 and f==1):
#                                print( DetectDot(BI,blackoneWord,SR.cut,SRprevCut,SR) ,SHPB>SHPA ,np.sum( blackoneWord[BI,SRPre.end:SRPre.start] ))
                            if(i+2<len(SRWord) and i-1>=0 and  not DetectDot(BI,blackoneWord,SR.cut,SRprevCut,SR) and SHPB>SHPA and  not  DetectHoles2(BI,blackoneWord,SRWord[i+2].cut,line[l]) ):
                                validoneWordSR.append(SRWord[i+2]) #HARF SEEN FY AKHER KELMA
                                
                                i=i+3
                                continue
                            if(i+3<len(SRWord) and IsStroke(SRWord[i+1],BI,blackoneWord,SR.cut,line[l])and DetectHoles2(BI,blackoneWord,SRWord[i+2].cut,line[l]) and not DetectDot(BI,blackoneWord,SRWord[i+1].cut,SR.cut,SRWord[i+1])):#harf sad we seenoh 
                                validoneWordSR.append(SR)  #HARF SAD WE SEN ELB3DO
                                validoneWordSR.append(SRWord[i+3])
                                i=i+4
                                continue
                            
                            if(i+3<len(SRWord)):
                                validoneWordSR.append(SR)
                                validoneWordSR.append(SRWord[i+3])
                                i=i+4
                                continue
                            else:
                                validoneWordSR.append(SR)
                                i=i+3
                                continue
                        if(i+2<len(SRWord) )  :
                            if(IsStroke(SRWord[i+2],BI,blackoneWord,SRWord[i+1].cut,line[l])and DetectDot(BI,blackoneWord,SRWord[i+2].cut,SRWord[i+1].cut,SRWord[i+2]) and IsStroke(SRWord[i+3],BI,blackoneWord,SRWord[i+2].cut,line[l]) and not DetectDot(BI,blackoneWord,SRWord[i+3].cut,SRWord[i+2].cut,SRWord[i+3]) ):
                                validoneWordSR.append(SR)
                                if(i+3<len(SRWord)):
                                    validoneWordSR.append(SRWord[i+3])
                                    i=i+4
                                    continue
                                else:
                                    i=i+3
                                    continue


                            if(IsStroke(SRWord[i+2],BI,blackoneWord,SRWord[i+1].cut,line[l])and  DetectDot(BI,blackoneWord,SRWord[i+2],SRWord[i+1].cut,SRWord[i+2])):
                                
                                i=i+1
                                continue
                            if not (IsStroke(SRWord[i+1],BI,blackoneWord,SR.cut,line[l])):
                                i=i+1   
                                continue
                            else:  #TO TEST ######################
                                validoneWordSR.append(SR)
                                i=i+1
                        else:  #TO TEST ######################FARAHHHHHHHH
                                validoneWordSR.append(SR) 
                                
                                i=i+1         
                    else:
                        
                        validoneWordSR.append(SR)
                        i=i+1
                        
                else:  #TO TEST ######################
                    
                    
                    validoneWordSR.append(SR)
                    i=i+1 
            validlineSR.append(validoneWordSR)        
        validSeparationRegions.append(validlineSR)
    return 	validSeparationRegions
    
	
	
def getCharcter(blackWordLines,SR,wordPosEachLine):
    lines=[]
#    word=[]
    for i in range(len(SR)):
        word=[]
        
        if(len(SR[i])==0):
       
            oneword=[]
            word.append(oneword)
            continue
        count=1   
        for j in range(len(SR[i])): #word
            if(len(SR[i][j])==0):
           
                oneword=[]
                word.append(oneword)
                count=count+2
               
                continue
            
            oneword=[]
            end=0  #first word
            start=SR[i][j][0].cut
            saveStart=start
           
            
#            print(end,start)
            oneCharacter=blackWordLines[i][j][:,end:start]
            oneword.append(oneCharacter)
            for k in range(len(SR[i][j])-1):
                end=SR[i][j][k].cut
                start=SR[i][j][k+1].cut
                saveStart=start
                oneCharacter=blackWordLines[i][j][:,end:start]
                oneword.append(oneCharacter) #####append char
          
            end=saveStart
            start=wordPosEachLine[i][count]
            if(end<start):
                
                
                oneCharacter=blackWordLines[i][j][:,end:start] 
                oneword.append(oneCharacter)#array conatin all charc of one word
                word.append(oneword) #array contain all words of one line
            count=count+2
        lines.append(word)    
    return lines
	
def getwordsNumber(blackWordLines):
    count=0
    for i in range(len(blackWordLines)):
        count=count+len(blackWordLines[i])
    return count

def getnumberofcharc(blackWordLines):
    count=0
    for i in range(len(blackWordLines)):
        for j in range(len(blackWordLines[i])):
            count=count+len(blackWordLines[i][j])
    return count
#def testAcc(word):
##    File= open("dataset2/text/capr1.txt",  encoding="utf8")
##    Lines = File.readlines()
##    for line in Lines:
##        wordsCount=0
##        for word in line.split():
##            wordsCount+=1
##            print(wordsCount)
##            #print(len(line))
##    print("farah")        
#    for i in range(len(words)):
#        print(i,len(words[i]))
	
	
#image = (rgb2gray(io.imread("scanned/scanned/capr3.png")))
def main(image,filename,var):
    
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
#    print(wordPosEachLine[0])
#    for i in range(len(Lines)):
#        show_images([Lines[i]])    
#    f =37
#    for j in range(len(words[f])):
#        show_images([words[f][j]])    
##    
#    show_images([Lines[38]])
    BI = BaselineIndex(Lines )
      
    
    MaxTransitionIndex =MaxTrans(Lines, BI)
    
    SeparationRegions=CutPointIdentification(Lines,words,MaxTransitionIndex)
    SeparationRegions= CutPointIdentificationFilteration(words,BI,MaxTransitionIndex,SeparationRegions,Lines)
    arraychar=getCharcter(words,SeparationRegions,wordPosEachLine)
#    print(len(words))
#    testAcc(words)
    
    nWords=getwordsNumber(arraychar)
##    nWords=getnumberofcharc(arraychar)
#    print(nWords)
       
        
    #f = 0
    #for f in range(len(arraychar)):
    #    for i in range(len(arraychar[f])):
    #        for j in range(len(arraychar[f][i]) ):
    #            show_images([arraychar[f][i][j]])  
            
    #f = 28
    #
#    for i in range(len(arraychar)):
#        for j in range(len(arraychar[i]) ):
#            show_images([arraychar[i][j]])          
            
    #show_images([arraychar[0][0][0]])         
            
#    word =words[150][0]
#    show_images([word])
#    sr=SeparationRegions[12][2]
##    
#    word[word==0]=2
#    word[word==1]=0
#    word[word==2]=1
#    S = sr[0]
#    
#    img = cv2.line(word,(S.cut,0),(S.cut,line.shape[0]),(0,255,0),1)
#    show_images([img])
#    
#    S = sr[1]
#    img = cv2.line(word,(S.cut,0),(S.cut,line.shape[0]),(0,255,0),1)
#    show_images([img])
#    
#    
#    S = sr[2]
#    img = cv2.line(word,(S.cut,0),(S.cut,line.shape[0]),(0,255,0),1)
#    show_images([img])
#    
#    
#    S = sr[3]
#    img = cv2.line(word,(S.cut,0),(S.cut,line.shape[0]),(0,255,0),1)
#    show_images([img])
#    
#    S = sr[4]
#    img = cv2.line(word,(S.cut,0),(S.cut,line.shape[0]),(0,255,0),1)
#    show_images([img])
    
    
#    S = sr[5]
#    img = cv2.line(word,(S.cut,0),(S.cut,line.shape[0]),(0,255,0),1)
#    show_images([img])
#    #
#    #
#    S = sr[6]
#    img = cv2.line(word,(S.cut,0),(S.cut,line.shape[0]),(0,255,0),1)
#    show_images([img])
    
    
#    S = sr[7]
#    img = cv2.line(word,(S.cut,0),(S.cut,line.shape[0]),(0,255,0),1)
#    show_images([img])
#    
#    
#    S = sr[8]
#    img = cv2.line(word,(S.cut,0),(S.cut,line.shape[0]),(0,255,0),1)
#    show_images([img])
#    
#    #
#    S =  sr[9]
#    img = cv2.line(word,(S.cut,0),(S.cut,line.shape[0]),(0,255,0),1)
#    show_images([img])
#    
#    S = sr[10]
#    img = cv2.line(word,(S.cut,0),(S.cut,line.shape[0]),(0,255,0),1)
#    show_images([img])
#    return arraychar

    var=Labeling(nWords,arraychar,filename,var)

    return var
        

#image = (rgb2gray(io.imread("scanned/scanned/caug1874.png")))
#words=main(image)
#print(len(words))
cv2.waitKey(0)
cv2.destroyAllWindows()