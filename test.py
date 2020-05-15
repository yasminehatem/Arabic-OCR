# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 21:28:24 2019

@author: farah_
"""
from untitled0 import *

import cv2
import os
import shutil

def WordLength(W):
    count = 1
    for i in range(len(W)-1):
        if W[i]+W[i+1] != "لا":
            count+=1
    return count    

def Train(NumberOfData):

    ImgCount = 0

    AllLength = 0
    AllCorrect = 0
    count = 0
    WrongImgs=0

    try:
        shutil.rmtree("train")
    except:
        print("No train folder")

    os.mkdir("train")


    # exit(0)
    for scanned in os.listdir('dataset2/scanned'):

        Path = 'dataset2/scanned/'+scanned
        # Path2 = 'dataset/scanned2/'+scanned
        print(Path)
        Img = io.imread(Path)
#        show_images([Img])
        Words = main(Img)
#        try:
#            S.Start()
#        except:
#            print("Error in reading image")
#            continue
        
        FileName = 'dataset2/text/'+scanned[:-4] +'.txt'
        print(FileName)
        wordsfile=[]
        wordsCount=0
        File = open(FileName,  encoding="utf8")
        Lines = File.readlines()
        for line in Lines:
            for word in line.split():
                wordsfile.append(word)
                wordsCount+=1
#        Words = S.GetSegmentedWords()
        # print(len(Words2))
        print("================================")
        nymwords=getwordsNumber(Words)
        if wordsCount != nymwords:
            print("Error in Words")
            print("Number Of True Words: "+str(wordsCount))
            print("Number of Words: " + str(nymwords))
            WrongImgs +=1
            continue
#
#        File = open("associtations.txt", "a")
        Correct = 0
        c=0
        for i in range(len(Words)):
            for j in range(len(Words[i])-1,-1,-1):
#                print(wordsfile[c])
#                print(len(Words[i][j]))
#                print()
#                for z in range(len(Words[i][j])):
#                    show_images([Words[i][j][z]])
                if(len(Words[i][j])==len(wordsfile[c])):
                    Correct += 1
#                    print(wordsfile[c])
#                    for z in range(len(Words[i][j])):
#                        show_images([Words[i][j][z]])
                c+=1
        
                    
                    
#        ImgCount+=1
#        if WordLength(RealWords[i]) == WL :
#            Correct += 1
#            print(RealWords[i])
#            for z in range(len(Words[Length-1-i])):
#                show_images([Words[Length-1-i][z]])
#            for j in range(WL):
#                    name = str(ImgCount)+".png"
#                    cv2.imwrite("train/"+name,Words[i][j])
#                    File.write(str(RealWords[i][j])+" " + name+"\n" )

        AllLength += wordsCount
        AllCorrect += Correct

        count += 1
        if count == NumberOfData:
            break

    File.close()
    AllAccuracy = (AllCorrect / AllLength) * 100
    print("Segmentation Finished")
    print(str(WrongImgs) + " Failed Images")
    print("Testing on " + str(AllLength) + " Words ")
    print(str(AllCorrect) + " Are Correct")
    print("Accuracy : "+str(AllAccuracy) +"%")


Train(2)
#image = (rgb2gray(io.imread("scanned/scanned/capr3.png")))
#main(image)
