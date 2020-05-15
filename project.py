
from untitledCharacter import *

import cv2
import os
import shutil
import sys
import xlrd
import csv
from scipy import ndimage
import xlwt 
from xlwt import Workbook 
from PIL import Image

import csv

var=-1
for scanned in os.listdir('dataset2/scanned'):
        
        Path = 'dataset2/scanned/'+scanned
        FileName = 'dataset2/text/'+scanned[:-4] +'.txt'
        image = (rgb2gray(io.imread(Path)))
        
        var=main(image,FileName,var)