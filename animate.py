# -*- coding: utf-8 -*-
"""

"""
from starrynight import *
import cv2

def doit(anArtist,params):
        frame=0
        #print params
        # params 0, split level
        # params 1,2,3 top color
        # params 4,5,6 bottom color
        # either blur, or not
        # rest are circle paths, of length 8 each
        #
        # Use caching to try and speed this up
        img=np.zeros(anArtist.shape,anArtist.dtype)
        cv2.imwrite(outputDir+"frame_%06d.png"%frame,img)
        frame+=1
        for c in range(3):
           si=params[0] #np.floor(s)
           #mantissa=s-si
           img[:si,:,c]=params[c+1]
        cv2.imwrite(outputDir+"frame_%06d.png"%frame,img)
        frame+=1
        for c in range(3):
           si=params[0] #np.floor(s)
           #mantissa=s-si
           img[si:,:,c]=params[c+4]
        cv2.imwrite(outputDir+"frame_%06d.png"%frame,img)
        frame+=1
        # a circle path is center x, y, offsetx, offsety, radius, r,g,b
        if (len(params)-7)%8 == 0:
           circles=params[7:]
           blurSize=None
        else:
           circles=params[8:]
           blurSize=params[7]
        assert len(circles)%8 == 0
        for i in range(0,len(circles),8):
            circ=circles[i:i+8]
            circ[0]=circ[0]-circ[2]//2
            circ[1]=circ[1]-circ[3]//2
            img=img.copy()
            for j in range(62):
                r=circ[0]+circ[2]*j//62 
                c=circ[1]+circ[3]*j//62 
                cv2.circle(img,(r,c),circ[7],anArtist.fromColorSubspace(anArtist.toColorSubspace(circ[4:7])),-1)
                cv2.imwrite(outputDir+"frame_%06d.png"%frame,img)
                frame+=1
            cv2.imwrite(outputDir+"frame_%06d.png"%frame,img)
            frame+=1
            cv2.imwrite(outputDir+"frame_%06d.png"%frame,img)
            frame+=1
        if not blurSize is None:
            for b in range(1,blurSize+1):
              cv2.imwrite(outputDir+"frame_%06d.png"%frame,cv2.blur(img,(b,b)))
              frame+=1
            return cv2.blur(img,(blurSize,blurSize))
        return img.copy()

sizelimit=1024
orig=cv2.imread('ORIGINAL.png')
pfd=open("foundparamsv27final_1_1024.txt")
outputDir="starryanim/"
if os.path.exists(outputDir):
     shutil.rmtree(outputDir)
os.mkdir(outputDir)

#foundparamsv13final_%d.txt
xstr=pfd.readline()
bstr=pfd.readline()
pfd.close()
#exec("x="+xstr)
#exec("b="+bstr)
exec("params="+xstr)
exec("bounds="+bstr)
anArtist=artist(orig)
doit(anArtist,params)
