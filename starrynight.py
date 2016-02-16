# -*- coding: utf-8 -*-
"""
Created for code golf starry night
http://codegolf.stackexchange.com/questions/69930/paint-starry-night-objectively-in-1kb-of-code

Some really interesting ideas from other posts taken here
- use of a blur to push the final score up a bit
- use of binary code directly in the final python output
- use of a 2D subspace of the color space
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from skimage import filters
import bz2
import zlib
import base64
import sys
import os
import shutil





def score(img,orig):
    z=(img.astype(np.float32)-orig.astype(np.float32)).ravel()**2
    return np.sum(z)/255./255.

def bound(a,low,high):
    return max(min(a,high),low)

class artist:
    def __init__(self,img):
        self.shape=img.shape
        self.dtype=img.dtype
        
    def doit(self,params):
        #print params
        # params 0, split level
        # params 1,2,3 top color
        # params 4,5,6 bottom color
        #s=bound(params[0],1,img.shape[0]-2)
        img=np.zeros(self.shape,self.dtype)
        for c in range(3):
           c1=bound(params[c+1],0,255)
           c2=bound(params[c+4],0,255)
           si=params[0] #np.floor(s)
           #mantissa=s-si
           img[:si,:,c]=c1
           img[si:,:,c]=c2
           #img[si+1:,:,c]=c2
           #img[si,:,c]=c1*mantissa+c2*(1.-mantissa)
        # a circle is x, y, radius, r,g,b
        if (len(params)-7)%6 == 0:
           circles=params[7:]
           blurSize=None
        else:
           circles=params[8:]
           blurSize=params[7]
        #print "circles",circles
        assert len(circles)%6 == 0
        for i in range(0,len(circles),6):
            #print "circle",circles[i:i+6]
            cv2.circle(img,(int(circles[0+i]),int(circles[1+i])),int(circles[5+i]),(float(circles[2+i]),float(circles[3+i]),float(circles[4+i])),-1)
    #        for row in range(max(0,int(circles[0+i]-circles[2+i])-3),min(img.shape[0],int(circles[0+i]+circles[2+i])+3)):
    #           for col in range(max(0,int(circles[1+i]-circles[2+i])-3),min(img.shape[1],int(circles[1+i]+circles[2+i])+3)):
    #               dr=circles[0+i]-row
    #               dc=circles[1+i]-col
    #               radius=np.sqrt(dr*dr+dc*dc)
    #               mantissa=bound( (circles[2+i]-radius)/2.+1,0,1)
    #               oneminus=1.-mantissa
    #               if mantissa>0:
    #                   for c in range(3):
    #                      img[row,col,c]=np.uint8(oneminus*img[row,col,c]+mantissa*circles[3+i+c])
        if not blurSize is None:
            img=cv2.blur(img,(blurSize,blurSize))
        return img


    def makeProgram(self,xopt):
        s=""
        if (len(xopt)-7)%6 == 0:
           circles=xopt[7:]
           blurSize=None
        else:
           circles=xopt[8:]
           blurSize=xopt[7]
        for i in range(0,len(circles),6):
            if i > 1:
                s+=","
            #print "s:",s
            #print tuple(xopt[i:i+6])
            s+="(%d,%d,%d,%d,%d,%d)"%tuple(circles[i:i+6])
        prg ="import cv2,numpy as n\n"
        prg+="z=n.ones((320,386,3),n.uint8)\n"
        prg+="z[:,:,:]=(%d,%d,%d)\n"%(xopt[3],xopt[2],xopt[1])
        prg+="z[%d:,:,:]=(%d,%d,%d)\n"%(xopt[0],xopt[6],xopt[5],xopt[4])
        prg+="for p,q,e,d,c,r in [%s]:\n"%s
        prg+=" cv2.circle(z,(p,q),r,(c,d,e),-1)\n"
        if not blurSize is None:
          prg+="cv2.imwrite('a.png',cv2.blur(z,(%d,%d)))\n"%(blurSize,blurSize)
        else:
          prg+="cv2.imwrite('a.png',z)\n"
        return prg


   
   
def integerOptimize1D(f,low,high,step):
    currentVal=np.inf
    for i in range(low,high,step):
        v=f(i)
        if v<currentVal:
            currentVal=v
            currentReturn=i
    return currentReturn,currentVal
    
def localopt(f,params,bounds,step):
    lparams=params[:]
    stepped=True
    cv=np.inf
    while stepped:
        stepped=False
        for i in range(len(lparams)):
           p=lparams[i]
           def loc(z):
               par=lparams[:]
               par[i]=z
               return f(par)
           l=max(bounds[i][0],p-5*step)
           h=min(bounds[i][1],p+5*step)
           p2,v2=integerOptimize1D(loc,l,h,step)
           if v2<cv:
               cv=v2
               stepped=True
               lparams[i]=p2
               if p2==l or p2==h:
                  params=lparams
                  break
    return lparams           

def compressProgram(prg):
    j=bz2.compress(prg,9)
    k=zlib.compress(prg,9)
    aj=base64.b64encode(j)
    ak=base64.b64encode(k)
    
    
    cprg= "import zlib,base64\n"
    cprg+="exec(zlib.decompress(base64.b64decode('%s')))\n"%ak


#        currentbest=f(params)
#    localp=params[:]
#    while stepped:
#       stepped=False
#       for p in range(len(params)):
#          localp=params[:]

def run(outdir,orig,params0,bounds,iterations):
    assert len(params0)==len(bounds)
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)
    anArtist=artist(orig)
    
    scorefile=open(outdir+"/scorefile.txt",'w')
    
    def optimizeme(params):
       img=anArtist.doit(params)
       s=score(img,orig)
       return s  
    
    
    #xopt=optimize.fmin_powell(optimizeme,params0,xtol=0.5)
    #res=optimize.differential_evolution(optimizeme,bounds)
    #xopt=res.x
    xopt=localopt(optimizeme,params0,bounds,5)
    xopt=localopt(optimizeme,xopt,bounds,1)
    print "found",xopt
    print "-------------------------------------"
    for q in range(iterations):
        img=anArtist.doit(xopt)
        z=np.sum((img.astype(np.float32)-orig.astype(np.float32))**2,2)
        z=z/np.max(z)
        z=filters.gaussian_filter(z,5,mode='constant')
    
        idx=np.unravel_index(np.argmax(z),z.shape)
        params0=list(xopt)
        params0.append(idx[1])
        bounds.append((idx[1]-20,idx[1]+20))
        params0.append(idx[0])
        bounds.append((idx[0]-20,idx[0]+20))
        params0.append(orig[idx[0],idx[1],0])
        bounds.append((0,255))
        params0.append(orig[idx[0],idx[1],1])
        bounds.append((0,255))
        params0.append(orig[idx[0],idx[1],2])
        bounds.append((0,255))
        params0.append(10)
        bounds.append((6,50))
        xopt=localopt(optimizeme,params0,bounds,5)
        xopt=localopt(optimizeme,xopt,bounds,1)
        #print "Found",xopt
        print "-------------------------------------"
        #xopt=optimize.fmin_powell(optimizeme,params0,xtol=0.5)
        #res=optimize.differential_evolution(optimizeme,bounds)
        #xopt=res.x
        prg=anArtist.makeProgram(xopt)
        print prg
        fd=open(outdir+"/draw_%0d.py"%(q+1),'w')
        print >>fd,prg,
        fd.close()
        
        j=bz2.compress(prg,9)
        k=zlib.compress(prg,9)
        aj=base64.b64encode(j)
        ak=base64.b64encode(k)
        
        
        cprg= "import zlib,base64\n"
        cprg+="exec(zlib.decompress(base64.b64decode('%s')))\n"%ak
        fd=open(outdir+"/cdraw_%0d.py"%(q+1),'w')
        print >>fd,cprg,
        fd.close()
        print "Program is",len(prg),len(cprg)
        print "Compressed zlib:",len(k),len(ak)
        print "Compressed bzip:",len(j),len(aj)
        
        img=anArtist.doit(xopt)
        s=score(img,orig)
        print "%d circles score: %f"%(q+1,s)
        print >>scorefile,q+1,s
        cv2.imwrite(outdir+'/test_%0d.png'%(q+1),img)




orig=cv2.imread('ORIGINAL.png')
mval=np.mean(orig,(0,1))
params0=[orig.shape[0]//2,int(mval[0]),int(mval[1]),int(mval[2]),int(mval[0]),int(mval[1]),int(mval[2]), 5]
bounds=[(1,orig.shape[0]),(0,255),(0,255),(0,255),(0,255),(0,255),(0,255),(1,10)]

run("outwithblur",orig,params0,bounds,120)

params0=[orig.shape[0]//2,int(mval[0]),int(mval[1]),int(mval[2]),int(mval[0]),int(mval[1]),int(mval[2])]
bounds=[(1,orig.shape[0]),(0,255),(0,255),(0,255),(0,255),(0,255),(0,255)]
run("outnoblur",orig,params0,bounds,120)



#plt.imshow(img[:,:,(2,1,0)])

#plt.show()

    