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
import time





def score(img,orig):
    z=(img.astype(np.float32)-orig.astype(np.float32)).ravel()**2
    return np.sum(z)/255./255.

def bound(a,low,high):
    return max(min(a,high),low)
    
def distanceFromBoxSquared(p,b1,b2):
    dx=0
    dy=0
    if p[0]<b1[0]:
        dx=b1[0]-p[0]
    if p[0]>b2[0]:
        dx=max(dx,p[0]-b2[0])
    if p[1]<b1[1]:
        dy=b1[1]-p[1]
    if p[1]>b2[1]:
        dy=max(dy,p[1]-b2[1])
    return dx*dx+dy*dy
        

def regularizer(params,orig):
    if (len(params)-7)%8 == 0:
       circles=params[7:]
       blurSize=None
    else:
       circles=params[8:]
       blurSize=params[7]
    #print "circles",circles
    assert len(circles)%8 == 0
    for i in range(0,len(circles),8):
        c1=(circles[0+i],circles[1+i])
        c2=(c1[0]+circles[2+i],c1[1]+circles[3+i])
        d=max(distanceFromBoxSquared(c1,(0,0),(orig.shape[1]-1,orig.shape[0]-1)),
              distanceFromBoxSquared(c2,(0,0),(orig.shape[1]-1,orig.shape[0]-1)))
        r=circles[7]
        if d>r*r:
            return 1000
    return 0
            

class artist:
    def __init__(self,img):
        self.shape=img.shape
        self.dtype=img.dtype
        self.cache=dict()
        
    def doit(self,params):
        #print params
        # params 0, split level
        # params 1,2,3 top color
        # params 4,5,6 bottom color
        #s=bound(params[0],1,img.shape[0]-2)
        if -1 in self.cache and self.cache[-1][0] == params[0:8]:
            img=self.cache[-1][1]
        else:
            self.cache.clear()
            img=np.zeros(self.shape,self.dtype)
            for c in range(3):
               si=params[0] #np.floor(s)
               #mantissa=s-si
               img[:si,:,c]=params[c+1]
               img[si:,:,c]=params[c+2]
            self.cache[-1]=(params[0:8],img.copy())
               #img[si+1:,:,c]=c2
           #img[si,:,c]=c1*mantissa+c2*(1.-mantissa)
        # a circle is x, y, radius, r,g,b
        if (len(params)-7)%8 == 0:
           circles=params[7:]
           blurSize=None
        else:
           circles=params[8:]
           blurSize=params[7]
        #print "circles",circles
        assert len(circles)%8 == 0
        for i in range(0,len(circles),8):
            circ=circles[i:i+8]
            if i in self.cache and self.cache[i][0]==circ:
                img=self.cache[i][1]
            else:
            #print "circle",circles[i:i+8]
                for k in self.cache.keys():
                    if k>=i:
                        del self.cache[k]
                img=img.copy()
                cv2.circle(img,(circ[0],circ[1]),circ[7],(circ[4],circ[5],circ[6]),-1)
                for j in range(31):
                    r=circ[0]+circ[2]*j//30 
                    c=circ[1]+circ[3]*j//30 
                    cv2.circle(img,(r,c),circ[7],(circ[4],circ[5],circ[6]),-1)
                self.cache[i]=(circ,img.copy())
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
            return cv2.blur(img,(blurSize,blurSize))
        return img.copy()


    def makeProgram(self,xopt):
        s=""
        if (len(xopt)-7)%8 == 0:
           circles=xopt[7:]
           blurSize=None
        else:
           circles=xopt[8:]
           blurSize=xopt[7]
        for i in range(0,len(circles),8):
            if i > 1:
                s+=","
            #print "s:",s
            #print tuple(xopt[i:i+6])
            s+="(%d,%d,%d,%d,%d,%d,%d,%d)"%tuple(circles[i:i+8])
        prg ="import cv2,numpy as n\n"
        prg+="z=n.ones((320,386,3),n.uint8)\n"
        prg+="z[:,:,:]=(%d,%d,%d)\n"%(xopt[1],xopt[2],xopt[3])
        prg+="z[%d:,:,:]=(%d,%d,%d)\n"%(xopt[0],xopt[4],xopt[5],xopt[6])
        prg+="for p,q,x,y,c,d,e,r in [%s]:\n"%s
        prg+=" for k in range(31):\n"
        prg+="  cv2.circle(z,(p+x*k/30,q+y*k/30),r,(c,d,e),-1)\n"
        if not blurSize is None:
          prg+="cv2.imwrite('a.png',cv2.blur(z,(%d,%d)))\n"%(blurSize,blurSize)
        else:
          prg+="cv2.imwrite('a.png',z)\n"
        return prg


   
   
def integerOptimize1D(f,low,high,step):
    currentVal=np.inf
    currentReturn=low
    for i in range(int(low),int(high),int(step)):
        v=f(i)
        if v<currentVal:
            currentVal=v
            currentReturn=i
    return currentReturn,currentVal
    
def localopt(f,params,bounds,include,step):
    assert(len(bounds)==len(params))
    assert(len(include)==len(params))
    lparams=params[:]
    stepped=True
    cv=np.inf
    while stepped:
        stepped=False
        for i in range(len(lparams)):
           if include[i]:
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
    def escape(b):
        assert len(b)==1
        if b == "\0":
            return "\\0"
        elif b=='\\':
            return '\\\\'
        elif b=='\'':
            return "\\'"
        elif b=='\n':
            return "\\n"
        elif b=='\r':
            return "\\r"
        elif b==b'\x8e':
            return '\\x8e'
        else:
            return b
    def packit(s):
        return bytearray("".join([escape(b) for b in s]))

    if len(j)<len(k):
        cprg= bytearray(b'\xEF\xBB\xBF')
        cprg+=bytearray(b"import bz2\n")
        # obnoxious to debug
        #z=packit(j)
        #for i in range(10,len(z)+1):
        #    if z[i-1] != 92:
        #       cprg+="a='byte %d is %d,"%(i,z[i-1])+z[:i]+"'\n"
        cprg+=bytearray(b"exec(bz2.decompress('")+packit(j)+bytearray(b"'))\n")
    else:
        cprg= bytearray(b'\xEF\xBB\xBF')
        cprg+=bytearray(b"import zlib\n")
        cprg+=bytearray(b"exec(zlib.decompress('")+packit(k)+bytearray(b"'))\n")
    #print type(cprg)
    #if 0 in cprg:
    #    print "zero in cprg"
    #else:
    #    print "no zero in cprg"
    #for i in cprg[:75]:
    #    if i>32 and i<127:
    #        print int(i),chr(i)
    #    else:
    #        print int(i)
    return cprg

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
    #real image range
    imgmax=np.max(orig,(0,1))
    imgmin=np.min(orig,(0,1))

    
    #z=anArtist.doit(params0)
    #print z.shape
    #plt.imshow(z)
    #plt.title("first params")
    #plt.show()
    
    scorefile=open(outdir+"/scorefile.txt",'w')
    
    def optimizeme(params):
       img=anArtist.doit(params)
       s=score(img,orig)+regularizer(params,orig)
       return s  
    
    
    #xopt=optimize.fmin_powell(optimizeme,params0,xtol=0.5)
    #res=optimize.differential_evolution(optimizeme,bounds)
    #xopt=res.x
    include=[True,]*len(params0)
    xopt=localopt(optimizeme,params0,bounds,include,5)
    xopt=localopt(optimizeme,xopt,bounds,include,1)
    print "found",xopt
    img=anArtist.doit(xopt)
    cv2.imwrite(outdir+'/test_001.png',img)
    s=score(img,orig)
    print "At",time.asctime()
    print "0 circles score: %f"%(s)
    print >>scorefile,0,s
    print "-------------------------------------"
    for q in range(iterations):
        lastscore=s
        itercount=0
        filtersize=5
        while itercount<10:
            img=anArtist.doit(xopt)
            z=img.astype(np.float32)-orig.astype(np.float32)
            shift=np.min(z)
            scale=np.max(z)-shift
            z=z-shift
            z=z/scale
            cv2.imwrite(outdir+'/error_%03d_prefilter.png'%(q+2),z*255)
            z=np.sum(np.abs(filters.gaussian_filter(z,filtersize,mode='constant',cval=-shift/scale)*scale+shift)**2,2)
            z=z/np.max(z)
            cv2.imwrite(outdir+'/error_%03d_postfilter.png'%(q+2),z*255)
            
            z=np.sum((img.astype(np.float32)-orig.astype(np.float32))**2,2)
            z=z/np.max(z)
            z=filters.gaussian_filter(z,filtersize,mode='constant')
            cv2.imwrite(outdir+'/error_%03d_oldfilter.png'%(q+2),z*255)
        
            idx=np.unravel_index(np.argmax(z),z.shape)
            params0=list(xopt)
            include=[False,]*len(params0)
            params0.append(idx[1])
            bounds.append((idx[1]-30,idx[1]+30))
            include.append(True)
            params0.append(idx[0])
            bounds.append((idx[0]-30,idx[0]+30))
            include.append(True)
            params0.append(0)
            bounds.append((-30,+30))
            include.append(True)
            params0.append(0)
            bounds.append((-30,+30))
            include.append(True)
            params0.append(float(orig[idx[0],idx[1],0]))
            bounds.append((imgmin[0],imgmax[0]))
            include.append(True)
            params0.append(float(orig[idx[0],idx[1],1]))
            bounds.append((imgmin[1],imgmax[1]))
            include.append(True)
            params0.append(float(orig[idx[0],idx[1],2]))
            bounds.append((imgmin[2],imgmax[2]))
            include.append(True)
            params0.append(6)
            bounds.append((1,50))
            include.append(True)
            xopt=localopt(optimizeme,params0,bounds,include,5)
            xopt=localopt(optimizeme,xopt,bounds,[True,]*len(bounds),1)
    
            
            xtry=xopt[:-8]
            img=anArtist.doit(xtry)
            stry=score(img,orig)
            img=anArtist.doit(xopt)
            s=score(img,orig)
            if stry > s:
                break
            else:
                itercount+=1
                print "Rejecting last circle as it is useless"
                print "score with",s
                print "score without",stry
                print "Full params were",xopt
                xopt=xtry
                bounds=bounds[:-8]
                filtersize=filtersize-1
                if filtersize<1: 
                    filtersize=10

        print "Found",xopt
        print "-------------------------------------"
        #xopt=optimize.fmin_powell(optimizeme,params0,xtol=0.5)
        #res=optimize.differential_evolution(optimizeme,bounds)
        #xopt=res.x
        prg=anArtist.makeProgram(xopt)
#        print prg
        fd=open(outdir+"/draw_%0d.py"%(q+1),'w')
        print >>fd,prg,
        fd.close()

        cprg=compressProgram(prg)
        fd=open(outdir+"/cdraw_%0d.py"%(q+1),'w')
        print >>fd,cprg,
        fd.close()
        print "Program is",len(prg),len(cprg),cprg[10:14]
        
        print "At",time.asctime()
        print "%d circles score: %f"%(q+1,s)
        print >>scorefile,q+1,s,len(prg),len(cprg),cprg[10:14]
        cv2.imwrite(outdir+'/test_%03d.png'%(q+2),img)
        if s>=lastscore:
           print "Score has not improved, stopping"
           break
    return xopt,bounds



orig=cv2.imread('ORIGINAL.png')
mval=np.mean(orig,(0,1))

params0=[orig.shape[0]//2,int(mval[0]),int(mval[1]),int(mval[2]),int(mval[0]),int(mval[1]),int(mval[2]), 5, 94, 267, 0, -10, 30, 43, 43, 38]
bounds=[(1,orig.shape[0]),(0,255),(0,255),(0,255),(0,255),(0,255),(0,255),(1,10), (94-20,94+20),(267-20,267+20),(-30,30),(-30,30),(3,50),(0,255),(0,255),(0,255)]

x,b=run("outwithblurbrushv5",orig,params0,bounds,120)


params0=[orig.shape[0]//2,int(mval[0]),int(mval[1]),int(mval[2]),int(mval[0]),int(mval[1]),int(mval[2]), 94, 267, 0, -10, 30, 43, 43, 38]
bounds=[(1,orig.shape[0]),(0,255),(0,255),(0,255),(0,255),(0,255),(0,255), (94-20,94+20),(267-20,267+20),(-30,30),(-30,30),(3,50),(0,255),(0,255),(0,255)]
run("outnoblurbrushv5",orig,params0,bounds,120)


run("outremoveblurbrushv5",orig,x[:7]+x[8:],b[:7]+b[8:],2)



#plt.imshow(img[:,:,(2,1,0)])

#plt.show()

    