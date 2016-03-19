# -*- coding: utf-8 -*-
"""
Created for code golf starry night
http://codegolf.stackexchange.com/questions/69930/paint-starry-night-objectively-in-1kb-of-code

Rupert Brooks, 15-FEB-2016

The objective was to use an optimization approach, using drawing primitives, 
not using a compressed or encoded version of the original image.  It seems
that using image compression algorithms inevitably comes up with a better
score (not surprising really).  However, it is aesthetically more interesting
in some way to do it with drawing primitives.

However, while avoiding off the shelf image compression, it uses compression 
of the program code itself.  So off the shelf compression does figure in some 
sense.

The drawing primitives are circles dragged between two points.  This looks 
vaguely like brushstrokes, but was fairly easy to implement in a small number
of bytes.

A minimal amount of tuning to the Starry night picture was done.

It starts out finding a horizon line, and then one circle is placed by hand.
It really helps for this image to place a circle in this large dark area 
early, but this is not a general approach.  The rest of the circles are 
found by automated search.

The optimization process is a pattern search, not a genetic algorithm as
was used in many other entries.  The space being searched is integer, not
continuous.  Intriguingly, searching in the compressed color space 
significantly reduced the effectiveness of the approach, so the search is 
done in RGB space, but the drawing process reduces the colors when it draws.

The optimization goes past the final size required, and then removes low value
primitives to get down to the necessary size.  This helps, because it has not
necessarily found them in the most effective order. 

Some really interesting ideas from other posts were adopted here
- use of a blur to push the final score up a bit
- use of binary code directly in the final python output
- use of a 2D subspace of the color space
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy import signal
from skimage import filters
from skimage.morphology import disk as mcircle
from skimage.filters import rank
import bz2
import zlib
import base64
import sys
import os
import shutil
import time
import re
import struct

def score(img,orig):
    z=(img.astype(np.float32)-orig.astype(np.float32)).ravel()**2
    return np.sum(z)/255./255.

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
    # give a very poor score to circles that fall off the image, 
    # prevent the optimizer optimizing away the circle entirely
    if (len(params)-7)%8 == 0:
       circles=params[7:]
       blurSize=None
    else:
       circles=params[8:]
       blurSize=params[7]
    assert len(circles)%8 == 0
    for i in range(0,len(circles),8):
        c1=(circles[0+i],circles[1+i])
        c2=(c1[0]+circles[2+i],c1[1]+circles[3+i])
        d=max(distanceFromBoxSquared(c1,(0,0),(orig.shape[1]-1,orig.shape[0]-1)),
              distanceFromBoxSquared(c2,(0,0),(orig.shape[1]-1,orig.shape[0]-1)))
        r=circles[6]
        if d>r*r:
            return 1000
    return 0
            
class artist:
    # does the heavy lifting
    def __init__(self,img):
        self.shape=img.shape
        self.dtype=img.dtype
        self.cache=dict()
        # create color subspace using PCA
        C=np.cov(orig.reshape(orig.shape[0]*orig.shape[1],3).T)
        eigenvalues,eigenvectors=np.linalg.eig(C)
        repack=sorted([(eigenvalues[i],eigenvectors[:,i]) for i in range(3) ],key=lambda x:-x[0],)
        R=np.array([v[1] for v in repack])
        self.mtx=R[0:2,:]
        self.meancolor=np.round(np.mean(img,(0,1)))
        meanoutimage=img.astype(np.int)-self.meancolor
        testimage=np.dot(self.mtx,(meanoutimage.reshape(img.shape[0]*img.shape[1],3).T*1.)).T
        
        # this scales and offsets the subspace so that it fits in integers 0-99
        self.org=np.array([[np.min(testimage[:,0]),np.min(testimage[:,1])]])
        self.scl=np.array([255./(np.max(testimage[:,0])-np.min(testimage[:,0])),255./(np.max(testimage[:,1])-np.min(testimage[:,1]))])

        scaledimage=np.round((testimage-self.org)*self.scl)
        assert(np.max(scaledimage)<256)
        assert(np.min(scaledimage)>=0)

        self.roff=np.round(np.dot(R.T[:,0:2],self.org.reshape(2,1))+self.meancolor.reshape(3,1)).astype(np.int).reshape(3)

        tmp=R.T[:,0:2].copy()
        tmp[:,0]/=self.scl[0]
        tmp[:,1]/=self.scl[1]
        self.rmtx=np.round(tmp,1)  #).astype(np.int)
        # empirical testing showed 1 decimal place was enough

    def toColorSubspace(self,color):
        assert(len(color)==3)
        z=np.round((np.dot(self.mtx,(color-self.meancolor).reshape(3,1))-self.org)*self.scl)
        return (np.clip(z[0,0],0,255),np.clip(z[1,0],0,255))
    def fromColorSubspace(self,color): 
        assert(len(color)==2)
        return np.clip(np.dot(self.rmtx,np.array(color).reshape(2,1)).reshape(3)+self.roff,0,255)

    def doit(self,params):
        #print params
        # params 0, split level
        # params 1,2,3 top color
        # params 4,5,6 bottom color
        # either blur, or not
        # rest are circle paths, of length 8 each
        #
        # Use caching to try and speed this up
        if -1 in self.cache and self.cache[-1][0] == params[0:8]:
            img=self.cache[-1][1]
        else:
            self.cache.clear()
            img=np.zeros(self.shape,self.dtype)
            for c in range(3):
               si=params[0] #np.floor(s)
               #mantissa=s-si
               img[:si,:,c]=params[c+1]
               img[si:,:,c]=params[c+4]
            self.cache[-1]=(params[0:8],img.copy())
        # a circle path is x, y, offsetx, offsety, radius, r,g,b
        if (len(params)-7)%8 == 0:
           circles=params[7:]
           blurSize=None
        else:
           circles=params[8:]
           blurSize=params[7]
        assert len(circles)%8 == 0
        for i in range(0,len(circles),8):
            circ=circles[i:i+8]
            if i in self.cache and self.cache[i][0]==circ:
                img=self.cache[i][1]
            else:
                for k in self.cache.keys():
                    if k>=i:
                        del self.cache[k]
                img=img.copy()
                # its slightly fewer bytes if we drop the negative signs on
                # the offsets
                if circ[2]<0 and circ[3]<0:
                    circ[0]+=circ[2]
                    circ[1]+=circ[3]
                    circ[2]=-circ[2]
                    circ[3]=-circ[3]
                cv2.circle(img,(circ[0],circ[1]),circ[7],self.fromColorSubspace(self.toColorSubspace(circ[4:7])),-1)
                for j in range(31):
                    r=circ[0]+circ[2]*j//30 
                    c=circ[1]+circ[3]*j//30 
                    cv2.circle(img,(r,c),circ[7],self.fromColorSubspace(self.toColorSubspace(circ[4:7])),-1)
                self.cache[i]=(circ,img.copy())
        if not blurSize is None:
            return cv2.blur(img,(blurSize,blurSize))
        return img.copy()

    def makeProgram(self,xopt):
        # make a program that draws exactly as the above
        # in a string
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
            circ=circles[i:i+8]
            if circ[2]<0 and circ[3]<0:
                circ[0]+=circ[2]
                circ[1]+=circ[3]
                circ[2]=-circ[2]
                circ[3]=-circ[3]
            c=self.toColorSubspace(circ[4:7])
            s+="(%d,%d,%d,%d,%d,%d,%d)"%(circ[0],circ[1],circ[2],circ[3],c[0],c[1],circ[7])
        prg ="import cv2 as v,numpy as n\n"
        prg+="z=n.ones((320,386,3),'u1')\n"
        prg+="z[:]=%d,%d,%d\n"%(xopt[1],xopt[2],xopt[3])
        prg+="z[%d:]=%d,%d,%d\n"%(xopt[0],xopt[4],xopt[5],xopt[6])
        prg+="for p,q,x,y,c,d,r in [%s]:\n"%s
        prg+=" for k in range(31):\n"
        prg+="  v.circle(z,(p+x*k/30,q+y*k/30),r,n.clip((%.1f*c%+.1f*d%+d,%.1f*c%+.1f*d%+d,%.1f*c%+.1f*d%+d),0,255),-1)\n"%(self.rmtx[0,0],self.rmtx[0,1],self.roff[0],self.rmtx[1,0],self.rmtx[1,1],self.roff[1],self.rmtx[2,0],self.rmtx[2,1],self.roff[2])
        if not blurSize is None:
          prg+="v.imwrite('a.png',v.blur(z,(%d,%d)))\n"%(blurSize,blurSize)
        else:
          prg+="v.imwrite('a.png',z)\n"
        return prg

# exhaustive line search over integers
def integerOptimize1D(f,low,high,step):
    currentVal=np.inf
    currentReturn=low
    for i in range(int(low),int(high),int(step)):
        v=f(i)
        if v<currentVal:
            currentVal=v
            currentReturn=i
    return currentReturn,currentVal

# search through parameters one by one, performing 1D optimize for each one
# authorized by include, within limits of bounds     
def localopt(f,params,bounds,include,step):
    #print "Optimizing ",len(params),"parameters"
    assert(len(bounds)==len(params))
    assert(len(include)==len(params))
    lparams=params[:]
    stepped=True
    cv=np.inf
    iters=0
    while stepped:
        iters+=1
        stepped=False
        for i in range(len(lparams)):
           if include[i]:
               p=lparams[i]
               def loc(z):
                   par=lparams[:]
                   par[i]=z
                   return f(par)
               l=max(bounds[i][0],p-3*step)
               h=min(bounds[i][1],p+3*step)
               p2,v2=integerOptimize1D(loc,l,h,step)
               if v2<cv:
                   cv=v2
                   stepped=True
                   lparams[i]=p2
                   if p2==l or p2==h:
                      params=lparams
                      break
    #print "Required",iters,"iteratiosn"             
    return lparams           

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
        return "\\x8e"
    else:
        return b
def packit(s):
    #z=[escape(b) for b in s]
    #print "elem len",sum(map(len,z))
    z=bytearray("".join([escape(b) for b in s]))
    z=re.sub(r'\\0([0-9])',r'\\000\1',z)
    return z
def compressProgram(prg):
   dataAsText=re.search('\\[\\(.+\\)\\]',prg)
   if dataAsText is None:
       return prg

   dataAsCode="dataAsList="+dataAsText.group(0)
   
   exec(dataAsCode)
   dataAsPacked=bytearray("")
   for d in dataAsList:
       try:
          dataAsPacked+=struct.pack("HHBBB",d[0]+31+(d[2]+31)*512,d[1]+31+(d[3]+31)*512,d[4],d[5],d[6])
       except:
          print "bad datum",d
          print "becomes",d[0]+31+(d[2]+31)*512,d[1]+31+(d[3]+31)*512,d[4],d[5],d[6]
          raise Exception("cantpack")
   global packcount
   packcount=0   
   dataAsEscaped=packit(str(dataAsPacked))
   #print len(dataAsPacked),len(dataAsEscaped),packcount
   #for i in range(len(dataAsPacked)):
   #    print "%03d "%(dataAsPacked[i]),
   #    if i%10==9:
   #        print ""
   #print ""
   #exec(b"dataTest='"+dataAsEscaped+"'")
   #assert dataTest==dataAsPacked
   cprg= bytearray(b'\xEF\xBB\xBF')
   for l in prg.split('\n'):
       if re.match("import",l):
         cprg+=l
         cprg+=",struct\n"
       elif re.match('for p,q',l):
         cprg+="s='"
         #z=len(cprg)
         cprg+=dataAsEscaped
         #print len(cprg)-z
         cprg+="'\n"
         #cprg+="for i in range(len(s)):\n"
         #cprg+="       print '%03d '%ord(s[i]),\n"
         #cprg+="       if i%10==9:\n"
         #cprg+="           print \"\"\n"
         #cprg+="print \"\\n\",len(s)\n"
         cprg+="m,h=512,31\n"
         cprg+="for a,b,c,d,r in[struct.unpack('HHBBB',s[7*i:7*i+7])for i in range(%d)]:\n"%(len(dataAsPacked)/7)
         #cprg+=" p=a%m-50\n"
         #cprg+=" q=b%m-50\n"
         cprg+=" x,y=a/m-h,b/m-h\n"
       elif re.search("range\\(31\\)",l):
           l=re.sub("31","h",l)
           cprg+=l#+'\n'
       elif re.search("p\\+x",l):
           l=re.sub('^ +',"",l)
           l=re.sub('p\\+',"a%m-h+",l)
           l=re.sub('q\\+',"b%m-h+",l)
           l=re.sub('1.0\\*',"",l)
           l=re.sub('([^0-9])0(\.[0-9])','\\1\\2',l)
           cprg+=l+'\n'
       else:
           cprg+=l+'\n'
   return cprg[:-2] # two newlines...

filtersize=3
# perform the optimization, up to iterations number of circles
# store the results in outdir
def searchForParameters(outdir,orig,params0,bounds,iterations):    
    assert len(params0)==len(bounds)
    global filtersize
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)
    anArtist=artist(orig)    
    scorefile=open(outdir+"/scorefile.txt",'w')
    
    def optimizeme(params):
       img=anArtist.doit(params)
       s=score(img,orig)+regularizer(params,orig)
       return s  
    
    include=[True,]*len(params0)
    #xopt=localopt(optimizeme,params0,bounds,include,5)
    #xopt=localopt(optimizeme,xopt,bounds,include,1)
    xopt=params0[:]
    #print "found",xopt
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
        f=filtersize
        while itercount<10:
            img=anArtist.doit(xopt)
            err=img.astype(np.float32)-orig.astype(np.float32)
            z=np.zeros_like(err)
            z2=np.zeros_like(err)
            kernel=mcircle(f,np.float)
            for i in range(3):
               z[:,:,i]=signal.convolve2d(err[:,:,i],kernel,'same')
               z2[:,:,i]=signal.convolve2d(err[:,:,i]**2,kernel,'same')
            qm=z/np.sum(kernel)   
            qs=z2/np.sum(kernel)-qm**2 +1
            #print np.min(qs),np.max(qs)
            z=np.sum(np.abs(qm/np.sqrt(qs)),2)
            z=z/np.max(z)
#            shift=np.min(z)
#            scale=np.max(z)-shift
#            z=z-shift
#            z=z/scale
            cv2.imwrite(outdir+'/error_%03d_prefilter.png'%(q+2),(err-np.min(err))/(np.max(err)-np.min(err))*255)
            #z=np.sum(np.abs(filters.gaussian_filter(z,filtersize,mode='constant',cval=-shift/scale)*scale+shift)**2,2)
            #z=z/np.max(z)
            cv2.imwrite(outdir+'/error_%03d_postfilter.png'%(q+2),z*255)
            idx0=np.unravel_index(np.argmax(z),z.shape)
            
            params0=xopt[:]

            include=[False,]*len(params0)

            params0.append(idx0[1])
            bounds.append((idx0[1]-30,idx0[1]+30))
            include.append(True)
            params0.append(idx0[0])
            bounds.append((idx0[0]-30,idx0[0]+30))
            include.append(True)
            params0.append(0)
            bounds.append((-30,+30))
            include.append(True)
            params0.append(0)
            bounds.append((-30,+30))
            include.append(True)
            c=orig[idx0[0],idx0[1],:]
            params0.append(float(c[0]))
            bounds.append((0,255))
            include.append(True)
            params0.append(float(c[1]))
            bounds.append((0,255))
            include.append(True)
            params0.append(float(c[2]))
            bounds.append((0,255))
            include.append(True)
            params0.append(11)
            bounds.append((1,50))
            include.append(True)
            xopt0=localopt(optimizeme,params0,bounds,include,5)
            xopt=localopt(optimizeme,xopt0,bounds,include,1)
            
            #xopt=localopt(optimizeme,xopt0,bounds,[True,]*len(bounds),1)
    
            # check if this actually did anything useful
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
#                print "Full params were",xopt
                xopt=xtry
                bounds=bounds[:-8]
                f=f-1
                if f<1: 
                    f=10

        #print "Found",xopt
        print "-------------------------------------"
        prg=anArtist.makeProgram(xopt)
        fd=open(outdir+"/draw_%0d.py"%(q+2),'w')
        print >>fd,prg,
        fd.close()

        cprg=compressProgram(prg)
        fd=open(outdir+"/cdraw_%0d.py"%(q+2),'w')
        print >>fd,cprg,
        fd.close()
        print "Program is",len(prg),len(cprg),cprg[10:14]
        
        print "At",time.asctime()
        print "%d circles score: %f"%(q+2,s)
        print >>scorefile,q+2,s,len(prg),len(cprg),cprg[10:14]
        scorefile.flush()
        cv2.imwrite(outdir+'/test_%03d.png'%(q+2),img)
        if s>=lastscore:
           print "Score has not improved, stopping"
           break
    return xopt,bounds

sizelimit=1024
orig=cv2.imread('ORIGINAL.png')
pfd=open("foundparamsv14final_3.txt")
#foundparamsv13final_%d.txt
xstr=pfd.readline()
bstr=pfd.readline()
pfd.close()
#exec("x="+xstr)
#exec("b="+bstr)
exec("params="+xstr)
exec("bounds="+bstr)
anArtist=artist(orig)
z=anArtist.doit(params)
s=score(z,orig)
#prg=anArtist.makeProgram(params)
#zprg=compressProgram(prg)
#print s,len(prg),len(zprg)
#fd=open("draw_test.py",'w')
#print >>fd,prg,
#fd.close()
#fd=open("cdraw_test.py",'w')
#print >>fd,zprg,
#fd.close()


def optimizeme(p):
   img=anArtist.doit(p)
   s=score(img,orig)+regularizer(p,orig)
   return s  
params=localopt(optimizeme,params,bounds,[True,]*len(bounds),1)
z=anArtist.doit(params)
s2=score(z,orig)
print "Started with score",s,"optimized to",s2
pfd=open("foundparamsv15_first.txt",'w')
print >>pfd,params
print >>pfd,bounds
pfd.close()


#mval=np.mean(orig,(0,1))
##
##params0=[orig.shape[0]//2,int(mval[0]),int(mval[1]),int(mval[2]),int(mval[0]),int(mval[1]),int(mval[2]), 5, 94, 267, 0, -10, 43, 43, 38, 30]
##bounds=[(1,orig.shape[0]),(0,255),(0,255),(0,255),(0,255),(0,255),(0,255),(1,10), (94-20,94+20),(267-20,267+20),(-30,30),(-30,30),(0,255),(0,255),(0,255),(3,50)]
##params0=[231, 149, 107, 79, 56, 49, 43, 9, 103, 270, -30, -30, 95, 0, 0, 49, 347, 49, 3, 12, 160, 156, 145, 35, 63, 180, -1, -30, 94, 0, 0, 23, 384, 178, -29, 3, 115, 0, 0, 16, 67, 122, -1, -30, 80, 4, 0, 8, 338, 156, 29, -13, 233, 114, 135, 19, 129, 168, -2, 3, 219, 173, 106, 23, 59, 57, 16, -29, 173, 0, 0, 26, 268, 208, -30, 10, 162, 0, 0, 16, 330, 196, -30, 10, 162, 5, 5, 11, 82, 55, -6, 1, 107, 124, 145, 11, 39, 151, 0, 7, 205, 18, 215, 11, 27, 14, 5, 1, 117, 118, 178, 8, 38, 225, -22, 2, 230, 74, 26, 12, 275, 76, -9, -4, 207, 98, 135, 12, 299, 183, 24, -24, 229, 85, 153, 12, 89, 14, 22, 28, 193, 30, 0, 20, 248, 187, 28, 2, 243, 54, 157.0, 11, 239, 27, -12, 1, 115, 112, 171, 7, 99, 117, 0, 10, 5, 0, 5, 4, 190, 238, -30, 2, 163, 11, 4, 21, 267, 39, 10, -29, 205, 25, 0, 18, 186, 34, 9, -29, 216, 10, 10, 23, 335, 254, 11, -8, 185, 15, 66, 18, 339, 66, 25, 6, 63, 117, 191, 6, 16, 192, 19, 16, 220, 113, 107, 9, 118, 103, 0, 3, 156, 146, 198, 6, 0, 37, 1, -30, 185, 5, 0, 10, 51, 6, -3, -6, 0, 0, 0, 3, 63, 1, -10, 10, 0, 0, 0, 1, 72, 9, 14, 5, 130, 250, 242, 1, 148, 87, -13, 10, 131, 0, 0, 6, 131, 36, 29, 3, 220, 0, 17, 11, 244, 67, -30, -18, 199, 15, 0, 8, 242, 1, -13, -1, 186, 15, 0, 12, 13, 71, 29, 13, 0, 0, 0, 2, 220, 230, 29, 12, 199, 0, 56, 22, 293, 238, 29, 2, 188, 0, 56, 15, 151, 22, 3, -1, 112, 134, 138, 6, 317, 288, 2, 0, 177, 2, 70, 9, 153, 197, -30, 13, 220, 20, 159.0, 8, 213, 191, 17, -4, 217, 71, 135.0, 7, 118, 127, -30, 12, 203, 20, 0, 8, 9, 120, 29, 3, 246, 112, 77, 10, 129, 21, -6, -11, 0, 170, 212, 1, 49, 52, -26, 2, 229, 32, 0, 21, 196, 195, -30, 3, 184, 113, 100, 9, 4, 144, 0, 1, 215, 142.0, 88, 8, 279, 311, -13, 8, 171, 0, 77, 10, 125, 6, -11, 17, 204, 179, 112, 2, 131, 9, -11, -1, 161, 154, 39, 5, 145, 9, 23, -4, 152, 0, 0, 2, 186, 214, -10, -5, 237, 0, 95, 11, 209, 269, -15, -4, 162, 0, 58, 10, 51, 131, -4, -15, 4, 0, 0, 2, 153, 62, 29, -3, 234, 5, 147.0, 10, 127, 168, 2, 2, 119, 174, 120, 5, 103, 165, 17, -8, 177, 174, 105, 1, 114, 168, 1, -9, 177, 161, 105, 1, 110, 155, -5, -3, 183, 169, 98, 1, 94, 198, 5, 29, 248, 0, 75, 1, 99, 191, -4, -8, 186, 31, 28, 11, 344, 41, -8, 17, 65, 132, 197, 5, 162, 303, -16, 4, 40, 0, 5, 11, 221, 106, -7, 8, 240, 126, 52, 11, 238, 107, 28, -1, 142, 0, 0, 1, 144, 128, 22, 15, 254, 22, 128, 12, 184, 2, -7, -2, 223, 5, 105, 11, 330, 184, -30, 7, 237, 63, 112, 6, 29, 254, -29, -8, 151, 0, 7, 12, 12, 166, 21, 12, 224, 69, 0, 13, 174, 170, -10, -15, 115, 0, 0, 2, 32, 95, 20, 0, 243, 46, 98.0, 11, 378, 257, -2, -22, 168, 0, 84, 12, 374, 259, 2, -2, 189, 0, 59, 11, 346, 257, -6, -30, 247, 54, 124, 1, 359, 254, 4, -18, 174, 73, 139, 1, 372, 256, 2, -23, 195, 23, 112, 1, 354, 285, 25, -5, 251, 55, 104, 1, 103, 89, -20, 0, 254, 5, 142, 12, 382, 128, -30, 20, 237, 73, 196.0, 9, 61, 63, 4, -30, 93, 0, 12, 4, 72, 74, 29, -6, 239, 25, 16, 5, 13, 319, 29, 0, 129, 33, 55, 12, 341, 297, -9, -1, 171, 0, 50, 11, 306, 14, -30, -2, 208, 58, 0, 11, 221, 167, 0, 1, 197, 0, 14, 6, 357, 24, 15, 10, 184, 143, 141.0, 12, 249, 126, 29, 2, 239, 11, 127.0, 6, 254, 158, 26, -4, 251, 5, 137.0, 13, 19, 89, -4, -2, 225, 102, 73, 10, 221, 66, -24, 2, 252, 0, 120, 11, 171, 105, -22, -4, 253, 0, 130, 11, 316, 64, 5, -30, 201, 158, 107, 9, 137, 75, -15, 7, 245, 18, 125, 11, 340, 169, 9, 2, 237, 86, 68, 6, 186, 212, 1, -17, 225, 23, 135, 2, 171, 236, 29, 12, 241, 85, 130, 1, 293, 275, 8, 17, 166, 0, 45, 10, 27, 291, -2, 11, 125, 15, 47.0, 13, 94, 108, -16, 7, 253, 251, 102, 1]
##bounds=[(1, 320), (0, 255), (0, 255), (0, 255), (0, 255), (0, 255), (0, 255), (1, 10), (74, 114), (247, 287), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (3, 50), (309, 369), (28, 88), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (39, 99), (122, 182), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (338, 398), (145, 205), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (37, 97), (77, 137), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (331, 391), (114, 174), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (99, 159), (151, 211), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (35, 95), (5, 65), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (237, 297), (183, 243), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (290, 350), (172, 232), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (51, 111), (26, 86), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (11, 71), (124, 184), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (-1, 59), (-16, 44), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (13, 73), (199, 259), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (240, 300), (44, 104), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (276, 336), (143, 203), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (68, 128), (-20, 40), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (227, 287), (153, 213), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (202, 262), (-3, 57), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (69, 129), (94, 154), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (138, 198), (194, 254), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (232, 292), (9, 69), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (166, 226), (6, 66), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (314, 374), (221, 281), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (315, 375), (38, 98), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (-9, 51), (168, 228), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (87, 147), (74, 134), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (-24, 36), (9, 69), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (49, 109), (-20, 40), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (49, 109), (-20, 40), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (49, 109), (-20, 40), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (108, 168), (62, 122), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (117, 177), (8, 68), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (211, 271), (33, 93), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (211, 271), (-25, 35), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (-2, 58), (45, 105), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (197, 257), (216, 276), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (278, 338), (217, 277), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (122, 182), (-9, 51), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (288, 348), (258, 318), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (122, 182), (167, 227), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (190, 250), (158, 218), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (81, 141), (102, 162), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (2, 62), (93, 153), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (95, 155), (-17, 43), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (15, 75), (28, 88), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (161, 221), (163, 223), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (-25, 35), (114, 174), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (248, 308), (280, 340), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (122, 182), (-23, 37), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (122, 182), (-23, 37), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (122, 182), (-23, 37), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (153, 213), (212, 272), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (180, 240), (238, 298), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (20, 80), (93, 153), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (124, 184), (31, 91), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (68, 128), (155, 215), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (68, 128), (155, 215), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (68, 128), (155, 215), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (68, 128), (155, 215), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (68, 128), (155, 215), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (68, 128), (156, 216), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (310, 370), (17, 77), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (116, 176), (278, 338), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (191, 251), (76, 136), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (212, 272), (75, 135), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (115, 175), (98, 158), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (154, 214), (-25, 35), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (300, 360), (153, 213), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (2, 62), (225, 285), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (-2, 58), (151, 211), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (137, 197), (134, 194), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (28, 88), (66, 126), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (332, 392), (254, 314), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (332, 392), (254, 314), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (332, 392), (254, 314), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (332, 392), (254, 314), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (332, 392), (254, 314), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (332, 392), (254, 314), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (72, 132), (58, 118), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (352, 412), (93, 153), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (43, 103), (41, 101), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (42, 102), (41, 101), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (-17, 43), (287, 347), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (309, 369), (272, 332), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (277, 337), (-12, 48), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (191, 251), (137, 197), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (301, 361), (-8, 52), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (221, 281), (97, 157), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (220, 280), (124, 184), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (-16, 44), (59, 119), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (193, 253), (35, 95), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (141, 201), (75, 135), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (280, 340), (32, 92), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (111, 171), (46, 106), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (313, 373), (137, 197), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (153, 213), (212, 272), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (153, 213), (212, 272), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (265, 325), (249, 309), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (-4, 56), (248, 308), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50), (54, 114), (79, 139), (-30, 30), (-30, 30), (0, 255), (0, 255), (0, 255), (1, 50)]
#for i in range(len(bounds)):
    
#x,b=searchForParameters("outputv10",orig,params0,bounds,10)

#pfd=open("foundparamsv10.txt",'w')
#print >>pfd,x
#print >>pfd,b
#pfd.close()


anArtist.doit(x)
for f in [1,3]:
    filtersize=f
    for tries in range(2):
        x=params[:]
        x,b=searchForParameters("outputv15_%d_%d"%(tries,f),orig,params,bounds,50)
        baseparams=x[:8]
        circles=[x[i:i+8] for i in range(8,len(x),8)]
        basebounds=b[:8]
        bcircle=[b[i:i+8] for i in range(8,len(b),8)]
        deltas=[]
        lastscore=np.inf
        for i in range(len(circles)+1):
            params=baseparams[:]
            for c in range(i):
                params.extend(circles[c])
            z=anArtist.doit(params)
            s=score(z,orig)
            if i>0:
                deltas.append(lastscore-s)
            prg=anArtist.makeProgram(params)
            zprg=compressProgram(prg)
            print i,s,lastscore-s,len(prg),len(zprg)
            lastscore=s
        z=anArtist.doit(x)
        lastscore=score(z,orig)
        deltas2=[]
        for i in range(len(circles)):
            params=baseparams[:]
            for c in range(len(circles)):
                if c != i:
                    params.extend(circles[c])
            z=anArtist.doit(params)
            s=score(z,orig)
            deltas2.append(s-lastscore)
            prg=anArtist.makeProgram(params)
            zprg=compressProgram(prg)
            print i,s,deltas[i],s-lastscore,len(prg),len(zprg)
            
        print "--------------------------------------------------------------"
        includeCircle=[True]*len(circles)
        for i in range(len(circles)/2):
            r=np.argmin(deltas)
            print "removing circle",r,"diff",deltas[r]
            deltas[r]=999999999.
            includeCircle[r]=False
            params=baseparams[:]
            bounds=basebounds[:]
            for c in range(len(circles)):
                if includeCircle[c]:
                    params.extend(circles[c])
                    bounds.extend(bcircle[c])
            z=anArtist.doit(params)
            s=score(z,orig)
            prg=anArtist.makeProgram(params)
            zprg=compressProgram(prg)
            print i,s,lastscore-s,len(prg),len(zprg)
            lastscore=s
            if len(zprg) <= sizelimit:
                    break
                
        
    pfd=open("foundparamsv15_%d.txt"%f,'w')
    print >>pfd,params
    print >>pfd,bounds
    pfd.close()
    params=localopt(optimizeme,params,bounds,[True,]*len(bounds),1)
    pfd=open("foundparamsv15final_%d.txt"%f,'w')
    print >>pfd,params
    print >>pfd,bounds
    pfd.close()
    s=score(z,orig)
    prg=anArtist.makeProgram(params)
    zprg=compressProgram(prg)
    print "Achieved final score:",s
    print "In program of size:",len(zprg)
    if len(zprg) <= 1024:
        fd=open("draw_finalv15_%d.py"%f,'w')
        print >>fd,prg,
        fd.close()
        fd=open("cdraw_finalv15_%d.py"%f,'w')
        print >>fd,zprg,
        fd.close()
