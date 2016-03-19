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
pfd=open("foundparamsv16final_1.txt") #4882
pfd=open("foundparamsv15final_1.txt") #4776
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
print "Starting with score:",s

def optimizeme(p):
   img=anArtist.doit(p)
   s=score(img,orig)+regularizer(p,orig)
   return s  

for sizelimit in [1100,1024]:
    for f in [15,10,5,1]:
        filtersize=f
        for tries in range(2):
            x=params[:]
            x,b=searchForParameters("outputv17_%d_%d"%(tries,f),orig,params,bounds,25)
            baseparams=x[:8]
            circles=[x[i:i+8] for i in range(8,len(x),8)]
            basebounds=b[:8]
            bcircle=[b[i:i+8] for i in range(8,len(b),8)]
            print "--------BUBBLESORT--------------------------------------------"
            z=anArtist.doit(x)
            s=score(z,orig)
            swaps=1
            while swaps>0:
                swaps=0
                for i in range(2,len(circles)+1):
                    params=baseparams[:]
                    for c in range(i-2):
                        params.extend(circles[c])
                    p1=params[:]
                    p2=params[:]
                    p1.extend(circles[i-2])
                    p1.extend(circles[i-1])
                    p2.extend(circles[i-1])
                    p2.extend(circles[i-2])
                    s1=score(anArtist.doit(p1),orig)
                    s2=score(anArtist.doit(p2),orig)
                    l1=len(compressProgram(anArtist.makeProgram(p1)))
                    l2=len(compressProgram(anArtist.makeProgram(p2)))
                    if s2<s1 or (abs(s1-s2)<1 and l2<l1):
                        print "Swapping circle",i-2,"with circle",i-1,"for score change",s1,s2,"length change",l1,l2
                        tmp=circles[i-1]
                        circles[i-1]=circles[i-2]
                        circles[i-2]=tmp
                        tmp=bcircle[i-1]
                        bcircle[i-1]=bcircle[i-2]
                        bcircle[i-2]=tmp
                        swaps=1
            params=baseparams[:]
            for c in circles:
                params.extend(c)
            sfinal=score(anArtist.doit(params),orig)
            print "Beginning score",s,"final score",sfinal
                
    
            print "--------DELTAS    --------------------------------------------"
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
                
            print "--------------------------------------------------------------"
            includeCircle=[True]*len(circles)
            for i in range(len(circles)/2):
                r=np.argmin(deltas)
                print "removing circle",r,"diff",deltas[r]
                olddelta=deltas[r]
                deltas[r]=999999999.
                nextr=np.argmin(deltas)
                includeCircle[r]=False
                params=baseparams[:]
                for c in range(len(circles)):
                    if includeCircle[c]:
                        params.extend(circles[c])
                z=anArtist.doit(params)
                s=score(z,orig)
                prg=anArtist.makeProgram(params)
                zprg=compressProgram(prg)
                print i,s,lastscore-s,len(prg),len(zprg)
                if s-lastscore > deltas[nextr]:
                    print "Change is worse than next circle, reject change"
                    includeCircle[r]=True
                    deltas[r]=s-lastscore
                else:
                    lastscore=s
                    if len(zprg) <= sizelimit:
                        break
            params=baseparams[:]
            bounds=basebounds[:]
            for c in range(len(circles)):
                if includeCircle[c]:
                    params.extend(circles[c])
                    bounds.extend(bcircle[c])
                    
            
        pfd=open("foundparamsv17_%d_%d.txt"%(f,sizelimit),'w')
        print >>pfd,params
        print >>pfd,bounds
        pfd.close()
        params=localopt(optimizeme,params,bounds,[True,]*len(bounds),1)
        pfd=open("foundparamsv17final_%d_%d.txt"%(f,sizelimit),'w')
        print >>pfd,params
        print >>pfd,bounds
        pfd.close()
        s=score(z,orig)
        prg=anArtist.makeProgram(params)
        zprg=compressProgram(prg)
        print "Achieved final score:",s
        print "In program of size:",len(zprg)
        if len(zprg) <= 1024:
            fd=open("draw_finalv17_%d_%d.py"%(f,sizelimit),'w')
            print >>fd,prg,
            fd.close()
            fd=open("cdraw_finalv17_%d_%d.py"%(f,sizelimit),'w')
            print >>fd,zprg,
            fd.close()
