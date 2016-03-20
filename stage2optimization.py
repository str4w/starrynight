# -*- coding: utf-8 -*-
"""
Enhance the optimization one more time
"""

from starrynight import *
import cv2

sizelimit=1024
orig=cv2.imread('ORIGINAL.png')

pfd=open("foundparamsv26final_1_1024.txt") #4764
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
print "Starting witih score",s

def optimizeme(p):
   img=anArtist.doit(p)
   s=score(img,orig)+regularizer(p,orig)
   return s  
nbprimitives=50

for sizelimit in [1100,1024]:
    for f in [20,5,1]:
        x=params[:]
        x,b=searchForParameters("outputv27_%d_%d"%(f,sizelimit),orig,params,bounds,nbprimitives,f)
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
                
        
        pfd=open("foundparamsv27_%d_%d.txt"%(f,sizelimit),'w')
        print >>pfd,params
        print >>pfd,bounds
        pfd.close()
        params=localopt(optimizeme,params,bounds,[True,]*len(bounds),1)
        pfd=open("foundparamsv27final_%d_%d.txt"%(f,sizelimit),'w')
        print >>pfd,params
        print >>pfd,bounds
        pfd.close()
        s=score(z,orig)
        prg=anArtist.makeProgram(params)
        zprg=compressProgram(prg)
        print "Achieved final score:",s
        print "In program of size:",len(zprg)
        if len(zprg) <= 1024:
            fd=open("draw_finalv27_%d_%d.py"%(f,sizelimit),'w')
            print >>fd,prg,
            fd.close()
            fd=open("cdraw_finalv27_%d_%d.py"%(f,sizelimit),'w')
            print >>fd,zprg,
            fd.close()
