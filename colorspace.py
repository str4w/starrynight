import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
orig=cv2.imread('ORIGINAL.png')

#real image range
imgmax=np.max(orig,(0,1))
imgmin=np.min(orig,(0,1))
print "RealImageRange",imgmax,imgmin

C=np.cov(orig.reshape(orig.shape[0]*orig.shape[1],3).T)
eigenvalues,eigenvectors=np.linalg.eig(C)
repack=sorted([(eigenvalues[i],eigenvectors[:,i]) for i in range(3) ],key=lambda x:-x[0],)
R=np.array([v[1] for v in repack])

meancolor=np.round(np.mean(orig,(0,1)))
meanoutimage=orig.astype(np.int)-meancolor

#testimage=np.dot(R[0:2,:],(meanoutimage.reshape(orig.shape[0]*orig.shape[1],3).T*1.)).T

# recovery test
#rmtx=R.T[:,0:2]
#rtestimage=(np.dot(rmtx,(testimage.T*1.)).T+meancolor).astype(np.uint8)
#print "Range of recovered image:",np.max(rtestimage,0),np.min(rtestimage,0)
#print "Score recovered:",np.sum((rtestimage-orig.reshape(orig.shape[0]*orig.shape[1],3).astype(np.float32))**2)/255./255.



print testimage.shape
print np.max(testimage[:,0]),np.min(testimage[:,0])
print np.max(testimage[:,1]),np.min(testimage[:,1])

org=np.array([[np.min(testimage[:,0]),np.min(testimage[:,1])]])
scl=np.array([99./(np.max(testimage[:,0])-np.min(testimage[:,0])),99./(np.max(testimage[:,1])-np.min(testimage[:,1]))])

scaledimage=np.round((testimage-org)*scl)

print "Range of scaled image:",np.max(scaledimage,0),np.min(scaledimage,0)

#sfactor=999.

#rscl=np.round(sfactor/scl).astype(np.int)
#rorg=np.round(sfactor*org).astype(np.int)
#rmtx=np.round(R.T[:,0:2]*sfactor).astype(np.int)
roff=np.round(np.dot(R.T[:,0:2],org.reshape(2,1))+meancolor.reshape(3,1))
print roff
tmp=R.T[:,0:2].copy()
tmp[:,0]/=scl[0]
tmp[:,1]/=scl[1]
rmtx2=np.round(tmp,1)  #).astype(np.int)
print rmtx2
#rscl=sfactor/scl
#rorg=sfactor*org
#rmtx=R.T[:,0:2]*sfactor

rtestimage=scaledimage.copy()
#rtestimage*=rscl
#rtestimage+=rorg
#rtestimage=np.clip( ((np.dot(rmtx,rtestimage.T)/(sfactor*sfactor)).T+meancolor),0,255)
rtestimage=np.clip( (np.dot(rmtx2,rtestimage.T).T+roff.reshape(3)),0,255)
print rtestimage.shape
print "Range of recovered image:",np.max(rtestimage,0),np.min(rtestimage,0)
print "Score recovered:",np.sum((rtestimage-orig.reshape(orig.shape[0]*orig.shape[1],3).astype(np.float32))**2)/255./255.

#colorcube=np.array([[(i&4>0)*imgmax[0]+(i&4==0)*imgmin[0],
#                     (i&2>0)*imgmax[1]+(i&2==0)*imgmin[1],
#                     (i&1>0)*imgmax[2]+(i&1==0)*imgmin[2]] for i in range(8)])
#print "Colorcube"
#print colorcube
#testcube=np.dot(R[0:2,:],((colorcube-meancolor).T)).T
#print "testcube"
#print testcube
#print np.max(testcube[:,0]),np.min(testcube[:,0])
#print np.max(testcube[:,1]),np.min(testcube[:,1])
#
#imtx=R.T
#itestcube=np.round(np.dot(imtx[:,0:2],testcube.T).T+meancolor).astype(np.uint8)
#print "testing invers"
#print itestcube
#
#rtestimage=np.round(np.dot(imtx[:,0:2],testimage.T).T.reshape(orig.shape)+meancolor).astype(np.uint8)
#rtestimage.astype(np.float32)-orig.astype(np.float32)
#plt.imshow(rtestimage[:,:,(2,1,0)])
#plt.show()
#delta=rtestimage.astype(np.float32)-orig.astype(np.float32)
#print np.max(np.abs(delta),(0,1))
#plt.imshow(delta)
#plt.show()
#
#
#mtestimage=testimage.copy()
#mtestimage-=org
#mtestimage*=scl
#mtestimage=np.round(mtestimage).astype(np.int)
#
#print "Range of image:"
#print np.max(mtestimage[:,0]),np.min(mtestimage[:,0])
#print np.max(mtestimage[:,1]),np.min(mtestimage[:,1])
#
#mtestcube=testcube.copy()
#mtestcube-=org
#mtestcube*=scl
#mtestcube=np.round(mtestcube).astype(np.int)
#
#print "Testcube:"
#print mtestcube
#print np.max(mtestcube[:,0]),np.min(mtestcube[:,0])
#print np.max(mtestcube[:,1]),np.min(mtestcube[:,1])
#
#rscl=np.round(999./scl).astype(np.int)
#rorg=np.round(999*org).astype(np.int)
#rmtx=np.round(R.T[:,0:2]*999).astype(np.int)
#
#rtestcube=mtestcube.copy()
#rtestcube*=rscl
#rtestcube+=rorg
#rtestcube=(np.dot(rmtx,rtestcube.T)/999/999).T+meancolor
#print rtestcube
#
#


  
