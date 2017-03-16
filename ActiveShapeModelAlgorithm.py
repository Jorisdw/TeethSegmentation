import principal
import numpy as np
import cv2
import os
import math
import copy

class asm:

    def __init__(self):
         'asm'

    def trainStatisticalModel(self,trainingLandmarks,meanVector, k,Lvl):
        princ = principal.pca()
        imageList = self.loadImages(Lvl)
        Size = trainingLandmarks[0].size
        mean = []
        S = []

        xm = np.sum(meanVector[0:meanVector.size/2])/(meanVector.size/2)
        ym = np.sum(meanVector[meanVector.size/2::])/(meanVector.size/2)

        for i in range (0,Size/2):
            gi = []
            for j in range(0,14):
                if j == 0:
                    continue
                else:
                    Landmarks = trainingLandmarks[j]
                    if i==Size/2-1 :
                        temp = self.getDerivativeSample(Landmarks[i],Landmarks[i+Size/2],xm,ym,imageList[j],k)
                    if i > 0 and not(i==Size/2-1):
                        temp = self.getDerivativeSample(Landmarks[i],Landmarks[i+Size/2],xm,ym,imageList[j],k)
                    if i == 0:
                        temp = self.getDerivativeSample(Landmarks[i],Landmarks[i+Size/2],xm,ym,imageList[j],k)
                    if np.isnan(temp).any():
                        print temp
                        print i
                    gi.append(temp)
            mean.append(princ.meanVector(gi))
            cov = princ.covarianceMatrix2(gi)
            S.append(cov)

        return S,mean

    def getNormalPoints(self, (xm,ym),(x1,y1),k):       # -> Is normaal in orde nu!
        # Loodrechte tov de twee punten ernaast
        #aBorder = float((y0-y2)/(x0-x2))
        #if aBorder == 0:
        #    a = float('inf')
        #else:
        #    a = -1/aBorder

        if xm == x1 :
            a = float('inf')
        else:
            a = (y1-ym)/(x1-xm)

        result = np.zeros(shape=(2*k+1,2))
        if abs(a) > 1 or a == float('inf') or a == float('-inf') or math.isnan(a):
            for j in range(0,2*k+1):
                i = j - k
                if a == float('inf') or a == float('-inf') or math.isnan(a):
                    result[j][0] = int(x1)
                    result[j][1] = int(y1+i)
                else:
                    x= i/a + x1
                    result[j][0] = int(x)
                    result[j][1] = int(y1+i)
                if y1+i > 1600:
                    print 'y1 = ' + str(y1)
                    print 'y = ' + str(y1+i)
                j += 1

        else:
            for j in range(0,2*k+1):
                i = j - k
                y = a*i + y1
                result[j][0] = int(x1+i)
                result[j][1] = int(y)
                if y > 1600:
                    print 'a = ' + str(a)
                    print 'i = ' + str(i)
                    print 'y = ' + str(y)
                j += 1
        return result


    def loadImages(self,Lvl):
        List = []
        for i in range(1,15):
            img = cv2.imread(os.path.dirname(os.path.abspath(__file__))+'/_Data/Radiographs/'+ str(i).zfill(2)+'.tif')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (0,0), fx=float(1)/Lvl, fy=float(1)/Lvl)
            if Lvl == 1:
                temp = cv2.GaussianBlur(img,(3,3),3)
                img = cv2.addWeighted(img, 1.5, temp, -0.5, 0, img)
            #img = cv2.medianBlur(img,3)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            List.append(img)
        return List

    def getDerivativeSample(self,x,y,xm,ym,img,k):
        g = np.zeros(2*k+1)
        normalPoints = self.getNormalPoints((xm,ym),(x,y),k+1)
        temp = img[normalPoints[0][1],normalPoints[0][0]]
        for i in range (1,normalPoints.shape[0]-2):
            g[i] = int(img[normalPoints[i][1],normalPoints[i][0]]) - int(temp)
            temp = img[normalPoints[i][1],normalPoints[i][0]]
        if np.linalg.norm(g) == 0:
            return g
        g = g/np.linalg.norm(g)
        return g

    def updateAllLandmarks(self, Landmarks,img, k, m, S, mean):
        landmarks = copy.deepcopy(Landmarks)
        Size = landmarks.size
        for i in range(0,Size/2):
            if i > 0 and i < (Size/2-1):
                    x,y = self.updateLandmark(img, landmarks[i], landmarks[i+Size/2], k, m, S[i], mean[i],landmarks[i-1],landmarks[i+Size/2-1],landmarks[i+1],landmarks[i+Size/2+1])
            else:
                if i == 0:
                    x,y = self.updateLandmark(img, landmarks[i], landmarks[i+Size/2], k, m, S[i], mean[i],landmarks[Size/2-1],landmarks[Size-1],landmarks[i+1],landmarks[i+Size/2+1])
                if i == Size/2-1:
                    x,y = self.updateLandmark(img, landmarks[i], landmarks[i+Size/2], k, m, S[i], mean[i],landmarks[i-1],landmarks[i+Size/2-1],landmarks[0],landmarks[Size/2])
            landmarks[i] = x
            landmarks[i+Size/2] = y
        return landmarks

    def updateLandmark(self, img, X, Y, k, m, S, mean,x0,y0,x2,y2):
        fmin = int(1000000)
        xm = np.sum(mean[0:mean.size/2])/(mean.size/2)
        ym = np.sum(mean[mean.size/2::])/(mean.size/2)
        normalPoints = self.getNormalPoints((xm,ym),(X,Y),m-k)
        Xn=0
        Yn=0



        for j in range(0,2*(m-k)+1):
            g = self.getDerivativeSample(normalPoints[j][0],normalPoints[j][1], xm,ym, img, k)
            Sinv = np.linalg.pinv(S)
            f = np.transpose(g-mean).dot(Sinv)
            f = f.dot(g-mean)
            if fmin == f:
                print '--------------------------------------------------------------'
            if f < fmin :
                Xn = normalPoints[j][0]
                Yn = normalPoints[j][1]
                fmin = f

        return Xn,Yn

    def fitModelToVector(self, vector, model, P):
         meanVector = copy.deepcopy(model)
            # vector en meanVector zijn 1x2n vectoren
         b = np.zeros(P.shape[1])
         #Y = vector
         i=0
         while i < 5:

            yaccent = vector/(vector.dot(meanVector)) # -> KIJK ES WAT HIER MOET KOMEN DAN
            b = np.transpose(P).dot(yaccent-meanVector)


            i += 1
         return b


    def fitModelToVector3(self, vector, model, P): # model -> x1 en vector -> x2 =>
        tx=0
        ty=0
        a = float(model.dot(vector))
        a = a/float(math.pow(np.linalg.norm(model),2))
        b = 0
        for i in range(0,vector.size/2):
            b = b + model[i]*vector[vector.size/2+i] - model[vector.size/2+i]*vector[i]
        b = float(b/math.pow(np.linalg.norm(model),2))
        s = math.sqrt(math.pow(a,2)+math.pow(b,2))
        theta = math.atan(b/a)

        vectorTx = np.sum(vector[0:model.size/2])/(model.size/2)
        vectorTy = np.sum(vector[model.size/2::])/(model.size/2)
        modelTx = np.sum(model[0:model.size/2])/(model.size/2)
        modelTy = np.sum(model[model.size/2::])/(model.size/2)

        tx = vectorTx - modelTx
        ty = vectorTy - modelTy

        return s,theta,tx,ty
