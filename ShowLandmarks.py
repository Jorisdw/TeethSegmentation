import cv2
import os
import numpy
import principal
import proc
import math
import ActiveShapeModelAlgorithm as ASM
import copy

class ShowLandmarks:

    def __init__(self):
        self.txy = None


    def readImg(self,name):
        img = cv2.imread(os.path.dirname(os.path.abspath(__file__))+'/' + name)
        return img

    def showResizedImage(self,img):
        r = 1000.0 / img.shape[1]
        dim = (1000, int(img.shape[0] * r))
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow('img',resized)
        cv2.waitKey(0)

    def readLandmarks(self,nb,ToothNumber,Lvl):
        array = []
        i = ToothNumber
        with open(os.path.dirname(os.path.abspath(__file__))+'/_Data/Landmarks/original/landmarks' + nb +'-'+ str(i) + '.txt') as f:
            for line in f: # read rest of lines
                array.append([int(float(x)) for x in line.split()])

        length = len(array)
        i = 0
        landmarks = []
        while (i < length) :
            landmarks.append((array[i][0]/Lvl,array[i+1][0]/Lvl))
            i = i+ 2
        return landmarks

    def placeLandmarks(self,img, landmarks,k):
        r = img.shape[1]/600
        for i in range(0,landmarks.size/2) :
            cv2.circle(img, (int(landmarks[i]),int(landmarks[i+landmarks.size/2])),r,((80*k)%255, (40*k)%255, (120*k)%255),-1)
        return img

    def placeContour(self, img, landmarks, k):
        for i in range(0,landmarks.size/2-1):
            cv2.line(img,(int(landmarks[i]),int(landmarks[i+landmarks.size/2])) , (int(landmarks[i+1]),int(landmarks[i+landmarks.size/2+1])) , ((80*k)%255, (40*k)%255, (120*k)%255),3)
        cv2.line(img,(int(landmarks[0]),int(landmarks[landmarks.size/2])) , (int(landmarks[landmarks.size/2-1]),int(landmarks[landmarks.size-1])) , ((80*k)%255, (40*k)%255, (120*k)%255),3)
        return img

    def makeBitMap(self,img, landmarks):
        zeros = numpy.zeros((landmarks.size/2,2))
        points = numpy.array(zeros, dtype='int32')
        img = numpy.zeros(img.shape)
        for i in range(0,landmarks.size/2):
            points[i] = [ int(landmarks[i]) , int(landmarks[landmarks.size/2+i]) ]
        cv2.fillPoly(img, [points], (255,0,0))
        return img

    def landmarkMatrix(self,landmarks):
        np = numpy.zeros(shape=(len(landmarks)*2))
        i = 0
        while i < len(landmarks):
            np[i] = landmarks[i][0]
            np[i+len(landmarks)] = landmarks[i][1]
            i += 1
        return np

    def mouse_callback(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.txy = (x,y)

    def getMouseXY(self):
        return self.txy

    def setBvalue(self, bTemp, eigenvalues):
        b = []
        for i in range(0,bTemp.size):
            if abs(bTemp[i]) < 3*math.sqrt(eigenvalues[i]):
                b.append(bTemp[i])
            elif bTemp[i] < 0:
                b.append(-3*math.sqrt(eigenvalues[i]))
            else:
                b.append(3*math.sqrt(eigenvalues[i]))
        return b

    def compareBitMaps(self,goal,result):
        img = numpy.zeros(goal.shape)
        for i in range(0,result.shape[0]):
            for j in range(0,result.shape[1]):
                gp = goal[i,j]
                rp = result[i,j]
                if numpy.linalg.norm(gp) > 0 and numpy.linalg.norm(rp) > 0:
                    img[i,j] = [ 0,255,0 ]
                elif numpy.linalg.norm(gp) > 0:
                    img[i,j] = [ 0,0,255 ]
                elif numpy.linalg.norm(rp) > 0:
                    img[i,j] = [ 255,0,0 ]
                else:
                    img[i,j] = [ 0,0,0 ]
        return img


class UpdateModel:

    def update(self,Tx, Ty, Lvl, ToothNumber, iterations,scale,A,ImagePath):
        reload(principal)
        reload(ASM)
        reload(proc)

        ####################
        # Inlezen van data #
        ####################
        SL1 = ShowLandmarks()
        allLandmarks = []
        for i in range(1,15):
            allLandmarks.append(SL1.landmarkMatrix(SL1.readLandmarks(str(i),ToothNumber,Lvl)))
            # allLandmarks is een lijst met 1xn vectoren van de trainingslandmarks
        if Tx == None and Ty == None :
            AvgX = 0
            AvgY = 0
            Sum = 0
            for lmrk in allLandmarks:
                AvgX = AvgX + numpy.sum(lmrk[0:lmrk.size/2+1])
                AvgY = AvgY + numpy.sum(lmrk[lmrk.size/2+1::])
                Sum = Sum + lmrk.size/2
            Tx = AvgX/Sum
            Ty = AvgY/Sum+10
        #########################
        # Aligneer trainingsset #
        #########################
        procr = proc.procrustes()
        matrixLijst = []
        vector0 = numpy.zeros((allLandmarks[0].size/2,2))
        vector0[:,0] = allLandmarks[0][0:allLandmarks[0].size/2]
        vector0[:,1] = allLandmarks[0][allLandmarks[0].size/2::]
        landmarks = numpy.zeros((allLandmarks[0].size/2,2))
        scaling = []
        translation = []
        for i in range(1,14):
            landmarks[:,0] = allLandmarks[i][0:allLandmarks[0].size/2]
            landmarks[:,1] = allLandmarks[i][allLandmarks[0].size/2::]
            mtx1,mtx2,z1,norm1,norm2,mean1,mean2 = procr.procrustes(vector0,landmarks)
            scaling.append(norm2)
            translation.append(mean2)
            vector = numpy.zeros(allLandmarks[0].size)
            vector[0:vector.size/2] = mtx2[:,0]
            vector[vector.size/2::] = mtx2[:,1]
            matrixLijst.append(vector)
        vector[0:vector.size/2] = mtx1[:,0]
        vector[vector.size/2::] = mtx1[:,1]
        scaling.insert(0,norm1)
        translation.insert(0,mean1)
        matrixLijst.insert(0,vector)
            # -> matrixLijst:   lijst van alle vectoren van de training samples die gecentreerd zijn rond de oorsprong,
            #                   geschaald en geroteerd zodat ze allemaal ongeveer hetzelfde eruit zien. Dit moet gebruikt
            #                   worden voor P te berekenden via PCA.
            # -> allLandmarks:  Lijst van alle vectoren van die trainingssamples die nog op hun originele plaats staan.
            #                   Deze worden gebruikt voor het berekenen van het statistisch model.

        ###############################
        # Bereken principal component #
        ###############################
        PCA = principal.pca()
        p, eigenvalues = PCA.pca(0.99, matrixLijst) # -> P-matrix
        Xmean = PCA.meanVector(matrixLijst)

        asm = ASM.asm()
        k = 7
        S, mean = asm.trainStatisticalModel(allLandmarks, Xmean, k, Lvl) # -> Covariance matrix en mean vector van de landmarks in de trainingsamples

        imgOrColor = cv2.imread(os.path.dirname(os.path.abspath(__file__))+ImagePath)
        imgOrColor = cv2.resize(imgOrColor, (0,0), fx=float(1)/Lvl, fy=float(1)/Lvl)
        imgOr = cv2.cvtColor(imgOrColor, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        imgOr = clahe.apply(imgOr)

        if Lvl == 1:
            temp = cv2.GaussianBlur(imgOr,(3,3),3)
            imgOr = cv2.addWeighted(imgOr, 1.5, temp, -0.5, 0, imgOr)
        #imgOr = cv2.medianBlur(imgOr,3)
        i=0

        b = numpy.zeros(p.shape[1])
        if A==None:
            A = numpy.identity(Xmean.size)
        subA = numpy.identity(Xmean.size/2)

        Ex = numpy.zeros(Xmean.size)
        Ex[0:Xmean.size/2] = numpy.ones(Xmean.size/2)

        Ey = numpy.zeros(Xmean.size)
        Ey[Xmean.size/2::] = numpy.ones(Xmean.size/2)

        s = scale
        m = 21
        Xi = s*A.dot(Xmean + p.dot(b)) + Tx*Ex + Ty*Ey

        while i < iterations:

            Y = asm.updateAllLandmarks(Xi,imgOr,k,m,S, mean)  # -> Dees werkt nog voor geen kanten!!


            ds,theta,tx,ty = asm.fitModelToVector3(Y,Xi,p)
            dA = numpy.zeros((A.shape[0],A.shape[1]))
            dA[0:(subA.shape[0]), 0:(subA.shape[1])] = math.cos(theta)*subA
            dA[0:subA.shape[0], subA.shape[1]::] = -math.sin(theta)*subA
            dA[subA.shape[0]::, 0:subA.shape[1]] = math.sin(theta)*subA
            dA[subA.shape[0]::, subA.shape[1]::] = math.cos(theta)*subA


            # b updaten
            XidXiInOrigin = numpy.linalg.inv(A).dot(Y - ((Tx)*Ex + (Ty)*Ey))/s
            bTemp = asm.fitModelToVector(XidXiInOrigin,Xmean,p)
            b = SL1.setBvalue(bTemp,eigenvalues)

            A = A.dot(dA)
            s = s*ds
            Tx = Tx + tx
            Ty = Ty + ty



            Xi = s*A.dot(Xmean + p.dot(b)) + (Tx)*Ex + (Ty)*Ey
            i += 1
            #if i == 1 or i == iterations:
            #    img = copy.deepcopy(imgOrColor)
            #    SL1.placeContour(img, Xi,2)
            #    SL1.showResizedImage(img)
        #cv2.destroyAllWindows()
        if Lvl == 1:
            return Tx,Ty,s,A, Xi
        else:
            return Tx,Ty,s,A

if __name__ == '__main__':

    # Input image
    ImagePath = '/_Data/Radiographs/01.tif'
    Segmentations = []
    # Toothnumber can range between 1 and 8 depending on which tooth
    for ToothNumber in range(5,6):
        s = 70
        Tx = None
        Ty = None
        A = None
        updater = UpdateModel()
        print 'Starting level 3'
        Tx,Ty,s,A = updater.update(None,None,4, ToothNumber, 6,2*s,None,ImagePath)
        print 'Starting level 2'
        Tx,Ty,s,A = updater.update(2*Tx,2*Ty,2, ToothNumber, 20,2*s,A,ImagePath)
        print 'Starting level 1'
        Tx,Ty,s,A,X = updater.update(Tx*2,Ty*2,1, ToothNumber, 40,2*s,A,ImagePath)
        Segmentations.append(X)

    img = cv2.imread(os.path.dirname(os.path.abspath(__file__))+'/_Data/Segmentations/01-4.png')
    SL1 = ShowLandmarks()
    i = 1
    print 'Drawing Result'
    for X in Segmentations:
        # This can be used when only the border of the tooth has to be drawn.
        #bit = SL1.placeContour(img, X,2)
        # Draws the binary image
        bit = SL1.makeBitMap(img, X)
        i += 1
    #res = SL1.compareBitMaps(img, bit)
    SL1.showResizedImage(bit)
