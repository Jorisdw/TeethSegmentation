import numpy as np

class pca:

    def __init__(self):
        print ''

    def meanVector(self,vectorLijst):
        result = np.zeros(vectorLijst[0].size)
        i=0
        for vector in vectorLijst:
            result = result + vector
            i += 1
        result = result/i
        return result

    def covarianceMatrix(self,vectorLijst):
        mean = self.meanVector(vectorLijst)
        a = len(vectorLijst[0])
        b = len(vectorLijst)
        D = np.zeros(shape=(a,b))
        i=0
        for vector in vectorLijst:
            D[:,i] = vector-mean
            i += 1

        S = D.dot(np.transpose(D))/i
        return S

    def covarianceMatrix2(self, vectorLijst):
        X = np.zeros(shape=(vectorLijst[0].size,len(vectorLijst)))
        for i in range (0,len(vectorLijst)):
            X[:,i] = vectorLijst[i]
        S = np.cov(X)
        return S


    def eigenDecomposition(self,vectorLijst):
        S = self.covarianceMatrix(vectorLijst)
        Eval, Evec = np.linalg.eig(S)
        return Eval, Evec

    def pca(self,f,vectorLijst):
        eigenwaarden,eigenvectoren = self.eigenDecomposition(vectorLijst)
        positiveEigenvalues = np.absolute(eigenwaarden)

        idx = (-np.array(positiveEigenvalues)).argsort()
        eigenValues = positiveEigenvalues[idx]
        eigenvectoren = eigenvectoren[:,idx]


        evecT = np.transpose(eigenvectoren)
        Vt = np.sum(eigenValues)
        PT = np.array(evecT[0])
        # Eigenvector matrix maken door rij per rij toe te voegen
        Sum = eigenValues[0]
        t = 1
        #print 'Vt = ' +str(Vt)
        while Sum < f*Vt:
            Sum = Sum + eigenValues[t]
            if Sum < f*Vt:
                t += 1
                PT = np.vstack([PT, evecT[t]])
        P = np.transpose(PT)
        return P, eigenValues[0:t]
