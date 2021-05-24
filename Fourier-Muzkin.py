# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:38:41 2021

@author: ED-DAKI Issam 
"""


######################### Fourier Mutzkin problem ###############

import numpy as np

class LP:
    def __init__(self,n,C,A,B):
        self.n=n
        self.C=C
        self.A=A
        self.B=B

    def permutation(self):
        def perm_index():
            permute_index=0
            for i in self.C:
                if i!=0:
                    return permute_index
                permute_index+=1
                
        def perm_matrix():
            I=np.eye(self.n)
            H=np.eye(self.n)
            H[self.n-1]=I[perm_index()]
            H[perm_index()]=I[self.n-1]
            return H
        return perm_index(),perm_matrix()
    def New_prameters(self):
        #les nouveu parametres à utiliser sont n,C*H,A*H,B 
        H=self.permutation()[1]
        C_prime=np.dot(self.C.T,H).T
        
        A_prime=np.concatenate((self.A,-np.eye(self.n)),axis=0)
        A_prime=np.dot(A_prime,H)
        
        B_prime=np.concatenate((self.B,np.zeros(self.n).reshape(self.n,1)),axis=0)
        
        return C_prime,A_prime,B_prime
    
    def changing_variabl(self):
        """
        Z=MY
        """
        C_prime,A_prime,B_prime=self.New_prameters()
        M=np.eye(self.n)
        M[-1]=C_prime.T
        M_inv=np.linalg.inv(M)
        
        C_final=np.dot(C_prime.T,M_inv)
        A_final=np.dot(A_prime,M_inv)
        B_final=B_prime
        
        return  C_final,A_final,B_final,M,M_inv
    
    def solve(self):
        A,b=self.changing_variabl()[1],self.changing_variabl()[2]

        def FM1(A,b):
            B=np.concatenate((A,-b.reshape((b.shape[0],1))),axis=1)#axis=1 pour concaténation horisontal(colones)
            return B

        def FM2(B):  #B de type np.array
            C = np.copy(B)
            for i in range(len(C)):
                if C[i][0] != 0:
                    C[i] /= np.abs(C[i][0])
         
            return C

        def FM3(C):  #C a le type np.array
            (p,q) = np.shape(C)
            E = np.array([]).reshape(0,q-1)
            G = np.array([]).reshape(0,q-1)
            D = np.array([]).reshape(0,q-1)
            for i in range(p):
                ligne = np.copy(C[i][1:]).reshape(1,q-1)
                if C[i][0] > 0:
                    D = np.concatenate((D,-ligne))
                elif C[i][0] < 0:
                    G = np.concatenate((G,ligne))
                else:
                    E = np.concatenate((E,ligne))
            for g in G:
                for d in D:
                    E = np.concatenate((E,(g-d).reshape(1,q-1)))
            
            return E,G,D

       
        def FM4(A,b):  #A a le type np.array
            E=FM1(A,b)
            E=FM2(E)
            (p,q)=np.shape(E)            

            list_G=[FM3(E)[1]]
            list_D=[FM3(E)[2]]
        
            for k in range(q-2):
                E=FM3(E)[0]
                E=FM2(E)
                
                list_G.append(FM3(E)[1])
                list_D.append(FM3(E)[2])
                    
            (p,q)=np.shape(E)
            mini=-10**12  # equivalent de - l'infini
            maxi= 10**12  # equivalent de + l'infini 
            for k in range(p):
                if E[k,0]>0:
                    maxi=min(maxi, -E[k,1])
                elif E[k,0]<0:
                    mini=max(mini, E[k,1])    
                    #je suppose ici que si E[k,0]=0 on a E[k,1]<0 !!             
            return [mini,maxi],list_G,list_D  #si mini>maxi, le polyedre est vide !!
        
        def argmin_problem(A,b):
            
            list_G= FM4(A,b)[1]
            list_D= FM4(A,b)[2]
            
            list_G.reverse()
            list_D.reverse()
                
            m=FM4(A,b)[0][0]# m: minimum de xn
            solution=np.array([[m],[1.]])
            for i in range(1,A.shape[1]):
                
                G=list_G[i]
                D=list_D[i]
                
                maximum=max(np.dot(G,solution))
                minimum=min(np.dot(D,solution))
                
                k=np.array([(minimum+maximum)/2]).reshape((1,1))
                
                solution=np.concatenate((k,solution),axis=0)
            return FM4(A, b)[0][0],solution[:-1,:]
        
        
        Z_star=argmin_problem(A,b)[1]
        M_inv=self.changing_variabl()[4]
        H=self.permutation()[1]
        print(H.shape,M_inv.shape,Z_star.shape)
        
        X_star=np.dot(H,M_inv)
        X_star=np.dot(X_star,Z_star)
        min_f=argmin_problem(A,b)[0]
        
        return min_f,X_star
    

n=3

################# Construction de la fonction Fourier Motzkin ##################
def FourierMotzkin(A,b,c):
    P=LP(n,c,A,b)#on cree une instance de probleme
    return P.solve()#retourne la solution









############  test ################
   
            
n=3
                
C=np.array([[100.,-11.,1.]]).T              
#C.shape 

A=np.array([[-1.,-1.,-1.],[-3.,1.,1.],[1.,-3.,1.],[2.,1.,-3.],[0.,1.,1.]])  

B=np.array([[1.],[1.],[1.],[1.],[4.]])  
#B.shape   

FourierMotzkin(A, B, C)




