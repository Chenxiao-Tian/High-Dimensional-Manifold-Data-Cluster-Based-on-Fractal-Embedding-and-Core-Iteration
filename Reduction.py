# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 19:16:02 2022

@author: 田晨霄
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 00:15:59 2019

@author: tcx
"""

import math
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
def distant(x,y,m):
    s=0
    for i in range(m):
        s=s+(x[i]-y[i])**2
    s=tf.sqrt(tf.cast(s,tf.float32)+0.0001)
    return s
def h(form):
    form.sort()
    return form[1]
def cos(x,y,z):
    return (x**2+y**2-z**2)/(2*x*y)
def newfindpoints(o,y,m,r,lamda1,lamda2,lamda3):
    u=[]
    if len(y)==3:
        for i in range(3):
            for j in range(i+1,3):
                u.append(distant(y[i],y[j],m))
        a=abs(r[0]-r[1])
        b=abs(r[0]+r[1])
        e=abs(r[1]+r[2])
        f=abs(r[1]-r[2])
        m=math.acos(cos(r[0],r[1],h([a,b,u[0]])))
        t=math.acos(cos(r[1],r[2],h([e,f,u[2]])))
        sita1=random.uniform(0,2*math.pi)
        sita3=-(1/3)*(m+t)+sita1
        sita2=(1/3)*(m+t)+sita1
        A=[sita1,sita2,sita3]
        BAW=[]
        for i in range(3):
            BAW.append([o[0]+max(lamda1*lamda2*(r[i]/max(r[0],r[1],r[2])),lamda3)*math.cos(A[i]),o[1]+max(lamda1*lamda2*(r[i]/max(r[0],r[1],r[2])),lamda3)*math.sin(A[i])])
        return BAW
    if len(y)==1:
        return [o]
    if len(y)==2:
        dis=distant(y[0],y[1],m)
        a=abs(r[0]-r[1])
        b=abs(r[0]+r[1])
        e=h([dis,a,b])
        sita1=random.uniform(0,2*math.pi)
        sita2=math.acos(cos(r[0],r[1],h([a,b,e])))+sita1
        A=[sita1,sita2]
        BBq=[]
        for i in range(2):
            BBq.append([o[0]+max(lamda1*lamda2*(r[i]/max(r[0],r[1])),lamda3)*math.cos(A[i]),o[1]+max(lamda1*lamda2*(r[i]/max(r[0],r[1])),lamda3)*math.sin(A[i])])
        return BBq

def judge(v,x,m,r):
    s=0
    for i in range(m):
        if abs(v[i]-x[i])<r:
            s=s+1
        if s==m:
            return True
#def cut(x,n,m,r):
#    z=[]
#    if len(x)==1:
#        return [x]
#    while len(x)>0:
#        u=[]
#        u.append(x[0])
#        s=0
#        for p in range(len(x)) :
##            if distant(x[0],v,m)<r:
##              u.append(v) 
#            if distant(x[p],x[0],m)<r and x[p]!=x[0]:
#             #if distant(x[0],v,m)<r and v!=x[0]:
#                u.append(x[p])
#                s=s+1
##                if x[p]==x[0]:
##            x.remove(x[p])
#                if s==2:
#                    break
#        z.append(u)
#        for i in range(len(v)):
#            x.remove(v[i])       
#    return z
#def cut(x,n,m,r):
#    z=[]
#    if len(x)==1:
#        return [x]
#    while len(x)>0:
#        u=[]
#        u.append(x[0])
#        for v in x :
##            if distant(x[0],v,m)<r:
##              u.append(v) 
#            s=0
#            if judge(v,x[0],m,r)==True and v!=x[0]:
#                u.append(v)
#                s=s+1
#                if v!=x[0]:
#                    x.remove(v)
#                if s==2:
#                    break
#        z.append(u)
#        x.remove(x[0])       
#    return z            
def cut(x,n,m,r,k):
    z=[]
    if len(x)==1:
        return [x]
    while len(x)>0:
        u=[]
        lamda=x[0]
        u.append(x[0])
        m=[]
        if len(x)==1:
            x.pop(0)
            u=u
            z.append(u)
            break
        elif len(x)==2:
            x.pop(0)
            u.append(x[0])
            z.append(u)
            break
    
        else:
            x.pop(0)
            for i in range(1,k+1):
                t=distant(x[int(len(x)/k)*i-1],lamda,3)
                m.append(t)
                hg=m
                v=x
                m.sort()
            for j in range(0,min(2,len(m))):
                u.append(x[hg.index(m[j])])
                v.remove(x[hg.index(m[j])])
#            if distant(x[0],v,m)<r:
#              u.append(v) 
            
#            if judge(v,x[0],m,r)==True and v!=x[0]:
#                u.append(v)
#                s=s+1
#                if v!=x[0]:
#                    x.remove(v)
#                if s==2:
#                    break
            z.append(u)
            x=v
#            for i in range(min(len(m),2)):
#                x.pop(hg.index(m[i]))

    
            
            
            #for i in range(len(u)):
                #x.remove(u[i])
        
    return z

def mediumpoint(x,m):
    v=[]
    a=len(x)
    if len(x)==1:
        return x[0]
    else:
        for j in range(m):
            s=0
            for i in range(a):
                s=s+x[i][j]
            v.append(s/max(a,1))
        return v
def allmediumpoint(cate,m):
    newneedcut=[]
    for i in range(len(cate)):
        a=mediumpoint(cate[i],m)
        newneedcut.append(a)
    return  newneedcut
#def averagedistant(x,m,k):
#    r=0
#    n=len(x)
#    v=[]
#    for i in range(1,k+1):
#        v.append(x[int(n/k)*i-1])
#    l=0
#    for i in range(len(v)):
#        for j in range(i,len(v)):
#            l=l+1
#            r=r+distant(v[i],v[j],m)
#    r=r/max(1,l)
#    return r
def alldistant(x):
    u=[]
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            a=distant(x[i],x[j],3)
            if a!=0:
                u.append(a)
    return u
def pick(x,k):
    u=[]        
    for i in range(1,k+1):
        u.append(x[int(len(x)/k)*i-1])
    return u
def newR(x,m,k,t):#(t要取得大一些，使得newr偏小一些但不能过小）
    u=pick(x,k)
    s=alldistant(u)
    s.sort()
#    return t*s[0]+(1-t)*s[-1]
    return t*s[0]+(1-t)*s[-1]
##def newIsomap1(x,m,k,t):#k在01之间
##    thenumberofeachstagepoints=[]
##    theRofeachstage=[]
##    while True:
##        if len(x)==1:
##            break
##        k=min(len(x),k)
##        r=newR(x,m,k,t)
##        theRofeachstage.append(r)
##        x=cut(x,len(x),2,r)
##        w=[]
##        for i in range(len(x)):
##            w.append(len(x[i]))
##        thenumberofeachstagepoints.append(w)
##        x=allmediumpoint(x,2)
##    return [thenumberofeachstagepoints,theRofeachstage]
##   
def fangshe(x,lamda):
    u=[]
    for i in range(len(x)):
        v=[]
        for j in range(len(x[i])):
            v.append(lamda*x[i][j])
        u.append(v)
    return u
print(fangshe([[1,0]],10))        
def draw(s):
    x=[s[0][0].numpy()]
    y=[s[0][1].numpy()]
    for i in range(1,len(s)):
        x.append(s[i][0].numpy())
        y.append(s[i][1].numpy())
    for i in range(len(x)):
        x[i]=float(x[i])
        y[i]=float(y[i])
    plt.scatter(x,y,s=10)
    plt.title("sss",fontsize=24)
    plt.tick_params(axis="both",which="major",labelsize=14)
    plt.show
def newIsomap3(x,m,k,t):
    W=[]
    while True:
        if len(x)==1:
            break
        k=min(len(x),k)
#        r=newR(x,m,k,t)
#        x=cut(x,len(x),m,r)
        x=cut(x,len(x),m,1.0,k)
        v=[]
        R=[]
        for i in range(len(x)):
            v.append(x[i])
            r=[]
            for j in range(len(x[i])):
                r.append(distant(x[i][j],mediumpoint(x[i],m),m))
            R.append(r)
        op=[v,R]
        W.append(op)
        x=allmediumpoint(x,m)
    return W

def newreduced2(x,m,k,t,lamda3):
    u=[]
    O=[[0.,0.]]
    W=newIsomap3(x,m,k,t)
    for i in range(len(W)):
        a=len(W)-i-1
        u=[]
        for j in range(len(W[a][0])):
            newo=newfindpoints(O[j],W[a][0][j],m,W[a][-1][j],1/3.0**(i-len(W)),1.0,lamda3)
            for k in range(len(newo)):
                u.append(newo[k])    
        O=u
    return O


def roll(n):
    u=[]
    for i in range(n):
        r=random.uniform(0,1)
        l=random.uniform(0,1)
        t=(3*math.pi)/2*(1 +2*r)
        x=t*math.cos(t)
        y=2*l
        z=t*math.sin(t)
        X=[x,y,z]
        u.append(X)
    return u
#print(cut([[1,0],[0,1],[1,1],[0,0]],4,2,1.0))
#W=newIsomap3(roll(100),3,100,0.95)
#for i in range(len(W)):
#    for j in range(len(W[i][0])):
#        print(len(W[i][0][j]))
#def draw(s):
#    x=[s[0][0].numpy()]
#    y=[s[0][1].numpy()]
#    for i in range(1,len(s)):
#        x.append(s[i][0].numpy())
#        y.append(s[i][1].numpy())
#    for i in range(len(x)):
#        x[i]=float(x[i])
#        y[i]=float(y[i])
#    
#    plt.scatter(x,y,s=10)
#    plt.title("sss",fontsize=24)
#    plt.tick_params(axis="both",which="major",labelsize=14)
#    plt.show
#draw(newreduced2(roll(10000),3,100,0.9))          
##        
#def ball(n):
#    v=[]
#    for i in range(n):
#        sita=random.uniform(0,2*math.pi)
#        fi=random.uniform(0,math.pi)
#        x=5*math.cos(sita)*math.sin(fi)
#        y=5*math.sin(sita)*math.sin(fi)
#        z=5*math.cos(fi)
#        v.append([x,y,z])
#        v.append
#    return v
#R=ball(1000)

    
import sklearn.datasets as sd
data=sd.make_swiss_roll(n_samples=1000,noise=0.0,random_state=None)
data0=data[0].tolist()
data1=data[1].tolist()
#X=newreduced2(data0,3,100,0.68)
#
#ghj=newreduced2(data0,3,10,1.0)
data0=fangshe(data0,1)
# global opopo
# opopo=[]
# for i in range(len(data0)):
#     opopo.append(data0[i])
# mty=[]
# for i in range(len(data0)):
#     mty.append(data0[i])
X=newreduced2(data0,3,250,0.68,60)
# print(len(X[0]))
# Vmax=cut(data0,len(data0),3,1,100)
# global QAZ
# QAZ=[]
# for i in range(len(Vmax)):
#     for j in range(len(Vmax[i])):
#         QAZ.append(Vmax[i][j])
# ##
# QA=[]
# for i in range(len(opopo)):
#     QA.append(QAZ.index(opopo[i]))
# #

# orderx=[]
# for i in range(len(X)):
#     orderx.append(X[QA[i]])
##    
##X=ghj
#print(data0,ghj)
#X=data0
#print(ghj)
#print(opopo,Vmax,QAZ,QA,X,orderx)
#print(xc,Vmax,QAZ,QA)
#print(QA)
#print(ghj)
#X=ghj










x=[]
y=[]
for i in range(len(X)):
    x.append(float(X[i][0]))
    y.append(float(X[i][1]))
# x=[]
# y=[]
# for i in range(len(X)):
#     x.append(float(X[i][0]))
#     y.append(float(X[i][1]))







data1=X

t=[]
for i in range(len(data1)):
    if i<len(data1)/4:
        t.append("r")
    if i>=len(data1)/4 and i<len(data1)*2/4:
        t.append("r")
    if i>=len(data1)*2/4 and i<len(data1)*3/4:
        t.append("y")
    if i>=len(data1)*3/4 and i<len(data1):
        t.append("y")

data=[data0,data1]
# dictionary=color(data,3,100,0.95)
# t=[] 
# for i in range(len(dictionary[1])):
#     t.append(dictionary[0][dicitionary[1][i]])
import matplotlib.pyplot as plt
cmap=plt.cm.hot
plt.scatter(x,y,c=t) 
plt.title("shujujiangwei",fontsize=1)
plt.tick_params(axis="both",which="major",labelsize=14)
plt.show
###优势，适合封闭凸曲面，速度快
###劣势，对于弯曲程度大的如swiss roll，会杂糅
###改进1，把欧式改为测地距离，中心改为曲面三角形的重心（问题：如何取重心）
###改进2，先把原流形剥离出本质维数（但此时慎用于封闭曲面）###算法主要侧重聚类，欧式距离近的近，欧式距离远的远，对于弯曲程度大的曲面保拓扑能力较差d
    

