# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 21:14:32 2021

@author: 田晨霄
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:21:39 2019

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
                t=distant(x[int(len(x)/k)*i-1],lamda,2)
                m.append(t)
                hg=m
                v=x
                m.sort()
            for j in range(0,min(2,len(m))):
                u.append(x[hg.index(m[j])])
                v.remove(x[hg.index(m[j])])
            z.append(u)
            x=v
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
def fangshe(x,lamda):
    u=[]
    for i in range(len(x)):
        v=[]
        for j in range(len(x[i])):
            v.append(lamda*x[i][j])
        u.append(v)
    return u   
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
def pingjunfenge(x,m,k,t,o):
    Top=[]
    form=[]
    for i in range(len(x)):
        Top.append(x[i])
        Top.append([i]) 
    for i in range(len(x)):
        form.append(x[i])
    L=len(x) #2
    while True:
        if len(x)==o:
            break
        k=min(len(x),k)
        x=cut(x,len(x),m,1.0,k) #[[1,0],[1,1]]
        for w in range(L):#L
            y=Top[2*w+1][-1]
            lp=form[y]
            q=0
            for p in range(len(x)):#len(x)
                if lp not in x[p]:
                    q=q+1
                else:
                    break
            Top[2*w+1].append(q)
        x=allmediumpoint(x,m)
        form=[]
        for i in range(len(x)):
            form.append(x[i])
    return Top  
def julei(x,m,k,t,o):
    L=len(x)
    Top=pingjunfenge(x,m,k,t,o)
    v=[]
    for j in range(o):
        u=[]
        for i in range(L):
           if Top[2*i+1][-1]==j:
               u.append(Top[2*i])
        v.append(u)
    return v
def newIsomap3(x,m,k,t):
    W=[]
    while True:
        if len(x)<=1:
            break
        k=min(len(x),k)
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
def newreduced2(chushizhongxin,x,m,k,t,lamda3):
    u=[]
    O=[chushizhongxin]
    W=newIsomap3(x,m,k,t)
    for i in range(len(W)):
        a=len(W)-i-1
        u=[]
        for j in range(len(W[a][0])):
            newo=newfindpoints(O[j],W[a][0][j],m,W[a][-1][j],1/1.3**(i-len(W)),1.0,lamda3)
            for k in range(len(newo)):
                u.append(newo[k])    
        O=u
    return O
def fangshe(x,lamda):
    u=[]
    for i in range(len(x)):
        v=[]
        for j in range(len(x[i])):
            v.append(lamda*x[i][j])
        u.append(v)
    return u
import sklearn.datasets as sd
data=sd.make_swiss_roll(n_samples=10,noise=0.0,random_state=None)
data0=data[0].tolist()
data1=data[1].tolist()

data0=[[1,0],[0,0],[0,1],[1,1]]
beifen1=[]
for i in range(len(data0)):
    beifen1.append(data0[i])
#print(beifen1)

Top=julei(beifen1,2,10,1,2) 
rtu=[]
for i in range(len(Top)):
    Top[i]=tuple(Top[i])
    rtu.append(Top[i])
TOPbeifen=[]
for i in range(len(Top))  :
    TOPbeifen.append(Top[i])

#print(Top)
#print(TOPbeifen,rtu)
#plo=[]
#for i in range(len(TOPbeifen)):
#    plo.append(cut(TOPbeifen[i],len(TOPbeifen[i]),3,1,len(TOPbeifen[i])))
##print(plo)
#mvp=[]
#for i in range(2):
#    for j in range(len(plo[i])):
#       for k in range(len(plo[i][j])):
#           mvp.append(data0.index(plo[i][j][k]))
#print(mvp)
chushizhongxinji=[[0,0],[6,0]]
#print(Top,rtu)
#print(newreduced2(chushizhongxinji[0],TOPbeifen[0],2,3,1,1))
v=[] 
for i in range(2):
    v.append(newreduced2(chushizhongxinji[i],TOPbeifen[i],2,len(TOPbeifen[i]),1,1))
print(v)

print(rtu)
plo=[]
for i in range(len(Top)):
    plo.append(cut(Top[i],len(Top[i]),3,1,len(Top[i])))
#print(plo)
mvp=[]
for i in range(2):
    for j in range(len(plo[i])):
       for k in range(len(plo[i][j])):
           mvp.append(data0.index(plo[i][j][k]))
print(mvp)

#u=[]
#for i in range(len(v)): 
#    for j in range(len(v[i])):
#        u.append(v[i][j])
#orderx=u
#newt=[]
#for i in range(len(data1)):
#    newt.append(data1[mvp[i]])
#x=[orderx[0][0]]
#y=[orderx[0][1]]
#for i in range(1,len(orderx)):
#    x.append(orderx[i][0])
#    y.append(orderx[i][1])
#for i in range(len(x)):
#    x[i]=float(x[i])
#    y[i]=float(y[i])
#import matplotlib.pyplot as plt
#cmap=plt.cm.hot
#plt.scatter(x,y,c=newt,cmap=plt.cm.hot) 
#plt.title("shujujiangwei",fontsize=1)
#plt.tick_params(axis="both",which="major",labelsize=14)
#plt.show
#####优势，适合封闭凸曲面，速度快
#####劣势，对于弯曲程度大的如swiss roll，会杂糅
#####改进1，把欧式改为测地距离，中心改为曲面三角形的重心（问题：如何取重心）
#####改进2，先把原流形剥离出本质维数（但此时慎用于封闭曲面）###算法主要侧重聚类，欧式距离近的近，欧式距离远的远，对于弯曲程度大的曲面保拓扑能力较差d
#    
