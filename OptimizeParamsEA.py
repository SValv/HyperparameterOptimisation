#%% Imports
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import copy
import math
import time

# We import os and joblib to save models and data - but its not used right now
import os
from joblib import dump, load

#%% Create Timeout Exception

from contextlib import contextmanager
import threading
import _thread

class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg

@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()

#%% Clamp helper

clamp = lambda n, minn, maxn: max(min(maxn, n), minn) 

#%% Classes
class population:
    
    def __init__(self,psize):
        
        self.size=psize
        self.individuals=list()
        self.best=None
        self.generation=0
        self.kernelFunctions=pd.DataFrame()
        for i in range(0, self.size):
            
            if i < self.size*0.16:
                Simonshinterkopfwirdkahl="linear"
            elif i < self.size*0.18:
                Simonshinterkopfwirdkahl="rbf"
            elif i < self.size*0.8:
                Simonshinterkopfwirdkahl="poly"
            elif i < self.size*1:
                Simonshinterkopfwirdkahl="sigmoid"
            
            gen=Candidate(Simonshinterkopfwirdkahl)
            gen.randinit()
            gen.fitness=-1
            self.individuals.append(copy.copy(gen))
            
        kernels=[]
        for i in self.individuals:
            kernels.append(i.type)
        self.kernelFunctions = pd.concat([self.kernelFunctions, pd.Series(kernels)], axis=1)
            
        self.evaluation()
            
    def evaluation(self):
        self.generation += 1
        print("\n \n Nr of Inds: " + str(len(self.individuals)) + "   in generation " + str(self.generation) + " \n \n")
        
        for i in range(len(self.individuals)):
            print(i)
            if self.individuals[i].fitness == -1:
                self.individuals[i].candeval()
                if self.best==None:
                    self.best=copy.copy(self.individuals[i])
                    print("erstes Maximum mit Accuracy "+str(self.best.fitness))
                elif self.individuals[i].fitness > self.best.fitness:
                    self.best=copy.copy(self.individuals[i])
                    print("neues Maximum in Generation "+str(self.generation)+" mit Accuracy "+str(self.best.fitness))
    
    def selection(self, strategie="plus"):
        if strategie=="komma":
            while len(self.individuals) < 2.5*self.size: # 2.5 arbitrary -> parameter?
                add = np.random.choice(self.individuals,1)[0].mutate()
                add.candeval()
                self.individuals.append(add)
            self.individuals = self.individuals[self.size:]
        self.individuals.sort(key = lambda x: x.fitness, reverse=True)
        self.individuals = copy.copy(self.individuals[:self.size])
        
        kernels=[]
        for i in self.individuals:
            kernels.append(i.type)
        self.kernelFunctions = pd.concat([self.kernelFunctions, pd.Series(kernels)], axis=1)
        print("Best of current Gen: " + str(self.best.fitness))
        
    def crossover(self, prop = .2):
        lm      = []
        rbf     = []
        poly    = []
        sigmoid = []
        for i in self.individuals:
            if i.type== "linear" and np.random.rand() <= prop: 
                lm.append(copy.copy(i))
            elif i.type=="poly" and np.random.rand() <= prop:
                poly.append(copy.copy(i))
            elif i.type=="sigmoid" and np.random.rand() <= prop:
                sigmoid.append(copy.copy(i))
            elif i.type=="rbf" and np.random.rand() <= prop:
                rbf.append(copy.copy(i))
        lm = self.crossover_inner(lm)
        poly = self.crossover_inner(poly)
        sigmoid = self.crossover_inner(sigmoid)
        rbf = self.crossover_inner(rbf)
        self.individuals = self.individuals + lm + poly + sigmoid + rbf

    def crossover_inner(self, liste):
        l = []
        for i in range(len(liste)//2):
            #Select two random parents
            np.random.shuffle(liste)
            x = liste.pop(); x.fitness = -1
            y = liste.pop(); y.fitness = -1

            #Create offspring via uniform crossover
            for attribute in x.solution:
                p = np.random.uniform(0,1)
                if p > 0.5:
                    x_val = x.solution[attribute]
                    y_val = y.solution[attribute]
                    x.solution[attribute] = y_val
                    y.solution[attribute] = x_val
            l.append(x);l.append(y)  # evtl copy
        if liste:
            l.append(liste[0])
        return l
            
    def mutation(self, prop = .2):
        for i in range(self.size):
            if np.random.rand() < prop:
                self.individuals.append(self.individuals[i].mutate())        
    
    def run(self,maxgen,termination):
        while True:
            if maxgen <= 0 or self.best.fitness > termination:
                print("Done!")
                break
            self.crossover()
            self.mutation()
            self.evaluation()
            self.selection()
            maxgen -= 1
            
     
class Candidate: 
    
    def __init__(self,typee):
        
        self.solution=dict()
        self.fitness=None
        self.type=typee
                
    def randinit(self):
        # Initialize encoding
        
        if self.type== "linear":
            self.solution= {"C":1, "tol": 0.001, "shrinking":  True}
        elif self.type=="poly":
            self.solution= {"C":1, "degree": 3, "gamma":0.1, "coef0":1, "tol": 0.001, "shrinking": True}
        elif self.type=="sigmoid":
            self.solution= {"C":1, "gamma":0.1, "coef0":1, "tol": 0.001, "shrinking": True}
        elif self.type=="rbf":
            self.solution= {"C":1, "gamma":0.1, "tol": 0.001, "shrinking": True}
        self.variations=np.repeat(1, len(self.solution))
        self.r1 = 1/math.sqrt(2*len(self.solution))
        self.r2 = 1/math.sqrt(2*math.sqrt(len(self.solution)))
        
    def candeval(self):
        try:
            with time_limit(60):
                SVM=svm.SVC(kernel = self.type, **self.solution)
                SVM.fit(X1_train, y1_train)
                Score1=SVM.score(X1_test, y1_test)
                SVM.fit(X2_train, y2_train)
                Score2=SVM.score(X2_test, y2_test)
                SVM.fit(X3_train, y3_train)
                Score3=SVM.score(X3_test, y3_test)
                self.fitness=np.mean([Score1,Score2,Score3])
                self.fitness=Score1
        #Sometimes an SVM takes an eternity to fit given certain Parameters, so we use this fix - not optimal
        except TimeoutException as e:          
            self.fitness = 0
            print("Timed out! Set fitness to 0")
        print(self.solution)
        print(self.type)
        print(self.fitness)

    def update_variances(self):
        N = np.random.normal(0,1,1)[0]
        self.variations = [i*math.exp(self.r1*N + self.r2 * np.random.normal(0,1,1)[0]) for i in self.variations]
        
    def mutate(self):
        # implementation of local variance adaptation
        self.update_variances()
        newInd = Candidate(self.type)
        newInd.randinit()
        for i,j in zip(self.solution.keys(), self.variations):
            newInd.solution[i] = max(0.0001, self.solution[i] + np.random.normal(0,j,1)[0])
        newInd.solution["shrinking"] = bool(np.random.randint(0,1))
        newInd.solution["C"] = clamp(newInd.solution["C"], 1,100)
        newInd.fitness = -1
        return newInd
            

#%% Data

## Read Data 

url="http://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
df = pd.read_csv(url, names=['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
                             '25','26','27','28','29','30','31','32','33','34'])
print (df.head())

#%%
## Variante 1: Train und test data im vorhinein 

y = df["34"].values
X = df.drop("34", axis=1).values

X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size = 0.2)

X2_train, X2_test, y2_train, y2_test = train_test_split(X, y, test_size = 0.3)

X3_train, X3_test, y3_train, y3_test = train_test_split(X, y, test_size = 0.2)

#%% Main & Test

p = population(100)
start = time.time()
p.run(25,1)
print("Finnished in " + str(time.time() - start))