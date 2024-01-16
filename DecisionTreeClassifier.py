#!pip install category_encoders
#!pip install deepdiff
import pandas as pd
import numpy as np
from array import array
from copy import deepcopy
from deepdiff import DeepDiff
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from category_encoders import *
from typing import List, Union, Tuple
import pprint





def compute_impurity (target_vector: np.array, criterion: str = 'gini') -> float:
    unique_vals, probas=np.unique(target_vector, return_counts=True)
    probas=probas/np.sum(probas)
    if criterion=='gini' :
        return probas@(1-probas)
    else :
        return -np.sum([p*np.log2(p) for p in probas if p>0])

def compute_criterion(target_vector: np.array, feature_vector: np.array, threshold: float, criterion: str = 'gini') -> float:
    assert criterion in ['gini', 'entropy'], "Критерий может быть только 'gini' или 'entropy'!"

    pass
    if  len(np.unique(feature_vector))==1 :
        return 0
    n=len(target_vector)
    n_left=sum(feature_vector<=threshold)
    n_right=sum(feature_vector>threshold)
    H_R=compute_impurity(target_vector=target_vector, criterion=criterion)
    H_Rl=compute_impurity(target_vector=target_vector[feature_vector<=threshold], criterion=criterion)
    H_Rr=compute_impurity(target_vector=target_vector[feature_vector>threshold], criterion=criterion)
    Q=H_R-n_left/n*H_Rl-n_right/n*H_Rr

    return Q


def find_best_split(feature_vector: np.ndarray, target_vector: np.ndarray, criterion: str = 'gini') -> Tuple:
    """
    Функция, находящая оптимальное рабиение с точки зрения критерия gini или entropy

    Args:
        feature_vector: вещественнозначный вектор значений признака
        target_vector: вектор классов объектов (многоклассовый),  len(feature_vector) == len(target_vector)
    Returns:
        thresholds: (np.ndarray) отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
                     разделить на две различные подвыборки, или поддерева
        criterion_vals: (np.ndarray) вектор со значениями критерия Джини/энтропийного критерия для каждого из порогов
                в thresholds. len(criterion_vals) == len(thresholds)
        threshold_best: (float) оптимальный порог
        criterion_best: (float) оптимальное значение критерия

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    """

    unq_vals = np.sort(np.unique(feature_vector))

    if len(unq_vals) == 1:
        return None, None, None, 0

    pass
    thresholds=(unq_vals[1:]+unq_vals[:-1])/2
    criterion_vals=np.array([compute_criterion(target_vector=target_vector, feature_vector=feature_vector, threshold=t, criterion=criterion) for t in thresholds])
    threshold_best=thresholds[0]
    criterion_best=criterion_vals[0]
    for t in range(len(criterion_vals)) :
        if criterion_vals[t]>criterion_best :
            criterion_best=criterion_vals[t]
            threshold_best=thresholds[t]
    return thresholds, criterion_vals, threshold_best, criterion_best



class DecisionTree(BaseEstimator):

    def __init__(
            self,
            feature_types: list,
            criterion: str = 'gini',
            max_depth: int = None,
            min_samples_split: int = None,
    ):
        """
        Args:
            feature_types: список типов фичей (может состоять из 'real' и "categorical")
            criterion: может быть 'gini' или "entropy"
            max_depth: максимальная глубина дерева
            min_samples_split: минимальное число объектов в листе, чтобы можно было расщиплять этот лист
        """

        self._feature_types = feature_types
        self._tree = {}
        # Сюда будут сохраняться обученные таргет энкодеры категориальных фичей
        self.target_encodings = {} # Dict[int<номер категориальной фичи>, category_encoders.target_encoder.TargetEncoder]
        self._criterion = criterion
        self.max_depth = max_depth
        self._min_samples_split = min_samples_split
        self.cols=[]
        self.enc=1

    def _fit_node(self, sub_X: np.ndarray, sub_y: np.ndarray, node: dict, level: int):
        """
        Ищет оптимальное расщепление для листа, состоящего из объектов sub_X и таргетов sub_y. Если для данного листа
        выполненые критерии останова - то завершает работу и обозначает тип листа терминальным (type="terminal")
        Args:
            sub_X: array размера (n, len(self._feature_types)), матрица объект-признак для объектов, попавших в текущих
                лист
            sub_y: array размера (n,) - вектор таргетов для объектов, попавших в текущих лист
            node: словарь, содержащий дерево, обученное к текущему моменту
        Returns:
                None
        ***
        В случае если фича типа "categorical" - нужно применить к ней таргет энкодинг и записать обученный энкодинг в
            self.target_encodings
        ***
        По сути этот метод нужен для рекурсивного вызова в методе self.fit(). Его цель - заполнение словаря node
            (он же self._tree)
        в node (self._tree) в результате обучения должны быть следующие ключи:
            "type" - может быть "terminal" или "nonterminal" (тип текущей вершины: лист ("terminal")
                или внутренняя вершина ("nonterminal"))
                Для листьев (вершин типа "terminal") должны быть следующие ключи:
                    "classes_distribution": список или np.ndarray с распределением классов в данном листе
                        (число объектов каждого класса)
                Для внутренних вершин (типа "nonterminal") должны быть следующие ключи:
                    "feature_type" - (str) тип переменной, по которой эта вершина дальше разделяется ("real" или "categorical")
                    "feature_number" - (int) номер переменной (начиная с нуля), по которой проиходит дальнейшее разделение
                        этой вершины
                    "threshold" - (float) порог рабиения
                    (Иными словами, дальнейшее разбиение этой вершины происходит по формуле:
                        [sub_X[:, feature_number] < threshold])
        Примеры обученных деревьев (self._tree):
            {
                'type': 'nonterminal',
                 'feature_type': 'real',
                 'feature_number': 1,
                 'threshold': 0.535,
                 'left_child': {
                          'type': 'nonterminal',
                          'feature_type': 'real',
                          'feature_number': 0,
                          'threshold': -0.408,
                          'left_child': {'type': 'terminal', 'classes_distribution': [84, 5]},
                          'right_child': {'type': 'terminal', 'classes_distribution': [99, 466]}
                      },
                 'right_child': {
                             'type': 'nonterminal',
                              'feature_type': 'categorical',
                              'feature_number': 3,
                              'threshold': 1.443,
                              'left_child': {'type': 'terminal', 'classes_distribution': [315, 13]},
                              'right_child': {'type': 'terminal', 'classes_distribution': [2, 16]}
                    }
            }
            Обратите внимание, что порядок в classes_distribution должен совпадать с порядком в таргете
                (то есть на нулевом месте - число объектов нулевого класса, на первом - первого и тд.)
        """
        level+=1

        if level==self.max_depth or len(sub_X)<self._min_samples_split or len(np.unique(sub_y))==1 :
            
            dic={}
            unique_vals=self.classes
            probas=[0 for i in range(len(unique_vals))]
            for k in range(len(unique_vals)) :
                for r in range(len(sub_y)) :
                    if sub_y[r]==unique_vals[k] :
                        probas[k]+=1
            dic={'type': 'terminal', 'classes_distribution': np.array(probas)}
            #dic.update({'type': 'terminal'})
            #dic.update({'classes_distribution': probas}) 
        else :
            
            dic={}
            indicator=0
            for i in range(len(self._feature_types)) :
                feature_vector=sub_X[:, i]
                if len(np.unique(feature_vector))!=1 :
                    ans=find_best_split(feature_vector=feature_vector, target_vector=sub_y, criterion=self._criterion)
                    thresholds=ans[0]
                    criterion_vals=ans[1]
                    for j in range(len(thresholds)) :
                        if indicator==0 :
                            threshold_best= thresholds[j]
                            criterion_best=criterion_vals[j]
                            feature_best=i
                            indicator=1 
                        else :
                            if criterion_vals[j]>criterion_best :
                                threshold_best= thresholds[j]
                                criterion_best=criterion_vals[j]
                                feature_best=i
            
            feature_vector=sub_X[:, feature_best]
            if self._feature_types[feature_best]=='categorical' :
                
                dic={'type': 'nonterminal', 
                     'feature_type':  'categorical', 
                     'feature_number': feature_best, 
                     'threshold' : threshold_best,
                     'left_child': self._fit_node(sub_X=sub_X[feature_vector<=threshold_best], sub_y=sub_y[feature_vector<=threshold_best], node=self._tree, level=level),
                     'right_child': self._fit_node(sub_X=sub_X[feature_vector>threshold_best], sub_y=sub_y[feature_vector>threshold_best], node=self._tree, level=level)
                     }
                
            else :
                
                dic={'type': 'nonterminal', 
                     'feature_type':  'real', 
                     'feature_number': feature_best, 
                     'threshold' : threshold_best,
                     'left_child':  self._fit_node(sub_X=sub_X[feature_vector<=threshold_best], sub_y=sub_y[feature_vector<=threshold_best], node=self._tree, level=level),
                     'right_child': self._fit_node(sub_X=sub_X[feature_vector>threshold_best], sub_y=sub_y[feature_vector>threshold_best], node=self._tree, level=level)
                     }
            #dic.update({'feature_number': feature_best})
            #dic.update({'threshold' : threshold_best })
           # feature_vector=sub_X[:, feature_best]
           # dic.update({'left_child': self._fit_node(sub_X=sub_X[feature_vector<=threshold_best], sub_y=sub_y[feature_vector<=threshold_best], node=self._tree, level=level)})
           # dic.update({'right_child': self._fit_node(sub_X=sub_X[feature_vector>threshold_best], sub_y=sub_y[feature_vector>threshold_best], node=self._tree, level=level)})
        return dic

    def _predict_proba_object(self, x: np.array, node: dict) -> Union[List, np.ndarray]:
        """
        Должен либо вернуть распределение классов для объекта х (Это будет нормированный classes_distribution
        из терминального листа), либо рекурсивно просеить его в левое или правое поддерево.
        Args:
            x: объект размера (len(self._feature_types),)
            node: обученное дерево, которое нужно применить для предсказания
        """
        #pprint.pprint(node['right_child'])
        if  node['type']=='terminal' :
            ans=node['classes_distribution']/np.sum(node['classes_distribution'])
            return ans
        else :
            if node['threshold']>x[node['feature_number']] :
                return self._predict_proba_object(x=x, node=node['left_child'])
            else :
                return self._predict_proba_object(x=x, node=node['right_child'])
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: матрица объект-признак размеров (n, len(self._feature_types))
            y: вектор таргетов (состоящий из int) размера (n,)
        """
        assert len(set(y)) > 1, 'Таргет должен содержать более одного класса!'
        
        X=pd.DataFrame(X)
        y=pd.DataFrame(y)
        
        
        for i in range(len(self._feature_types)) :
            if self._feature_types[i]=="categorical" :
                self.cols.append(i)
        #print(cols)
        if len(self.cols)!=0 :
            self.enc=TargetEncoder(cols=self.cols).fit(X, y)
            #print(self.enc)
            #print(self.enc.mapping)
            X=self.enc.transform(X)
        X=X.to_numpy()
        y=y.to_numpy()
        self.classes=np.unique(y)
        self.classes=np.sort(self.classes)
        if self._min_samples_split==None :
            self._min_samples_split=2
        self._tree =self._fit_node(sub_X=X, sub_y=y, node=self._tree, level=0)
        #pprint.pprint(self._tree)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Применяет self._predict_proba_node для каждой строки из X
        Args:
            X: множество объектов, для которых сделать предсказание, матрица размеров (m, len(self._feature_types))
        Returns:
            np.ndarray размера (len(X), len(set(y)) (где y - вектор таргетов, участвовавший в обучении (self.fit))
        """
        assert self._tree != {}, "Cначала обучите модель!"
        if len(self.cols)!=0 :
            X=pd.DataFrame(X)
            X=self.enc.transform(X)
            X=X.to_numpy()
       
        predicted = []
        for x in X:
            a=self._predict_proba_object(x, self._tree)
            predicted.append(a)
        return np.array(predicted)


    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X=X), axis=1).ravel()
