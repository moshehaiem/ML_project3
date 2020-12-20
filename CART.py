import numpy as np
from collections import Counter
import sys

# Below is a class that implement the binary tree search. 
# It shows how recursion is done in a typical binary tree structure. 
# You may apply similar ideas (recursion) in CART as you need to recursively split the left and right node until a stop criterion is met.
# For your reference only. 
NULL = 0 
class bsearch(object):
    '''
    binary search tree, with public functions of search, insert and traversal
    '''
    def __init__ (self, value) :
        self.value = value
        self.left = self.right = NULL

    def search(self, value) :
        if self.value==value :
            return True 
        elif self.value>value :
            if self.left==NULL :
                return False 
            else:
                return self.left.search(value)
        else :		
            if self.right==NULL : 
                return False 
            else :
                return self.right.search(value)

    def insert(self, value) :
        if self.value==value :
            return False 
        elif self.value>value :
            if self.left==NULL :
                self.left = bsearch(value)
                return True 
            else :
                return self.left.insert(value)
        else :
            if self.right==NULL :
                self.right = bsearch(value)
                return True 
            else :
                return self.right.insert(value)

    def inorder(self)  :
        if self.left !=NULL  :
            self.left.inorder()
        if self != NULL : 
            print (self.value, " ", end="")
        if self.right != NULL : 
            self.right.inorder()


# -------------------------------Main code starts here-------------------------------------#
class TreeNode(object):
    '''
    A class for storing necessary information at each tree node.
    Every node should be initialized as an object of this class. 
    '''
    def __init__(self, d=None, threshold=None, l_node=None, r_node=None, label=None, is_leaf=False, gini=None, n_samples=None):
        '''
        Input:
            d: index (zero-based) of the attribute selected for splitting use. int
            threshold: the threshold for attribute d. If the attribute d of a sample is <= threshold, the sample goes to left 
                       branch; o/w right branch. float
            l_node: left children node/branch of current node. TreeNode
            r_node: right children node/branch of current node. TreeNode
            label: the most common label at current node. int/float
            is_leaf: True if this node is a leaf node; o/w False. bool
            gini: stores gini impurity at current node. float
            n_samples: number of samples at current node. int
        '''
        self.d = d
        self.threshold = threshold
        self.l_node = l_node
        self.r_node = r_node
        self.label = label
        self.is_leaf = is_leaf
        self.gini = gini
        self.n_samples = n_samples


def load_data(fdir):
    '''
    Load attribute values and labels from a npy file. 
    Data is assumed to be stored of shape (N, D) where the first D-1 cols are attributes and the last col stores the labels.
    Input:
        fdir: file directory. str
    Output:
        data_x: feature vector. np ndarray
        data_y: label vector. np ndarray
    '''

    data = np.load(fdir)
    data_x = data[:, :-1]
    data_y = data[:, -1].astype(int)
    print(f"x: {data_x.shape}, y:{data_y.shape}")
    return data_x, data_y


class CART(object):
    '''
    Classification and Regression Tree (CART). 
    '''
    def __init__(self, max_depth=None, giniInit = 0, rowSizeInit = 1):
        '''
        Input:
            max_depth: maximum depth allowed for the tree. int/None.
        Instance Variables:
            self.max_depth: stores the input max_depth. int/inf
            self.tree: stores the root of the tree. TreeNode object
        '''
        self.max_depth = float('inf') if max_depth is None else max_depth 
        self.tree = None
        self.giniInit = giniInit
        self.rowSizeInit = rowSizeInit
        ###############################
        # TODO: your implementation
        # Add anything you need
        ###############################
        pass

    # function to find the threshold value between data to use for gini impurity
    def findDVal(self, smallestDandGini):
        sizeDandGini = len(smallestDandGini)
        dVal = 0
        currentGini = float('inf')
        currentdAverage = float('inf')
        for index in range(sizeDandGini):
            if smallestDandGini[index][1] < currentGini:
                currentGini = smallestDandGini[index][1]
                currentdAverage = smallestDandGini[index][0]
                dVal = smallestDandGini[index][2]
        return currentGini, currentdAverage, dVal
    

    #recursive function to make the decision tree
    def treeNodeRecursive(self, nodeDepth, dCol, giniTarget, gini = 10, currentdAverage = 0, prune = True):
        dColSize = dCol.shape[0]
        xShape = dCol.shape[1]

        #base conditions are to check for pruning, to check if column is 0, and to check if maximum depth has been reached
        if dColSize == 0:
            return None
        
        if prune == True:
            if dColSize < self.rowSizeInit:
                return TreeNode(label=np.bincount(giniTarget).argmax(), gini = gini, threshold = currentdAverage, is_leaf=True, n_samples=dColSize)

            if gini < self.giniInit:
                return TreeNode(label=np.bincount(giniTarget).argmax(), gini = gini, is_leaf=True, threshold = currentdAverage, n_samples=dColSize)
        

        if nodeDepth == self.max_depth: 
            return TreeNode(label=np.bincount(giniTarget).argmax(), is_leaf=True, n_samples=dColSize)

        if np.all(giniTarget == giniTarget[0]): 
            return TreeNode(label=giniTarget[0], gini = 0, is_leaf=True, n_samples=dColSize)

        #collect gini values and threshold values
        smallestDandGini = []
        for col in range(xShape):
            UnsortedcolumnD = dCol[:, col].reshape(-1)
            giniTarget = giniTarget.reshape(-1)
            smallestDandGiniVals = self.getGini(UnsortedcolumnD, giniTarget, col)
            smallestDandGini.append(smallestDandGiniVals)


        currentGini, currentdAverage, dVal = self.findDVal(smallestDandGini)


        #count the amount of bad good and normal wine
        badCount = 0
        normalCount = 0
        goodCount = 0
        tempGiniTarget = giniTarget.reshape(-1)
        sizeGini = len(tempGiniTarget)
        for gin in range(sizeGini):
            if (tempGiniTarget[gin] == 1): 
                normalCount += 1
            if (tempGiniTarget[gin] == 0): 
                badCount += 1
            if (tempGiniTarget[gin] == 2): 
                goodCount += 1
        giniNew =  1 - (badCount/sizeGini)**2 - (normalCount/sizeGini)**2 - (goodCount/sizeGini)**2



        dColCopy = dCol.copy()
        ab = dColCopy[:,dVal] > currentdAverage
        bel = dColCopy[:,dVal] <= currentdAverage
        dAbove = dColCopy[ab]
        dBelow = dColCopy[bel]
        giniAbove = giniTarget[ab]
        giniBelow = giniTarget[bel]



        #if prune conditions are true, do the below
        if prune == True:
            l_node_rec = self.treeNodeRecursive(nodeDepth+1, dBelow, giniBelow, giniNew, currentdAverage, True)
            r_node_rec = self.treeNodeRecursive(nodeDepth+1, dAbove, giniAbove, giniNew, currentdAverage, True)
        
        if prune == False:
            l_node_rec = self.treeNodeRecursive(nodeDepth+1, dBelow, giniBelow, giniNew, currentdAverage, False)
            r_node_rec = self.treeNodeRecursive(nodeDepth+1, dAbove, giniAbove, giniNew, currentdAverage, False)

        currentNode = False
        if (l_node_rec == None) or (r_node_rec == None):
            currentNode = True
        
        if currentNode == True:
            l_node_rec = False
            r_node_rec = False

        


        #return a recursive call to this function with the new values specified above
        return TreeNode(dVal, currentdAverage, l_node_rec, r_node_rec, np.bincount(giniTarget).argmax(), currentNode, giniNew, dCol.shape[0])


    #function to get gini for a given column with threshold value and target
    def getGini(self, UnsortedcolumnD, giniTarget, dVal):
        SortedDColumn = np.unique(np.sort(UnsortedcolumnD))
        sizeD = SortedDColumn.size
        DAverageArray = np.empty(sizeD)
        if sizeD == 1:
            return SortedDColumn[0], 1, dVal

        for D in range(sizeD-1):
            DAverageArray[D] = np.average([SortedDColumn[D], SortedDColumn[D+1]])
        smallestGini = float('inf')
        smallestDAverage = 0

        for DValue in DAverageArray:
            targetBelowD = [0, 0, 0, 0]
            targetAboveD = [0, 0, 0, 0]

            for D in range(UnsortedcolumnD.size):
                if(UnsortedcolumnD[D] > DValue):
                    if (giniTarget[D] == 0):
                        targetAboveD[0]+=1
                    elif (giniTarget[D] == 1):
                        targetAboveD[1]+=1
                    else:
                        targetAboveD[2]+=1
                else:
                    if (giniTarget[D] == 0):
                        targetBelowD[0]+=1
                    elif (giniTarget[D] == 1):
                        targetBelowD[1]+=1
                    else:
                        targetBelowD[2]+=1
            #get overall total for above and below.  
            totalBelowD = 0
            totalAboveD = 0
            for index in range(3):
                totalAboveD += targetAboveD[index]
                totalBelowD += targetBelowD[index]
            
            setToInfAbove = False
            setToInfBelow = False
            if totalAboveD == 0:
                totalAboveD = float('inf')
                setToInfAbove = True

            if totalBelowD == 0:
                totalBelowD = float('inf')
                setToInfBelow = True

            targetBelowD[3] = 1 - ((targetBelowD[0]/totalBelowD)**2)- ((targetBelowD[1]/totalBelowD)**2)- ((targetBelowD[2]/totalBelowD)**2)

            targetAboveD[3] = 1 - ((targetAboveD[0]/totalAboveD)**2)- ((targetAboveD[1]/totalAboveD)**2)- ((targetAboveD[2]/totalAboveD)**2)

            if setToInfAbove:
                totalAboveD = 0
            if setToInfBelow:
                totalBelowD = 0

            currentGini = (totalAboveD/UnsortedcolumnD.size) * targetAboveD[3]+ (totalBelowD/UnsortedcolumnD.size) * targetBelowD[3]

            if currentGini < smallestGini:
                smallestDAverage = DValue
                smallestGini = currentGini
        return smallestDAverage, smallestGini, dVal


    def testHelper(self, tree, rowSize):
        if tree.is_leaf:
            return tree.label
        if tree.threshold <= rowSize[tree.d]:
            return self.testHelper(tree.r_node, rowSize)
        else:
            return self.testHelper(tree.l_node, rowSize)
            
            

            

            



    def train(self, X, y):
        '''
        Build the tree from root to all leaves. The implementation follows the pseudocode of CART algorithm.  
        Input:
            X: Feature vector of shape (N, D). N - number of training samples; D - number of features. np ndarray
            y: label vector of shape (N,). np ndarray
        '''
        #populate this list by getting columns of each X and running
        #gini function with y values. This is a list of both D and Gini
        x_copy = X.copy()
        y_copy = y.copy()
        self.tree = self.treeNodeRecursive(0, x_copy, y_copy, prune=False)

    def test(self, X_test):
        '''
        Predict labels of a batch of testing samples. 
        Input:
            X_test: testing feature vectors of shape (N, D). np array
        Output:
            prediction: label vector of shape (N,). np array, dtype=int
        '''
        ###############################
        # TODO: your implementation
        ###############################
        xTest = X_test.copy()
        classX = np.empty(xTest.shape[0])
        for data in range(xTest.shape[0]):
            classX[data] = self.testHelper(self.tree, xTest[data,:])
        return classX
        



    def visualize_tree(self):
        '''
        A simple function for tree visualization.
        '''
        print('ROOT: ')
        def print_tree(tree, indent='\t|', dict_tree={}, direct='L'):
            if tree.is_leaf == True:
                dict_tree = {direct: str(tree.label)}
            else:
                print(indent + 'attribute: %d/threshold: %.5f' % (tree.d, tree.threshold))

                if tree.l_node.is_leaf == True:
                    print(indent + 'L -> label: %d' % tree.l_node.label)
                else:
                    print(indent + "L -> ",)
                a = print_tree(tree.l_node, indent=indent + "\t|", direct='L')
                aa = a.copy()

                if tree.r_node.is_leaf == True:
                    print(indent + 'R -> label: %d' % tree.r_node.label)
                else:
                    print(indent + "R -> ",)
                b = print_tree(tree.r_node, indent=indent + "\t|", direct='R')
                bb = b.copy()

                aa.update(bb)
                stri = indent + 'attribute: %d/threshold: %.5f' % (tree.d, tree.threshold)
                if indent != '\t|':
                    dict_tree = {direct: {stri: aa}}
                else:
                    dict_tree = {stri: aa}
            return dict_tree
        try:
            if self.tree is None:
                raise RuntimeError('No tree has been trained!')
        except:
            raise RuntimeError('No self.tree variable!')
        _ = print_tree(self.tree)

        
#grid search is done here
def GridSearchCV(X, y, depth=[1, 40]):
    '''
    Grid search and cross validation.
    Apply 5-fold cross validation to find the best depth. 
    Input:
        X: full training dataset. Not split yet. np ndarray
        y: full training labels. Not split yet. np ndarray
        depth: [minimum depth to consider, maximum depth to consider]. list of integers
    Output:
        best_depth: the best max_depth value from grid search results. int
        best_acc: the validation accuracy corresponding to the best_depth. float
        best_tree: a decision tree object that is trained with 
                   full training dataset and best max_depth from grid search. instance
    '''
    ###############################
    # TODO: your implementation
    ###############################
    xTemp = X.copy()
    yTemp = y.copy()
    split_x = np.array_split(xTemp, 5, axis=0)
    split_y = np.array_split(yTemp, 5)

    #the split of data is done here (in 5 different ways), and in each iteration, each one is used to find the best value
    best_acc = 0
    best_depth = 0
    for node in range(depth[0], depth[1], 4):
        best_accuracy = 0
        for n in range(5):
            tempx = 0
            tempy = 0
            valx = 0
            valy = 0
            if n == 0:
                tempx = np.vstack((split_x[0], split_x[1], split_x[2], split_x[3]))
                tempy = np.hstack((split_y[0], split_y[1], split_y[2], split_y[3]))
                valx = split_x[4]
                valy = split_y[4]
            elif n == 1:
                tempx = np.vstack((split_x[0], split_x[1], split_x[2], split_x[4]))
                tempy = np.hstack((split_y[0], split_y[1], split_y[2], split_y[4]))
                valx = split_x[3]
                valy = split_y[3]
            elif n == 2:
                tempx = np.vstack((split_x[0], split_x[1], split_x[3], split_x[4]))
                tempy = np.hstack((split_y[0], split_y[1], split_y[3], split_y[4]))
                valx = split_x[2]
                valy = split_y[2]
            elif n == 3:
                tempx = np.vstack((split_x[0], split_x[2], split_x[3], split_x[4]))
                tempy = np.hstack((split_y[0], split_y[2], split_y[3], split_y[4]))
                valx = split_x[1]
                valy = split_y[1]
            else:
                tempx = np.vstack((split_x[1], split_x[2], split_x[3], split_x[4]))
                tempy = np.hstack((split_y[1], split_y[2], split_y[3], split_y[4]))
                valx = split_x[0]
                valy = split_y[0]

            cart = CART(node, .3, 30)
            cart.train(tempx, tempy)
            testVar = cart.test(valx)
            counter = 0
            for i in range(len(testVar)):
                if testVar[i] == valy[i]:
                    counter += 1
            best_accuracy += counter/len(testVar)
        best_accuracy/=5
        if best_accuracy > best_acc:
            best_acc = best_accuracy
            best_depth = node

    best_tree = CART(best_depth, .3, 30)
    best_tree.train(X, y)
    return best_depth, best_acc, best_tree



X_train, y_train = load_data('winequality-red-train.npy')
cart = CART(8)
cart.train(X_train, y_train)
cart.visualize_tree()
cart.test(X_train)
# classX = cart.test(X_train)
# counter = 0
# for i in range(len(classX)):
#     if classX[i] == y_train[i]:
#         counter+=1
# print(counter)
best_depth, best_acc, best_tree = GridSearchCV(X_train, y_train, [1, 40])
# print(best_depth)
# print(best_acc)
print('Best depth from 5-fold cross validation: %d' % best_depth)
print('Best validation accuracy: %.5f' % (best_acc))
# print('Best depth from 5-fold cross validation: %d' % best_depth)
# print('Best validation accuracy: %d' % (best_acc))
 
