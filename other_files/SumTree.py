'''
This Sum Tree implementation is from Simonini Thomas's Deep RL course: https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20%28%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay%29.ipynb
'''
import numpy as np

class SumTree:
    #Binary SumTree: leaves contain priorities and data array contains index to leaves
    #index of each leaf in sum tree is index of experience in data
    # for tree of size n, the leaf nodes have index n/2, n/2+1,n/2+2....n or, tree[-size:] are leaf nodes
    
    def __init__(self,size):
        self.data_pointer = 0
        self.size=size #number of leaf nodes
        #initialize Tree with zero nodes
        self.tree = np.zeros(2*size - 1) # Each node has 2 children, and root node is counted twice
        #initialize data array with zeroes
        self.data = np.zeros(size, dtype = object) # we are storing pointers to other data. so we can perform operations on this object
        
    def add(self,data,priority): 
        #add priority score to leaf and experience to data
        
        index = self.data_pointer+self.size-1 #Calculate index for new entry
        self.data[self.data_pointer] = data   #Insert new entry as to data array
        #print(priority)
        self.update(index,priority)           #update table
        self.data_pointer+=1                  #update tree pointer
        if self.data_pointer>=self.size :  
            self.data_pointer = 0             # Overwrite if we exhaust array
        
        
    def update(self,index,priority):
        #update leaf priority by percolation and update priority of previous samples
        # for ith node, (i-1)//2 is parent, 2i+1 is left child and 2i+2 is right child 
        
        delta = priority - self.tree[index]
        #print(index)
        self.tree[index] = priority
        while index!=0:
            index = (index-1)//2            #index to parent node
            self.tree[index]+=delta    # add the change to update values of parent to parent+change
    
    def get_leaf(self,value):
        #get priority score, experience tuple and index for leaf given value of leaf
        parent = 0
        while True: 
            left_child = 2*parent+1
            right_child = left_child+1   
            
            if left_child >= len(self.tree):
                leaf = parent
                break
           
            #Tree data is indexed from left to right 
            else:
                if value<=self.tree[left_child]:
                    parent = left_child                 #Follow left sub tree
                else:
                    value-= self.tree[left_child]   # get the remainder and follow right sub tree
                    parent=right_child
        #index of data and tree follows the equation tree_pointer = tree_size -1 + array_index 
        data_index = leaf + 1 - self.size
        return self.data[data_index],self.tree[leaf],leaf
        
    def total_priority(self):
        #get value of total priority from root node.
        #since the tree is a sum tree, the total priority is just the value of the root node
        return self.tree[0] 

