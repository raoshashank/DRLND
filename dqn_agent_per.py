import numpy as np
import random
from collections import namedtuple, deque
from SumTree import SumTree
from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size,seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= BATCH_SIZE:
                #print("---Learning---")
                sample_outputs = self.memory.sample()
                experiences = sample_outputs[0:5]
                is_weights = sample_outputs[5]
                indices = sample_outputs[6]
                self.learn(experiences, GAMMA,is_weights,indices)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma,is_weights,indices):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        #Double DQN: choose action using Local DQN and evaluate using Target DQN
        # Compute Q targets for current states 
        #print(len(states))
        action_select = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1,action_select)
        
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss using importance sampling weights
        is_weights = torch.from_numpy(is_weights).detach().float().to(device)
        loss = (is_weights*((Q_expected - Q_targets)**2)).mean()
        abs_loss = (abs(Q_expected-Q_targets)).detach().cpu().numpy()
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #update memory with new priorities
        self.memory.update(indices,abs_loss)

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


            
#in order to use Prioritized ER, we need 3 more hyperparameters, and a prioritized sampling distribution 
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.e=0.01          #offset to prevent zero TD error samples from getting neglected
        self.a=0.6           #priority v/s random distribution
        self.b=0.4           #correction term for biased distribution, annealed to 1 
        self.b_decay=0.99   #annealing factor for b
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])        
        self.action_size = action_size
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        #self.memory.append(e)
        # For the time being, new node is appended as leaf with max priority and then updated later on
        max_priority = np.max(self.tree.tree[-self.tree.size:])
        if max_priority==0:
            max_priority = 1
        self.tree.add(e,max_priority)
    
    def sample(self):
        '''
        In order to sample stochastically using proportional prioritization,
        each sample has a priority of pi = TD_error + e
        the probability of a sample i being picked is given by pi**a / sum(pi**a)
        paper mentions that "To sample a minibatch of size k, [0,p_total] is divided into k equal ranges]
        next,  a value is uniformly sampled from each range,samples corresponding to this priority is retreived from the 
        sumtree. 
        '''
        #experiences = random.sample(self.memory, k=self.batch_size)
        indices = np.empty((BATCH_SIZE,),dtype=np.int32)
        is_weights = np.empty((BATCH_SIZE,1), dtype=np.float32) 
        experiences = []
        p_total = self.tree.total_priority()
        segments = np.linspace(0,p_total,num=BATCH_SIZE+1)
        self.b = np.min([1.0,self.b/self.b_decay])        #anneal b per sampling step
        '''
        For importance sampling, the max weight was capped at 1 for stable learning. 
        wi = (BATCH_SIZE * P(i))**(-b)
        so, w_max corresponds to P(p_minimum)
        '''
        p_min = np.min(self.tree.tree[-self.tree.size:])/self.tree.total_priority()
        max_weight = (BATCH_SIZE * p_min)**(-self.b)
        
       
        for i,(a,b) in enumerate(zip(segments,segments[1:])):
            p = np.random.uniform(a,b)                      #sample uniformly across segments
            data,priority,ind = self.tree.get_leaf(p)         #get sample from tree for value
            sample_probability = priority/self.tree.total_priority() # since we are saving p^a directly
            is_weights[i,0] =  ((BATCH_SIZE * sample_probability)**(-self.b)) / max_weight
            indices[i]=ind                 #Keep track of which samples are being used to update the priorities later
            experiences.append(data)
        
        #print(type(experiences[0]))
        try:
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        except AttributeError:
            print(type(experiences[0]))
            print(experiences)
            print(experiences[-1])
            print(len(experiences))
        
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return states, actions, rewards, next_states,dones,is_weights,indices


    def update(self,indices,td_error):
            # new priority = td_error + e
            # according to paper, td_errors should be clipped to [-1,1] 
            # update weights of samples corresponding to indices sampled
            priorities = td_error + self.e
            clipped_priorities = np.clip(priorities,-1.0,1.0)
            priorities = np.power(clipped_priorities,self.a)
            for index,priority in zip(indices,priorities):
                self.tree.update(index,priority)
                                        

    def __len__(self):
        """Return the current size of internal memory."""
        #data_pointer is incremented every time a new experience is added
        return self.tree.data_pointer