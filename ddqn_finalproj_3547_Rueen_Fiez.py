####################################################################################################################################################################################################################################################################################################################################################
'''
Author:	Rueen Fiez Moghaddam
Date:	March 19, 2020
Desc:	Double Deep Q Learning performing in the Gym env's Lunar Lander (v2)
Refs:	<> https://towardsdatascience.com/double-deep-q-networks-905dd8325412
		<> https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287
		<> https://towardsdatascience.com/a-beginners-guide-to-q-learning-c3e2a30a653c
		<> https://www.youtube.com/watch?v=wYIiMH1cIis,https://www.youtube.com/watch?v=jDll4JSI-xo&t=17s,https://www.youtube.com/watch?v=UCgsv6tMReY&t=241s
		<> Deep Reinforcement Learning with Double Q-learning (Hado van Hasselt and Arthur Guez and David Silver) - https://arxiv.org/pdf/1509.06461.pdf
		<> Reinforcement Learning - An Introduction, 2nd edition by Richard Sutton & Andrew G Barto.
Notes:
		<> Epsilon greedy means that if less than epsilon you take a random action and if greater than epsilon then take a Greedy action. 
			<-> Epsilon decays over time, at the beginning of the episodes you take random actions and over time you choose more greedy actions. Epsilon will eventually due to decay settle on a "completed" or "end" value so that the Agent will take way more Greedy actions than random ones over time.
		<> The Deep Neural Network will approximate the action-value function
		<> The Agent has memory, selecting action, managing memories, all contained within its own class.
		<> In a Deep Q Learning it uses a single network to select the action and determine the value of that action, using an argmax which leads to over estimation / over-estimating  the value of the action-value function. This creates a MAXIMIZATION BIAS
			<-> In Double Deep Q Network it uses two networks to solve the maximization bias problem from Deep Q Learning. One network finds the policy, and one to calculate the value of the state action-value functions.  This way it eliminates the MAXIMIZATION BIAS.
		<> Q Learning is a type of temporal difference learning, it learns at each time step of the episode.
'''
####################################################################################################################################################################################################################################################################################################################################################

# Load Dependencies
print("\nLoading Dependencies....")
import numpy as np
import gym
from gym import wrappers
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
import time
import shutil
import matplotlib.pyplot as plt


# Global variables & Hyperparameters
DEBUGGING = False  # Will fill up console output with DEBUG info 
PATH_TO_VIDEOS = "C:/Users/rueen/Desktop/AI/UOFT/AI/3547_REIN_LEARNING/FINAL_PROJECT/vids"
'''
Toggles for the agent's visibility and monitoring. Videos and video compilation video or not.
'''
MONITOR_AGENT=True  # Will take longer as it has to render this for each episode. Will also create video per episode in your present working directory folder called "vids" which it will create.
MONITOR_AGENT_WINDOW_DELAY_SECONDS=0.1  # This is how long to keep the rendered visual window open for...
DO_VIDEO_COMPILATION_OF_ALL_EPISODES_AT_END=False  # Toggle if you want a file created called "output" at end in your present working directory (pwd) which is a compilation of all the videos saved in the "vids" folder
#KEEP_INDIVIDUAL_EPISODE_VIDEOS=False   # Toggle to keep the "vids" folder that the video mechanism creates or not. 
NO_OF_EPISODES_TO_AVERAGE_SCORE_FOR=100
'''
These are HYPERPARAMETERS for the Agent's DDQN. 
Could even add more stuff if time for fine-tuning but only limited time :)
'''
NO_OF_EPISODES=500
FREQ_UPDATE_TARGET_NETWORK=100
EPSILON_DECAY_RATE=0.9997
EPSILON_COMPLETE=0.01
EPSILON_RESET_TOGGLE=True
MEMORY_SZ=1000000
DENSE_1_SZ=64
DENSE_2_SZ=64
#CONV1D_SZ=64
DROPOUT_SIZE=0.05

'''
These are used to instantiate the AGENT_DDQN class.
'''
ALPHA=0.0008
GAMMA=0.99
POSSIBLE_ACTIONS=4
EPSILON=1.0
BATCH_SZ=64
INPUT_D=8

PLOT_TITLE = "EPS_1_EPS_COMP_001_EPS_DEC_09997_TGT_NET_UPT_100"

####################################################################################################################################################################################################################################################################################################################################################
# Global methods / functions


'''
This network will approximate the action-value function.
Method to support DQN network creation only.  Note for self:  Requires fine-tuning / hyperparameter tuning after testing....
Basic sequential model only with limited number of layers....... can make more robust as required after testing....
'''
def deep_q_network(learn_rate, possible_actions, input_d, fc1_d, fc2_d):
	dqn_model = Sequential()
	dqn_model.add(Dense(fc1_d, input_shape=(input_d,), activation="relu"))
	#dqn_model.add(BatchNormalization())
	#dqn_model.add(Dropout(DROPOUT_SIZE))
	dqn_model.add(Dense(fc2_d, activation="relu"))
	#dqn_model.add(BatchNormalization())
	#dqn_model.add(Dense(16, activation="relu"))
	dqn_model.add(Dense(possible_actions, activation="relu"))
	dqn_model.compile(loss="mse", optimizer=Adam(lr=learn_rate))
	return dqn_model
 

####################################################################################################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
# Global classes

'''
Class:	AGENT_DDQN
Desc:	Agent handler. Agent has a memory, funcitonality for choosing actions, learning, & storing memories
Params:	<> alpha: learning rate
		<> gamma: discount factor for the agent, means the agent will reduce the contribution of estimates of rewards from future states to its estimates of the value of a given state. ie; Agent is in this state with some reward that has all the value of the reward, next state has some value discounted by factor gamma. 
				  Discounts it because its taking an epsilon greedy strategy where it may or may not end up in set of actions it predicts for the action-value function.
		<> FREQ_UPDATE_TARGET_NETWORK: number of how many episodes before updating the weights of the target network. 
		<> batch_sz: for memory.
		<> input_d:  the input dimensions.
		<> EPSILON_DECAY: how fast to make epsilon go down for the explore vs exploit strategy. 
		<> EPSILON_COMPLETE: what is the minimum of the epsilon to "settle" on.
		<> memory_sz: the memory size to use. 
'''
class AGENT_DDQN(object):

	'''
	Constructor
	'''
	def __init__(self, alpha, gamma, possible_actions, EPSILON, batch_sz, input_d, EPSILON_DECAY=EPSILON_DECAY_RATE, EPSILON_COMPLETE=EPSILON_COMPLETE, memory_sz=MEMORY_SZ, FREQ_UPDATE_TARGET_NETWORK=FREQ_UPDATE_TARGET_NETWORK):
		self.possible_actions = possible_actions
		self.action_space=[i for i in range(self.possible_actions)]
		self.gamma = gamma
		self.EPSILON = EPSILON
		self.EPSILON_DECAY=EPSILON_DECAY
		self.EPSILON_COMPLETE=EPSILON_COMPLETE
		self.batch_sz=batch_sz
		self.FREQ_UPDATE_TARGET_NETWORK=FREQ_UPDATE_TARGET_NETWORK
		self.BUFFER_memory = MEM_BUFFER(memory_sz, input_d, possible_actions, True)
		self.Q_EVAL = deep_q_network(alpha, possible_actions, input_d, DENSE_1_SZ, DENSE_2_SZ)
		self.Q_TARGET = deep_q_network(alpha, possible_actions, input_d, DENSE_1_SZ, DENSE_2_SZ)   # the "alpha" ie the Learning Rate, is not even useful here because we don't fit, required only by the function.....

	'''
	Method to manage transition. Stores memory.
	'''
	def manage_memory(self, state, action, reward, new_state, done):
		self.BUFFER_memory.manage_transition(state, action, reward, new_state, done)


	'''
	Method to choose an action. Like the Policy. Takes the state as input. Agent will chose an action based on its current state. 
	'''
	def select_action(self, state):
		# Reshape the state, because of the network shape. Allow the network to handle batch training.
		state = state[np.newaxis,:]
		r = np.random.random()  
		if r < self.EPSILON:  # IF random is less than EPSILON we pick an action at random, ELSE calculate the actions predicted by the state 
			action=np.random.choice(self.action_space)
			if DEBUGGING:
				print("DEBUG::: Action = RANDOM...")
		else:
			actions=self.Q_EVAL.predict(state)
			action=np.argmax(actions)  # pick the best action
			if DEBUGGING:
				print("DEBUG::: Action = BEST...")
		return action


	'''
	The learning method. 
	'''
	def learning(self):
		if self.BUFFER_memory.mem_counter>self.batch_sz:   # NOTE: Dont fill up agent's memory with random actions, just make sure that the memory_counter is greater than the batch_sz. 
			state,action,reward,new_state,done=self.BUFFER_memory.sample(self.batch_sz)
			action_vals = np.array(self.action_space, dtype=np.int8) 
			action_indexes = np.dot(action,action_vals)
			Q_FWD=self.Q_TARGET.predict(new_state)
			Q_EVAL=self.Q_EVAL.predict(new_state)
			Q_PRED=self.Q_EVAL.predict(state)
			max_actions=np.argmax(Q_EVAL,axis=1)
			Q_TARGET=Q_PRED
			index_of_batch=np.arange(self.batch_sz, dtype=np.int32)
			Q_TARGET[index_of_batch, action_indexes]=reward+self.gamma*Q_FWD[index_of_batch,max_actions.astype(int)] * done
			a = self.Q_EVAL.fit(state,Q_TARGET, verbose=0)
			# Check to see where EPSILON value stands and reset if required.
			if EPSILON_RESET_TOGGLE:   # Only do this if epsilon reset toggle is set to True 
				if self.EPSILON == self.EPSILON_COMPLETE:
					if DEBUGGING:
						print("DEBUG::: EPSILON reset!")
					self.EPSILON = 1.0  # Re-introduce EPSILON when it's down to minimum.
			# Handling the EPSILON decay
			self.EPSILON = self.EPSILON * self.EPSILON_DECAY if self.EPSILON > self.EPSILON_COMPLETE else self.EPSILON_COMPLETE
			if DEBUGGING:
				print("DEBUG::: current value of EPSILON=",self.EPSILON)

			if self.BUFFER_memory.mem_counter % self.FREQ_UPDATE_TARGET_NETWORK == 0:
				self.network_update()
				if DEBUGGING:
					print("DEBUG::: Target Network updated.")


	def network_update(self):
		self.Q_TARGET.model.set_weights(self.Q_EVAL.model.get_weights())


####################################################################################################################################################################################################################################################################################################################################################

'''
Class:	MEM_BUFFER
Desc:	Allows agent to sample state action reward new state transitions across diff episodes. Provides additional methods for supporting 
Params: <> max_size: max number of memories we want to store
		<> input_shape: the shape of the observation of the environment, for lunar lander vector of 8 elements
		<> n_actions: number of actions, 4 for lunar lander
'''
class MEM_BUFFER(object):

	'''
	Method to manage our transitions into memory  
	'''
	def manage_transition(self, state, action, reward, state_, done):
		idx = self.mem_counter % self.mem_sz  # Find the address of the first available memory. Keep memory finite.
		self.mem_state[idx] = state
		self.mem_new_state[idx] = state_
		if self.discrete:
			actions = np.zeros(self.mem_action.shape[1])
			actions[action] = 1.0
			self.mem_action[idx] = actions
		else:
			self.mem_action[idx] = action
		self.mem_reward[idx] = reward  # Store the current rewards
		self.mem_term[idx] = 1-int(done) # 1-done because we're multiplying by this flag, when the episode is over we want it to be a 0 whereas when episode is over done flag us True
		self.mem_counter = self.mem_counter + 1  # Update memory counter


	'''
	Sample the buffer of memories. Only a subset. We don't wanna sample the zeros, you want to find the memories that have been filled. 
	Agent can sample non sequential memories using below 
	'''
	def sample(self, batch_sz):
		mem_maximum = min(self.mem_counter, self.mem_sz)
		batch = np.random.choice(mem_maximum, batch_sz)
		term = self.mem_term[batch]
		actions = self.mem_action[batch]
		rewards = self.mem_reward[batch]
		states = self.mem_state[batch]
		states_new = self.mem_new_state[batch]
		return states, actions, rewards, states_new, term

	'''
	Constructor
	Params: <> max_size: max number of memories we want to store
			<> input_shape: the shape of the observation of the environment, for lunar lander vector of 8 elements
			<> n_actions: number of actions, 4 for lunar lander
	'''
	def __init__(self, max_size, input_shape, n_actions, discrete=False):
		self.discrete = discrete
		if self.discrete:
			data_type = np.int8
		else:
			data_type = np.float32
		self.mem_sz = max_size
		self.mem_counter = 0
		self.mem_state = np.zeros((self.mem_sz, input_shape))
		self.mem_new_state = np.zeros((self.mem_sz, input_shape))
		self.mem_action = np.zeros((self.mem_sz, n_actions), dtype=data_type)
		self.mem_reward = np.zeros(self.mem_sz)
		# mem_term is because we are sampling transitions from many episodes, when the episode is over we don't want to sample the next state because the expected future reward is 0 in this state. 
		self.mem_term = np.zeros(self.mem_sz, dtype=np.float32)


####################################################################################################################################################################################################################################################################################################################################################
if __name__=='__main__':
	print("\nInitializing LunarLander-v2 environment....")
	env=gym.make('LunarLander-v2')
	AGENT_SCORE=[]
	EPSILON_SCORE=[]
	print("\nInitializing Agent (DDQN) ....") 
	AGENT = AGENT_DDQN(ALPHA,GAMMA,POSSIBLE_ACTIONS,EPSILON,BATCH_SZ,INPUT_D)

	if MONITOR_AGENT:
		env=wrappers.Monitor(env, 'vids', video_callable=lambda episode_id: True, force=True)

	agg_ep_rewards = {'ep': [], 'avg': [], 'avg_epsilon': []}
	for i in range(NO_OF_EPISODES):
		done=False
		score=0
		observation=env.reset()
		while not done:
			action=AGENT.select_action(observation)
			observation_,reward,done,info=env.step(action)
			score = score + reward
			AGENT.manage_memory(observation,action,reward,observation_,done)
			observation=observation_
			AGENT.learning()
		AGENT_SCORE.append(score)
		EPSILON_SCORE.append(AGENT.EPSILON)
		score_average_val=np.mean(AGENT_SCORE[max(0,i-NO_OF_EPISODES_TO_AVERAGE_SCORE_FOR):(i+1)])   # Average of last "100" games.
		epsilon_average_val=np.mean(EPSILON_SCORE[max(0,i-NO_OF_EPISODES_TO_AVERAGE_SCORE_FOR):(i+1)])   # Average of last "100" games.
		print("| EPISODE=",i, "| SCORE= %.2f"%score, "| AVERAGE_SCORE=",score_average_val, "| Averaged over:", NO_OF_EPISODES_TO_AVERAGE_SCORE_FOR, "episodes |")
		# For plotting purposes later (visual)
		agg_ep_rewards['ep'].append(i)
		agg_ep_rewards['avg'].append(score_average_val)
		agg_ep_rewards['avg_epsilon'].append(epsilon_average_val)
		if MONITOR_AGENT:
			time.sleep(MONITOR_AGENT_WINDOW_DELAY_SECONDS)			
	print("Closing environment...")

	# For plotting purposes later (visual)
	plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['avg'], label='Avg Rewards (per 100 epsidoes)')
	plt.title("DDQN Agent's Rewards Tracking", loc='center')
	plt.xlabel('Episode #')
	plt.ylabel('Avg Rewards')
	plt.legend(loc=4)
	plt.grid(True)
	plt.show()

	plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['avg_epsilon'], label='Avg Epsilon (per 100 episodes)')
	plt.title(PLOT_TITLE, loc='center')
	plt.xlabel('Episode #')
	plt.ylabel('Avg Epsilon Value')
	plt.legend(loc=4)
	plt.grid(True)
	plt.show()
	
	env.close()
'''
This will create a literal compilation video of a bunch of .mp4s representing each episode rendering above...
Awesome.....
'''
if MONITOR_AGENT and DO_VIDEO_COMPILATION_OF_ALL_EPISODES_AT_END:
	'''
	Attempting to combine multiple .mp4's into one .mp4 for demo purposes.
	Reference: https://stackoverflow.com/questions/56920546/combine-mp4-files-by-order-based-on-number-from-filenames-in-python
	'''
	from moviepy.editor import *
	import os
	from natsort import natsorted
	import shutil

	L =[]

	for root, dirs, files in os.walk(PATH_TO_VIDEOS):

	    #files.sort()
	    files = natsorted(files)
	    for file in files:
	        if os.path.splitext(file)[1] == '.mp4':
	            filePath = os.path.join(root, file)
	            video = VideoFileClip(filePath)
	            L.append(video)

	final_clip = concatenate_videoclips(L)
	final_clip.to_videofile("output.mp4", fps=24, remove_temp=False)

	# if not KEEP_INDIVIDUAL_EPISODE_VIDEOS:
	# 	time.sleep(3) # adding a delay to avoid race condition to delete the individual episodes folder. 
	# 	print("Deleting individual episode videos from [vids] folder...")
	# 	shutil.rmtree(PATH_TO_VIDEOS)


print("DDQN Agent - Lunar Lander v2 is complete.")










