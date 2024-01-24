from multiprocessing import Pool
import UpdataFunTopic
from UpdataFunTopic import *
import import_ipynb
import os
import openai
import requests
import json
import re
import pickle
from pathlib import Path
import numpy as np
import random
import time
import copy
import time


if __name__ ==  '__main__': 
    
    # load initial setting
    generateUserDict = pickle.load(open("generateUserDict", "rb"))
    smallFollowDict = pickle.load(open("followingDict", "rb"))
    embedtotweetDict = pickle.load(open('embedtotweetDict', "rb"))
    initialFollowDict = pickle.load(open("followingDict", "rb"))
    initialTweetDict = pickle.load(open("initialTweetDict", "rb"))
    
    # choose agent
    agent = random.choice(generateUserDict.keys())
    slice_space = len(smallFollowDict[agent])
    follow_user = len(smallFollowDict[agent])
    observe_env_agents = smallFollowDict[agent]
    if follow_user == 1:
        state_space_size = slice_space
    else:   
        state_space_size = follow_user**(slice_space)
        
    # action: 0,1,2,3,4,5
    action_space_size = 5
    
    # size of q_table
    personal_space_size = len(smallCaseUser)
    q_table=np.zeros(((personal_space_size+1),state_space_size,action_space_size))

    # Number of episodes and step
    num_episodes = 500
    max_steps_per_episode = 5
    
    learning_rate = 0.01
    discount_rate = 0.99
    linkRate = 0.8
    breakRate = 0.5
    exploration_rate = 0.01
    num_same = 2
    epsilon = 0.005
    reward_all_episodes = []

    
    num_processors = 4

    
    for episodes in range(0,num_episodes):
        print("episodes:",episodes)
        reward_list = []
        rewards_current_episode = 0
        reward = 0
        action_list = []
        agentFollower = []
        generateUserDict = copy.deepcopy(smallCaseUser)
        followDict = copy.deepcopy(smallFollowDict)
        userList = list(generateUserDict.values())
        tweetDict = copy.deepcopy(initialTweetDict)
        polarityDict = makeInitialPolarity(tweetDict)
        tweetEmbedDict = makeInitialEmbed(polarityDict)
        modified_exploration_rate = min((exploration_rate + epsilon*episodes),0.95)
        followerState = 0
        
        
        time.sleep(5)
        for step in range(max_steps_per_episode): 
            stateIndex = locateState(observe_env_agents,slice_space,tweetEmbedDict)
            exploration_rate_threshold = random.uniform(0,1)
            if exploration_rate_threshold > modified_exploration_rate:
                action = random.choice(action_space)
            else:
                action = np.argmax(q_table[followerState,stateIndex,:])
            action_list.append(action)
            
                
            inputTurpleList = [] 
            for i in generateUserDict.keys():
                sentence = interactInput(i,generateUserDict,followDict,tweetDict,step)
                inputTurple = (sentence,1)
                inputTurpleList.append(inputTurple)               

            p2 = Pool(processes = num_processors)
            results = p2.starmap(updateFixInteractTweet,inputTurpleList)
            p2.close()
            p2.join()
            
            t = 0
            for i in generateUserDict.keys():
                
                if results[t] == "Retry":
                    print("Retry")
                    a,b = inputTurpleList[t]
                    gptResult = updateRetryTweet(a,b)
                    results[t] = gptResult
                tweetDict[i].append(results[t])
                t += 1
                       
            p3 = Pool(processes = num_processors)
            polResults = p3.map(returnPolarity,results)
            p3.close()
            p3.join()
            
            t = 0
            for i in generateUserDict.keys():
                currentPolarity = polResults[t]
                polarityDict[i].append(currentPolarity)
                currentPosition = embed(currentPolarity)
                tweetEmbedDict[i].append(currentPosition)
                t += 1


            newStateIndex = locateState(observe_env_agents,slice_space,tweetEmbedDict)           
            
            if step >= (num_same-1):
                newFollowDict = update_link(followDict,tweetEmbedDict,linkRate, breakRate, num_same, step)
                reward = newRewardCalculate(followDict, newFollowDict,agent)
                reward_list.append(reward)
            else:
                newFollowDict = followDict
                reward = len(newFollowDict[agent])
                reward_list.append(reward)

            newFollowerState = (int(rewards_current_episode)+reward)

            
        #Update Q-table for Q(s,a)
            q_table[followerState, stateIndex, action] = q_table[followerState, stateIndex, action] * (1 - learning_rate) + \
                learning_rate * (reward + discount_rate * np.max(q_table[newFollowerState,newStateIndex, :]))
            stateIndex = newStateIndex
            rewards_current_episode += reward
            followDict = newFollowDict
            followerState = newFollowerState

        reward_all_episodes.append(rewards_current_episode)

        print("reward_all_episodes:",reward_all_episodes)
