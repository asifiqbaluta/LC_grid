## -*- coding: utf-8 -*-
#"""
#Created on Mon Sep 28 16:19:42 2020
#
#@author: iqbal
#"""
#
#
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#
#torch.manual_seed(1)
#
#
#lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
#inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5
#
## initialize the hidden state.
#hidden = (torch.randn(1, 1, 3),
#          torch.randn(1, 1, 3))
#for i in inputs:
#    # Step through the sequence one element at a time.
#    # after each step, hidden contains the hidden state.
#    out, hidden = lstm(i.view(1, 1, -1), hidden)
#    
## alternatively, we can do the entire sequence all at once.
## the first value returned by LSTM is all of the hidden states throughout
## the sequence. the second is just the most recent hidden state
## (compare the last slice of "out" with "hidden" below, they are the same)
## The reason for this is that:
## "out" will give you access to all hidden states in the sequence
## "hidden" will allow you to continue the sequence and backpropagate,
## by passing it as an argument  to the lstm at a later time
## Add the extra 2nd dimension
#inputs = torch.cat(inputs).view(len(inputs), 1, -1)
#hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
#out, hidden = lstm(inputs, hidden)
#print(out)
#print(hidden)    

import numpy as np 


def read_and_create_grid(input_file):
    if type(input_file)!=type(''):
        print("invalid input")
        return

    file = open(input_file, 'r')
    Lines = file.readlines()
    init_line = Lines[0].split()

    out = np.zeros( (int(init_line[0]),int(init_line[1]) ) ,dtype=np.int)
    
    for i in Lines[1:]:
        temp_line = i.split()
        print(temp_line)
        if temp_line[0] == 'W':
            
            for j in np.arange(int(temp_line[3]),int(temp_line[4])+1):                
                out[int(temp_line[1]):int(temp_line[2])+1,j] = 1

        elif temp_line[0] == 'O':           
            for j in np.arange(int(temp_line[3]),int(temp_line[4])+1):
                out[int(temp_line[1]):int(temp_line[2])+1,j] = int(temp_line[5])

        elif temp_line[0] == 'A':
            out[int(temp_line[1])][int(temp_line[2])] = -1
            
    return out



a = read_and_create_grid("grid.txt")

print(a.shape)
#
print(a)

    
                
    
    