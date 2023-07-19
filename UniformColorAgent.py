import aima
from aima.search import *
from collections import *

#Definiamo il problema

#(self, initial, depth, height)
# Esempio
# stato = ( [1,0,3,3] , 2 )
# (stato, 2, 2)
# Nuova idea
# stato = ([1,0,3,3,2])

class UniformColor(Problem): 

  def __init__(self, initial, width, height):

    self.w = width
    self.height = height
    self.Tindex = len(initial)-1
    self.init_pos = initial[self.Tindex]

    self.one=[1 for a in range(width*height)] 
    self.one[self.init_pos]=0
    self.two=[2 for a in range(width*height)] 
    self.two[self.init_pos]=0
    self.three=[3 for a in range(width*height)] 
    self.three[self.init_pos]=0

    super().__init__(initial, goal=(self.one,self.two,self.three))

  def actions(self, state):
    cursor = state[self.Tindex]

    possible_actions = ['right', 'up', 'left', 'down', 'col-B', 'col-G', 'col-Y']

    
    if cursor % self.w == self.w - 1:
      possible_actions.remove('right')
    if cursor < self.w:
      possible_actions.remove('up')
    if cursor % self.w == 0:
      possible_actions.remove('left')
    if cursor >= self.w*(self.height - 1):
      possible_actions.remove('down')

    if state[0:len(state)-1] in self.goal:
      possible_actions.remove('col-B')
      possible_actions.remove('col-G')
      possible_actions.remove('col-Y')

    else:    
      if state[cursor] == 1:
        possible_actions.remove('col-B')
      if state[cursor] == 2:
        possible_actions.remove('col-Y')
      if state[cursor] == 3:
        possible_actions.remove('col-G')
      if state[cursor] == 0:
        possible_actions.remove('col-B')
        possible_actions.remove('col-G')
        possible_actions.remove('col-Y')

    return possible_actions

  def get_t(self, state):
    return state[self.Tindex]


  def result(self, state, action):

    #index of cursor
    cursor = state[self.Tindex]

    #transform tuple in list for coloring or switching position
    new_state = state[0:self.Tindex]
    
    if(action == 'col-B' or action == 'col-G' or action == 'col-Y'):

      coloring = {'col-B': 1, 'col-Y': 2, 'col-G': 3}

      new_state[cursor] = coloring[action]
    
    else:

      #dictionary to move and index
      moving = { 'right' : 1, 'up' : -self.w, 'left' : -1, 'down' : self.w }

      #index of T moved
      cursor += moving[action]

      #swap plate
      #new_state[cursor], new_state[neighbor] = new_state[neighbor], new_state[cursor]
      
    out_state = new_state + [cursor]
    return out_state

  #def goal_test(self, state):
  #  return state[0:self.Tindex] in self.goal and state[self.Tindex] == self.init_pos

  def goal_grid(self, state):
    return state[0:len(state)-1] in self.goal

  def goal_cell(self, state):
    return state[self.Tindex] == self.init_pos

  def path_cost(self, c, state1, action, state2):
    coloring = {'col-B': 1, 'col-Y': 2, 'col-G': 3}
    #costo unitario, sola distanza
    if action in coloring:
      paint_cost = coloring[action]
      return c + paint_cost
    else:
      return c + 1
