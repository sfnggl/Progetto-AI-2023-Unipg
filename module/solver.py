import cv2
import time
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import display
from search import *
from collections import *

def breadth_search_graph(problem):
  node = Node(problem.initial)

  if problem.goal_test(node.state):
    return node

  frontier = deque([node])
  explored = set()

  while frontier:
    node = frontier.popleft()
    print(node)
    explored.add(tuple(node.state))
    for child in node.expand(problem):
      print(f"child state: {child.state}")
      print(f"Testina: {problem.get_t(child.state)}")
      if tuple(child.state) not in explored and child not in frontier:
        if problem.goal_test(child.state):
          return child
        frontier.append(child)
  return None


def depth_search_graph(problem):
  node = Node(problem.initial)

  if problem.goal_test(node.state):
    return node

  frontier = deque([node])
  explored = set()

  while frontier:
    node = frontier.pop()
    print(node)
    explored.add(tuple(node.state))
    for child in node.expand(problem):
      print(f"child state: {child.state}")
      print(f"Testina: {problem.get_t(child.state)}")
      if tuple(child.state) not in explored and child not in frontier:
        if problem.goal_test(child.state):
          return child
        frontier.append(child)
  return None

def depth_limited_search(problem, l):
  frontier= ([Node(problem.initial)])
  solution = 'failure'
  while frontier:
    node = frontier.pop()
    if problem.goal_test(node.state):
      solution = node
      return solution
    elif node.depth > l:
      solution = 'cutoff'
    elif not is_cycle(node):
      for child in node.expand(problem):
        print(f"child state: {child.state}")
        print(f"Testina: {problem.get_t(child.state)}")
        frontier.append(child)
  return solution

def is_cycle(node, k=5):
  """costa troppo controllare un ciclo di M lunghezza con M
  numero grande di passi. Imponiamo un limite k a 50"""
  def find_cycle(ancestor, k):
    """implementazione ricorsiva, altro pericolo"""
    return (ancestor is not None and k > 0
            and (ancestor.state == node.state or
            find_cycle(ancestor.parent, k-1)))
  return find_cycle(node.parent, k)

def interative_depending_search(problem):
    for limit in range(1, sys.maxsize):
        result = depth_limited_search(problem, limit)
        if result != 'cutoff':
            return result

def best_first_search_graph_h(problem, f, no_memoize = False, t = -1):
  init = Node(problem.initial)

  if problem.goal_test(init.state):
    return init

  f1 = f
  f2 = f

  f = memoize(f1, 'f')
  frontier = PriorityQueue('min', f)
  frontier.append(init)

  explored = set()

  #t = -1

  while frontier:
    node = frontier.pop()
    if ( t == 0 ) and problem.goal_cell(node.state):
        #print("Trovata la soluzione!")
        return node
    explored.add(tuple(node.state))
    for child in node.expand(problem):
      if tuple(child.state) not in explored and child not in frontier:
        if ( t == -1 ) and problem.goal_grid(child.state):
          t = 0
          f = memoize(f2, 'f')
          frontier = PriorityQueue('min', f)
          #print("\n\n\n\n\n\n\nALL COLORED! Coming back home\n\n\n\n\n\n")
          frontier.append(child)
          break
        frontier.append(child)
      elif child in frontier:
        incumbent = frontier.get_item(child)
        if f(incumbent) > f(child):
          del frontier[incumbent]
          frontier.append(child)
    #print(f"heuristic: {f(node)}")
  return None

def unif_cost_search(problem):
  return best_first_search_graph_h(problem, lambda node: node.path_cost)

def show_solution(node):
  if node is None:
      print("no solution")
  else:
      print("solution: ", node.solution())

def a_star_search(problem, h = None):
  h = memoize(h or problem.h, 'h')
  return best_first_search_graph_h(problem, lambda n : h(n), no_memoize = False)

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

    out_state = new_state + [cursor]
    return out_state

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

def h(problem, node):
  return min(sum( s != g for (s,g) in zip(node.state, problem.one) ), sum( s!=g for (s,g) in zip(node.state, problem.two) ), sum( s!=g for (s,g) in zip(node.state, problem.three) ))

def hGrid(problem, node):
  return min([len(node.state) - node.state.count(k) - 1 for k in range(1,4)])

def hAll(problem, node):
  return min(
    sum([node.path_cost + 1 + 1  for a in range(0,len(node.state)-1) if node.state[a] != problem.one[a]]),
    sum([node.path_cost + 1 + 2  for a in range(0,len(node.state)-1) if node.state[a] != problem.two[a]]),
    sum([node.path_cost + 1 + 3  for a in range(0,len(node.state)-1) if node.state[a] != problem.three[a]])
)
  
W, B, Y, G, W_T, B_T, Y_T, G_T = np.zeros((3,3,3), np.uint8), np.zeros((3,3,3), np.uint8), np.zeros((3,3,3), np.uint8), np.zeros((3,3,3), np.uint8), np.zeros((3,3,3), np.uint8), np.zeros((3,3,3), np.uint8), np.zeros((3,3,3), np.uint8), np.zeros((3,3,3), np.uint8)
W[::], B[::], Y[::], G[::], W_T[::], B_T[::], Y_T[::], G_T[::] = (255,255,255), (0,0,255), (255,255,0), (0,255,0), (255,255,255), (0,0,255), (255,255,0), (0,255,0)
W_T[1,1] = B_T[1,1] = Y_T[1,1] = G_T[1,1] = (255, 0, 0)

color = [W, B, Y, G, W_T , B_T, Y_T, G_T]

def visual(path_solution, move_solution, w, h):

  #path_solution is containo node, we extract the node.state
  state_solution = []

  for i in path_solution:
    state_solution.append(i.state.copy())  #copy only value, not the reference

  for state in range(len(state_solution)):

    #find index T
    T_index = state_solution[state][-1]

    #make index t colored us testina
    state_solution[state][T_index] = state_solution[state][T_index] + 4

    #remove index t in state
    state_solution[state] = state_solution[state][0:-1]

  #widget PART STARTING

  a = widgets.IntSlider(min=0,max=len(state_solution)-1,step=1,value=0,description='Step: ', continuous_update=True)
  ui = widgets.HBox([a])

  def f(a):

    for i in range(len(state_solution[a])):
      plt.subplot(h,w, i+1)
      plt.xticks([])
      plt.yticks([])
      plt.imshow(color[state_solution[a][i]])

    if(a==0):
      plt.suptitle(f"Initial stete, Next Move: {move_solution[a]}")
    elif(a <= len(move_solution)-1):
      plt.suptitle(f"Applied move: {move_solution[a-1]} | Next Move: {move_solution[a]}")
    else:
      plt.suptitle(f"Final stete, Applied move: {move_solution[a-1]}")

    plt.show()

  out = widgets.interactive_output(f, {'a': a})
  display(ui, out)
