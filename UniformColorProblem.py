import numpy as np
import time
import random
from os import sys
from UniformColorAgent import *
from searches import *
from collections import *

def UniformColorProblem(array, width, height):

    cleanup = array + [array.index(0)]

    p1 = UniformColor(cleanup, width, height)

    # // -- code --
    # 1. Performe check for euristic
    # 2. Run the solver
    # 3. publish result
    start_time = time.time()
    solPuzzle = a_star_search(p1, lambda node : hAll(p1, node))
    end_time = time.time()
    #print_sol(solPuzzle.path(), solPuzzle.solution(), width, height)

    print(solPuzzle)
    print("\n",show_solution(solPuzzle))
    print('cost = ', solPuzzle.path_cost)
    print('time = ', end_time-start_time)
    print('steps = ', len(solPuzzle.solution()))

    return p1

if __name__ == '__main__' :
    #UniformColorProblem([1,1,1,2,2,2,1,2,3,2,3,2,1,2,3,1,3,1,2,0,3,3,1,3,3,2,2,1,2,3,1,2,3,1,1,3,2,3,1,2,3,3,1,1,3,3,3,3], 6, 8)
    #UniformColorProblem([3,3,2,1,0,2,3,1], 4, 2)
    UniformColorProblem([3,3,3,3,3,2,2,3,2,1,0,2,2,1,1,2,3],17,1)
    #UniformColorProblem([2,3,3,1,0,3], 3, 2)

#NOTA SULL EURISTICA:
# EURIS NON PESATA DA PRIORITA ALLA SOLA QUANTITA DI CASELLE MANCANTI
# SOLUZIONE CON CONTESTO MA NON SEMPRE OTTIMALE
# EURIS PESATA DA PRIORITA AL SOLO COSTO DELLE AZIONI
# SOLUZIONE STUPIDA MA SPESSO OTTIMALE
# EURIS NON PESATA E PESATA COINCIDONO SE LA CASELLA PIU PRESENTE,
#  MOLTIPLICATA PER IL SUO COSTO + 1, DA UN VALORE INFERIORE AL COLORE PIU
#  PIU ECONOMICO
# IMPLEMENTARE CHECK DI QUESTO PER SCEGLIERE L'EURIS PIU SIGNIFICATIVA 
# PER OGNI PROBLEMA
# Dim.
# Sia M una griglia di dim. k = b*h casualmente riempita di valori B, G e Y, i.i.d.
# Per k abbastanza grandi: #{B} = k/3, #{Y} = k/3, #{G} = k/3
# e Sum(i)PATH(B) = Sum(i)PATH(Y) = Sum(i)PATH(G) = k
# dunque, volendo colorare la griglia del colore X, dovremmo percorrere
# circa k celle e colorarne 2/3k, dunque si puo ben vedere che:
# (2/3k + k)*1 < (2/3k + k)*2 < (2/3k + k)*3. Dunque sarà spesso molto più
# economico colorare tutto di blu.
# Ciò non è vero unicamente per il caso in cui, presuppendo:
# i) PATH(X / B) <= PATH(B) 
# ii) p*3 =~ q*2 < t*1
# con p, n. di celle non verdi, q non gialle e t non blu
# Riarrangiando i termini, la strategia di colorare tutto di blu non è ottimale se
# p <= t/3 o q <= t/2,
# ossia quando le caselle blu sono 2 volte più presenti di quelle gialle
# o 3 volte più presenti di quelle verdi
# Dunque questo sovrastante sarà il nostro check
#