import scipy as sp
import scipy.spatial.distance as ssd
import numpy as np
import pandas as pd

# returnHome is lame, but sometimes the salesman wants to end at the city from whence he came
returnHome = True

def parse_latlon(x):
    d, m, s = map(float, x.split(':'))
    ms = m/60. + s/3600.
    if d<0:
        return d - ms
    return d + ms

def path_length(citylist, distmatrix):
  n = len(citylist)
  if returnHome:
    lens = [0 for i in range(n)]
  else:
    lens = [0 for i in range(n-1)]
  for i in range(n-1):
    lens[i] = distmatrix[citylist[i]][citylist[i+1]]
  # add distance of last city to first city
  if returnHome:
    lens[n-1] = distmatrix[citylist[n-1]][citylist[0]]
  return lens

def energy(citylist, distmatrix):
  return sum(path_length(citylist, distmatrix))

# more recent numpy has random.choice
def weighted_values(values, probabilities, size=1):
  bins = np.cumsum(probabilities)
  return values[np.digitize(np.random.random_sample(size), bins)]

cities = pd.read_csv('brasil_capitals.txt', names=['city','lat','lon'])[['lat','lon']].applymap(parse_latlon)

dist = ssd.euclidean
locations = cities.values
# might as well make distances integer values
precision = pow(10, 4)
distmat = [np.array([int(round(dist(a,b)*precision)) for a in locations]) for b in locations]
n = len(cities)

# return similar route by transposing two cities
def neighbor(x, distmat, k=1):
  # get distance of each route
  paths = np.array(path_length(x, distmat))
  # remove last path
  if returnHome:
    paths.resize(len(paths)-1)
  y = x.copy()
  if k > 1:
    # discrete uniform choice
    choice = np.random.randint(0, len(paths)-k+1)
    z = y[choice+k]
    for i in range(k, 0, -1):
      y[choice+i] = y[choice+i-1]
    y[choice] = z
  else:
    # probability of being chosen
    probs = paths/float(paths.sum())
    # chose a city, weighted by path distance
    choice = weighted_values(range(len(paths)), probs)
    z = y[choice+1]
    y[choice+1] = y[choice]
    y[choice] = z
  return y

# return probability between 0 and 1
def gomove(a, b, t):
  if a > b:
    return 1
  return min(1, np.exp((a - b)/float(t)))

# simulated annealing
def sa_go(distmat, cooling, tau, init=None, nk=1):
  acceptableDist = 1*pow(10, 6)
  if init is None:
    s = np.random.permutation(range(len(distmat)))
  else:
    s = init.copy()
  e = energy(s, distmat)
  sbest = s.copy()
  ebest = e
  for j in range(len(cooling)):
    coolsteps = cooling[j]
    samples = np.random.random_sample(coolsteps)
    for k in range(coolsteps):
      snew = neighbor(s, distmat, nk)
      enew = energy(snew, distmat)
      if gomove(e, enew, tau[j]) > samples[k]:
        s = snew.copy()
        e = enew
      if enew < ebest:
        sbest = snew.copy()
        ebest = enew
      if e < acceptableDist:
        break
  return [sbest, ebest]

iterations = 20
cooling = [75]*(iterations/4) + [125]*(iterations/4) + [175]*(iterations/4) + [250]*(iterations/4)
tau = [50000 * 0.9**i for i in range(iterations)]

if returnHome:
  s = np.array(range(n))
else:
  s = None
nochange = i = past = 0
answer = None
global_min = pow(10, 8)
# "cheat"-ing when you returnHome doesn't help much
if returnHome:
  cheater = range(5)
else:
  cheater = range(n)
while nochange < 8:
  s, d = sa_go(distmat, cooling, tau, s, nk=i%4+1)
  if (i % 10) == 0:
    if d == past:
      print i
      # stuck? reverse path
      s = s[range(len(s)-1, -1, -1)]
      nochange += 1
    else:
      print s
      print d
      nochange = 0
    past = d
  # try a new approach, iterate over every possible starting point
  if nochange == 5 and len(cheater):
    # set a new start point
    split = cheater[np.random.randint(0, len(cheater))]
    cheater.remove(split)
    print cheater
    s = np.concatenate([s[range(split,n)], s[range(0,split)]])
    nochange = 0
  if d < global_min:
    global_min = d
    answer = s.copy()
  i += 1

print answer
print global_min

# best so far
# 1102786 ~= 110 unscaled
#[13 16 17 15 20 25 24 23 22 21 18 19 14 11  9  8  7  5  6  3  2  1  0  4 10 12]
# with returnHome set...
# 1260174
# huge = np.array([0,1,2,3,6,5,7,8,9,11,14,13,16,17,18,19,21,22,23,24,25,20,15,12,10,4])
