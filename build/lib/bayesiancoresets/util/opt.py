import numpy as np
import sys
from scipy.optimize import line_search


def nn_opt(x0, grd, nn_idcs=None, opt_itrs=1000, step_sched=lambda i : 1./(i+1), b1=0.9, b2=0.999, eps=1e-8, verbose=False):
  x = x0.copy()
  m1 = np.zeros(x.shape[0])
  m2 = np.zeros(x.shape[0])
  for i in range(opt_itrs):
    g = grd(x)
    if verbose:
      active_idcs = np.intersect1d(nn_idcs, np.where(x==0)[0])
      inactive_idcs = np.setdiff1d(np.arange(x.shape[0]), active_idcs)
      sys.stdout.write('itr ' + str(i+1) +'/'+str(opt_itrs)+': ||inactive constraint grads|| = ' + str(np.sqrt((g[inactive_idcs]**2).sum())) + '                \r')
      sys.stdout.flush()
    m1 = b1*m1 + (1.-b1)*g
    m2 = b2*m2 + (1.-b2)*g**2
    upd = step_sched(i)*m1/(1.-b1**(i+1))/(eps + np.sqrt(m2/(1.-b2**(i+1))))
    x -= upd
    #project onto x>=0
    if nn_idcs is None:
        x = np.maximum(x, 0.)
    else:
        x[nn_idcs] = np.maximum(x[nn_idcs], 0.)

  if verbose:
    sys.stdout.write('\n')
    sys.stdout.flush()
  return x


def an_opt(x0, grd, search_direction, opt_itrs=1000, step_sched=lambda i : 1./(i+1)):
  # Define starting point
  x = x0.copy()
  for i in range(2):
    # g = search_direction(x)
    # # print("Norm of gradient is: {}".format(np.sqrt(np.sum(grd(x)**2))))
    # # print("a of gradient is: {}".format(np.dot(grd(x),g)))
    #
    # # upd = 0.01/step_sched(i)*g
    # upd = 0.1 * g
    # # if i<5:
    # #   upd = 0.1*g
    # # else:
    # #   upd = g
    #
    # x += upd
    # # project onto x>=0
    # x = np.maximum(x, 0.)

    # Wolfe Curvature Condition Line Search


    g = search_direction(x)
    # print("Norm of gradient is: {}".format(np.sqrt(np.sum(grd(x)**2))))
    print("a of gradient is: {}".format(np.dot(grd(x),g)))


    a=1.0
    k=0.9
    fail = 0
    for j in range(10):
      test = np.dot(grd(x+a*g),g)/np.dot(grd(x),g)
      # print("test = {}".format(test))
      if test >= k or test <= 0:
        a = a/1.5
      else:
        upd = a*g
        # print("alpha = {}".format(a))
        # print("a = {}".format(np.dot(grd(x + a * g), g)))
        fail = 1
        break

    if fail == 0:
      break
    x += upd
    # project onto x>=0
    x = np.maximum(x, 0.)



  return x
