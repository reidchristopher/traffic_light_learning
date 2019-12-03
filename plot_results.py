#!/usr/bin/env python

import matplotlib.pyplot as plt
import pickle

with open("trad_test_results.pickle", "rb") as f:
    trad_results = pickle.load(f)
    
plt.plot(range(len(trad_results)), trad_results)

with open("trad_net_test_results.pickle", "rb") as f:
    trad_net_results = pickle.load(f)

plt.plot(range(len(trad_net_results)), trad_net_results)
    
with open("evolutionary_results.pickle", "rb") as f:
    evo_results = pickle.load(f)

plt.plot([(i + 1) * 10 for i in range(len(evo_results))], evo_results)    
    
with open("actor_critic_results_v2.pickle", "rb") as f:
    ac_results = pickle.load(f)
    
plt.plot(range(len(ac_results)), ac_results)

plt.xlabel("Episode #")
plt.ylabel("Episode reward (-Cumulative waiting time [s])")

plt.legend(['Sequential Policy', 'Network trained from sequential policy', 'Evolutionary Policy', 'Advantage Actor Critic Policy'], loc='lower center', bbox_to_anchor=(0.75, 0), fontsize="small")

plt.show()
