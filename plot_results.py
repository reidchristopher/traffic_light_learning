#!/usr/bin/env python

import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import stats

sequential = False
sequential_average = True
sequential_net = False
sequential_net_average = False
evo = True
evo_transfer = True
ac = False
ac_transfer = False

x_limits = [0, 100]
y_limits = [-100000, 0]

legend_list = []

fig, ax = plt.subplots(1)

def plot_avg_with_std(results_list):

    results = np.array(results_list)
    
    results_average = results.mean(axis=0)
    
    results_stderr = stats.sem(results) # results.std(axis=0)
    
    lower_bound = results_average + results_stderr
    upper_bound = results_average - results_stderr
    
    ax.plot(range(len(results_average)), results_average)
    ax.fill_between(range(len(results_average)), lower_bound, upper_bound, alpha=0.5)

    

with open("trad_test_results.pickle", "rb") as f:
    trad_results = pickle.load(f)

if sequential:

    ax.plot(range(len(trad_results)), trad_results, 'b')

    legend_list.append('Sequential Policy')



with open("trad_net_test_results.pickle", "rb") as f:
    trad_net_results = pickle.load(f)

if sequential_net:

    ax.plot(range(len(trad_net_results)), trad_net_results, 'r')
    
    legend_list.append("Network trained w/ Sequential Policy")



if evo:

    evo_results_list = []
    
    for i in range(10):
        with open("evolutionary_results_%d.pickle" % i, "rb") as f:
            evo_results = pickle.load(f)

            #ax.plot(range(len(evo_results)), evo_results)
            
            evo_results_list.append(evo_results[:100])
            
    plot_avg_with_std(evo_results_list) 

    legend_list.append("Evolution from random weights")

if ac:

    ac_results_list = []

    for i in range(11):
        with open("actor_critic_results_%d.pickle" % i, "rb") as f:
            
            ac_results = pickle.load(f)
            #ax.plot(range(len(ac_results)), ac_results)
            ac_results_list.append(ac_results[:100])
            
    plot_avg_with_std(ac_results_list)
    
    legend_list.append("Advantage Actor Critic from random weights")



if evo_transfer:

    evo_transfer_results_list = []
    
    for i in range(10):

        with open("evolutionary_transfer_results_%d.pickle" % i, "rb") as f:
            evo_transfer_results = pickle.load(f)

            #ax.plot(range(len(evo_transfer_results)), evo_transfer_results)
            
            evo_transfer_results_list.append(evo_transfer_results[:100])
            
    plot_avg_with_std(evo_transfer_results_list)
    
    legend_list.append("Evolution w/ Transferred Networks")   



if ac_transfer:

    ac_transfer_results_list = []

    for i in range(11):
        with open("actor_critic_transfer_results_%d.pickle" % i, "rb") as f:
            ac_results = pickle.load(f)
            #ax.plot(range(len(ac_results)), ac_results)
            ac_transfer_results_list.append(ac_results[:100])
        
    plot_avg_with_std(ac_transfer_results_list)
    
    legend_list.append("Advantage Actor Critic w/ Transferred Network")

if sequential_average:
    seq_average = np.average(trad_results)
    ax.plot(x_limits, [seq_average, seq_average], '--')

    legend_list.append("Sequential Policy Average")

    
if sequential_net_average:
    seq_net_average = np.average(trad_net_results)

    ax.plot(x_limits, [seq_net_average, seq_net_average], '--')

    legend_list.append("Network trained w/ Sequential Policy Average")


ax.set_xlabel("Episode #")
ax.set_ylabel("Episode reward (-Cumulative waiting time [s])")

ax.legend(legend_list, loc='lower center', bbox_to_anchor=(0.75, 0), fontsize="small")

ax.set_xlim(x_limits)
ax.set_ylim(y_limits)

ax.grid()

plt.show()
