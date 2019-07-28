import pickle
from matplotlib import pyplot, ticker 

with open('./Generated Data/valPlotOpt1.pkl', 'rb') as f:
    valPlot1 = pickle.load(f)

with open('./Generated Data/valPlotOpt2.pkl', 'rb') as f:
    valPlot2 = pickle.load(f)

pyplot.rcParams["font.family"] = 'Times New Roman'
fig, ax = pyplot.subplots(figsize=(6, 4), dpi=100)
pyplot.plot(valPlot1, linewidth=1)

pyplot.xlabel("Epoch", fontsize=18)
pyplot.ylabel("Cost", fontsize=18)
pyplot.yticks(fontsize=16)
pyplot.xticks([0] + list(range(4, 34, 5)), [1] + list(range(50, 350, 50)), fontsize=16)
ax2 = ax.twinx()
ax2.plot(valPlot2, linewidth=1, color='green')
pyplot.yticks(fontsize=16)
pyplot.title("Validation cost for custom optimiser", fontsize=22)
# pyplot.yticks([0.5, 0.75, 1], fontsize=16)
# pyplot.xticks([0] + list(range(9, 69, 10)), [1] + list(range(10, 70, 10)),fontsize=16)
pyplot.grid()
pyplot.subplots_adjust(top=0.905, bottom=0.145, right=0.97, left=0.145, hspace=0.2)
pyplot.show()
exit