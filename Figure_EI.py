import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          # 'figure.figsize': (15, 5),
# "text.usetex": True,
         'axes.labelsize': 20,
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
matplotlib.use('TkAgg')
import seaborn as sns

data = pd.read_excel('./VARMA_dri50_epi50_sim50_num6_p2q2.xlsx')

# print the overal ATE estimaion results
ATE_emprical_true = 2.24
ATE_pd = data[['Method', 'ATE_estimator']].copy()
ATE_pd['MSE'] = (ATE_pd['ATE_estimator'] - ATE_emprical_true) ** 2
ATEs_MSE_ave = ATE_pd.groupby('Method')['MSE'].mean().sort_values(ascending=True)
print(ATEs_MSE_ave)


# plot the empirical distribution of the efficiency indicators

data_AD = data[data['Method'] == 'ATE_AD']
data_AT = data[data['Method'] == 'ATE_AT']
data1 = data_AD['sum_theta']
data2 = data_AT['sum_theta_minus']

plt.figure(figsize=(10, 6))

# Plot KDEs for both datasets
sns.kdeplot(x=data1, fill=True, color="blue", label=r"$\text{EI}_{\text{AD}}$", linewidth=2)
sns.kdeplot(x=data2, fill=True, color="red", label=r"$\text{EI}_{\text{AT}}$", linewidth=2)

# Add annotations
plt.axvline(0, color="black", linestyle="--", linewidth=1)
# plt.axvline(1, color="red", linestyle="--", linewidth=1, label="AT")

# Add titles and legend
plt.title("Empirical Distribution", fontsize=30)
plt.xlabel("Efficiency Indicators", fontsize=30)
plt.ylabel("Density", fontsize=30)
plt.legend(fontsize=30)

# Show the plot
plt.show()


