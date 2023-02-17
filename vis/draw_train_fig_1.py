from matplotlib import pyplot as plt
import numpy as np


def get_fig_data(filepath):
    data_lst = []
    with open(filepath, 'r') as f:
        flag_log = False  # 是否处于记录数据的状态
        cnt = 0
        epoch_lst = []
        for line in f:
            if line == "\n":
                continue
            if line.lstrip().startswith('IoU'):
                flag_log = True
                continue
            if flag_log is True:
                tmp_lst = line.split(' ')
                epoch_lst.append(float(tmp_lst[-1]))
                cnt += 1
                if cnt == 12:
                    # print(epoch_lst)
                    cnt = 0
                    flag_log = False
                    data_lst.append(epoch_lst)
                    epoch_lst = []
            elif flag_log is False:
                continue
    return np.array(data_lst)


fp = './log/log_v002.txt'

data = get_fig_data(fp)

n_row = 1
n_col = 2
fig, axs = plt.subplots(n_row, n_col, figsize=(11, 5))
line_sst = ['r*--', 'g*-', 'm--', 'grey']
marker_sst = ['.','+','x','D']
label_sst = []
x_lab_sst = ['x','x']
y_lab_sst = ['y','y']
title_sst = ['title_1','title_2','title_3','title_4']
idx = 0


def subplot(ax, data, ylabel):
    ax.plot(data, line_sst[0], marker=marker_sst[0], linewidth=1.)
    ax.legend(loc='lower right') # 图例位置
    ax.set_xlabel('epoch')
    ax.set_ylabel(ylabel)


ax1, ax2 = axs
subplot(ax1, data[:, 0], 'AP--0.50:0.95')
subplot(ax2, data[:, 6], 'AR--0.50:0.95')
plt.tight_layout() # 让图形不那么挤
plt.show()
fig.savefig('./vis/train_fig.svg', format='svg', dpi=150)
