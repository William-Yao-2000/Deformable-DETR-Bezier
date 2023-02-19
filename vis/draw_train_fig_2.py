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


f1 = './log/log_v002-no_aux_loss.txt'
f2 = './log/log_v002.txt'

data1, data2 = get_fig_data(f1), get_fig_data(f2)
print(data1)
print("")
print(data2)


f3 = './log/recog/log_v002.txt'
f4 = './log/recog/log_v002-bbox-ref.txt'

data3, data4 = get_fig_data(f3), get_fig_data(f4)
print("")
print(data3)
print("")
print(data4)


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

def subplot(ax, data1, data2, data3, data4, ylabel):
    ax.plot(data1, line_sst[0], marker=marker_sst[0], linewidth=1., label='bezier--no aux loss')
    ax.plot(data2, line_sst[1], marker=marker_sst[1], linewidth=1., label='bezier--with aux loss')
    ax.plot(data3, line_sst[2], marker=marker_sst[2], linewidth=1., label='recog--with aux loss')
    ax.plot(data4, line_sst[3], marker=marker_sst[3], linewidth=1., label='recog--bbox refinement')
    ax.legend(loc='lower right') # 图例位置
    ax.set_xlabel('epoch')
    ax.set_ylabel(ylabel)



ax1, ax2 = axs
subplot(ax1, data1[:, 0], data2[:, 0], data3[:, 0], data4[:, 0], 'AP--0.50:0.95')
subplot(ax2, data1[:, 6], data2[:, 6], data3[:, 6], data4[:, 6],  'AR--0.50:0.95')
plt.tight_layout() # 让图形不那么挤
plt.show()
fig.savefig('./vis/train_fig.svg', format='svg', dpi=150)
