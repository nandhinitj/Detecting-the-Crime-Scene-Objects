import matplotlib
from Glob_Vars import Glob_Vars
from plot_gradient import gradient_bar, addlabels, gradient_image
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn import metrics
import matplotlib
matplotlib.use('TkAgg')
from prettytable import PrettyTable
from itertools import cycle
from sklearn.metrics import roc_curve


def Plot_Results():
    for a in range(1):
        Eval = np.load('Eval_all.npy', allow_pickle=True)[a]
        Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
        Graph_Term = [0,1,2 ,3, 4, 5,6,7,8, 9]
        Algorithm = ['TERMS', 'DO-TMA Yolov7', 'EOO-TMA Yolov7', 'AVOA-TMA Yolov7', 'LO-TMA Yolov7', 'MP-AVLO-TMA Yolov7']
        Classifier = ['TERMS',  'Yolov3', 'Yolov5', 'Yolov7', 'MP-AVLO-TMA Yolov7']

        value = Eval[4, :, 4:]
        value[:, :-1] = value[:, :-1] * 100
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, :])
        print('--------------------------------------------------Dataset_'+str(a+1)+'Algorithm Comparison - ',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[7 + j - 1, :])
        print('----------------------------------------------------Dataset_'+str(a+1)+'Classifier Comparison - ',
              '--------------------------------------------------')
        print(Table)

        Eval = np.load('Eval_all.npy', allow_pickle=True)[a]
        Batch =  [4, 8, 16, 32, 48, 64]
        for j in range(len(Graph_Term)):
            Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
            for k in range(Eval.shape[0]):
                for l in range(Eval.shape[1]):
                    if Graph_Term[j] == 9:
                        Graph[k, l] = Eval[k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = Eval[k, l, Graph_Term[j] + 4] * 100

            plt.plot(Batch, Graph[:, 0], color='r', linewidth=3, marker='*', markerfacecolor='blue', markersize=16,
                     label="DO-TMA Yolov7")
            plt.plot(Batch, Graph[:, 1], color='g', linewidth=3, marker='*', markerfacecolor='red', markersize=16,
                     label="EOO-TMA Yolov7")
            plt.plot(Batch, Graph[:, 2], color='b', linewidth=3, marker='*', markerfacecolor='green', markersize=16,
                     label="AVOA-TMA Yolov7")
            plt.plot(Batch, Graph[:, 3], color='c', linewidth=3, marker='*', markerfacecolor='cyan', markersize=16,
                     label="LO-TMA Yolov7")
            plt.plot(Batch, Graph[:, 4], color='m', linewidth=3, marker='*', markerfacecolor='black', markersize=16,
                     label="MP-AVLO-TMA Yolov7")
            plt.xlabel('Batch Size')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_line_1.png" % (str(a + 1), Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


def plot_gradient():
    for a in range(1):
        N = 4
        x = np.arange(N) + 0.15
        Eval = np.load('Eval_all.npy', allow_pickle=True)[a]
        Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
        Graph_Term = [0,1,2, 3, 4, 5,6,7,8, 9]
    # for b in range(6):
        for k in range(len(Graph_Term)):
            Evals = Eval[5,6:,4+Graph_Term[k]] *100
            fig, ax = plt.subplots()
            # fig, ax=plt.figure(figsize=(10, 5))
            ax.set(xlim=(0, 4), ylim=(0, round(max(Evals))+5))
            Glob_Vars.k = k
            # color1 = ['plt.cm.winter','plt.cm.winter','plt.cm.hot','plt.cm.copper','plt.cm.hsv']
            # background image
            if a==0:
                gradient_image(ax, direction=1, extent=(0, 1, 0, 1), transform=ax.transAxes,cmap=plt.cm.twilight, cmap_range=(0.2, 0.8), alpha=0.5)
            else:
                gradient_image(ax, direction=1, extent=(0, 1, 0, 1), transform=ax.transAxes, cmap=plt.cm.terrain,
                               cmap_range=(0.2, 0.8), alpha=0.5)

            y = Eval[5,5:,4+Graph_Term[k]]*100
            plt.xticks(x+0.3, ('Yolov3','Yolov5','Yolov7','MP-AVLO-TMA Yolov7'))
            plt.xlabel('Batch_Size --> 64')
            plt.ylabel(Terms[Graph_Term[k]])
            gradient_bar(ax, x, y, width=0.7)
            addlabels(x, y)
            path1 = "./Results/Dataset_%s_Terms_%s_bar.png" % (str(a+1),str(k+1))
            plt.savefig(path1)
            plt.show()





def Confusion_matrix():
    # Confusion Matrix
    for a in range(1):
        Eval = np.load('Eval_all.npy', allow_pickle=True)[a]
        value = Eval[3, 4, :5]
        val = np.asarray([0, 1, 1])
        data = {'y_Actual': [val.ravel()],
                'y_Predicted': [np.asarray(val).ravel()]
                }
        df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'][0], df['y_Predicted'][0], rownames=['Actual'], colnames=['Predicted'])
        value = value.astype('int')

        confusion_matrix.values[0, 0] = value[1]
        confusion_matrix.values[0, 1] = value[3]
        confusion_matrix.values[1, 0] = value[2]
        confusion_matrix.values[1, 1] = value[0]

        sn.heatmap(confusion_matrix, annot=True).set(title='Accuracy = ' + str(Eval[3, 4, 4] * 100)[:5] + '%')
        sn.plotting_context()
        path1 = './Results/Confusion_%s.png' %(str(a+1))
        plt.savefig(path1)
        plt.show()

def Plot_ROC():
    lw = 2
    cls = ['CNN', 'GRU', 'LSTM', 'RAN','ASPP-RAN']
    colors1 = cycle(["plum", "red", "palegreen", "chocolate", "navy", ])
    colors2 = cycle(["hotpink", "plum", "chocolate", "navy", "red", "palegreen", "violet", "red"])
    for n in range(1):
        for i, color in zip(range(5), colors1):  # For all classifiers
            Predicted = np.load('roc_score.npy', allow_pickle=True)[n][i].astype('float')
            Actual = np.load('roc_act.npy', allow_pickle=True)[n][i].astype('int')
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[:, -1], Predicted[:, -1].ravel())

            auc = metrics.roc_auc_score(Actual[:, -1], Predicted[:,
                                                       -1].ravel())

            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label="{0}".format(cls[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/_roc_%s.png"  %(str(n+1))
        plt.savefig(path1)
        plt.show()


def Plot_Fitness():

    for a in range(1):
        conv = np.load('Fitness.npy', allow_pickle=True)[a]
        ind = np.argsort(conv[:, conv.shape[1] - 1])
        x = conv[ind[0], :].copy()
        y = conv[4, :].copy()
        conv[4, :] = x
        conv[ind[0], :] = y

        Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Algorithm = ['DO-TMA Yolov7', 'EOO-TMA Yolov7', 'AVOA-TMA Yolov7', 'LO-TMA Yolov7', 'MP-AVLO-TMA Yolov7']

        Value = np.zeros((conv.shape[0], 5))
        for j in range(conv.shape[0]):
            Value[j, 0] = np.min(conv[j, :])
            Value[j, 1] = np.max(conv[j, :])
            Value[j, 2] = np.mean(conv[j, :])
            Value[j, 3] = np.median(conv[j, :])
            Value[j, 4] = np.std(conv[j, :])

        Table = PrettyTable()
        Table.add_column("ALGORITHMS", Statistics)
        for j in range(len(Algorithm)):
            Table.add_column(Algorithm[j], Value[j, :])
        print('--------------------------------------------------Dataset_'+str(a+1)+'Statistical Analysis--------------------------------------------------')
        print(Table)

        iteration = np.arange(conv.shape[1])
        plt.plot(iteration, conv[0, :], color='r', linewidth=3, marker='>', markerfacecolor='blue', markersize=8,
                 label="DO-TMA Yolov7")
        plt.plot(iteration, conv[1, :], color='g', linewidth=3, marker='>', markerfacecolor='red', markersize=8,
                 label="EOO-TMA Yolov7")
        plt.plot(iteration, conv[2, :], color='b', linewidth=3, marker='>', markerfacecolor='green', markersize=8,
                 label="AVOA-TMA Yolov7")
        plt.plot(iteration, conv[3, :], color='m', linewidth=3, marker='>', markerfacecolor='yellow', markersize=8,
                 label="LO-TMA Yolov7")
        plt.plot(iteration, conv[4, :], color='k', linewidth=3, marker='>', markerfacecolor='cyan', markersize=8,
                 label="MP-AVLO-TMA Yolov7")
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=4)
        path1 = "./Results/conv_%s.png"  %(str(a+1))
        plt.savefig(path1)
        plt.show()


def Plot_Method():
    Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'TNR', 'NPV', 'FDR', 'F1-Score',
             'MCC']
    species = ('CNN', 'GRU', 'LSTM', 'RAN', 'ASPP-RAN')
    eval = np.load('Eval_all2.npy', allow_pickle=True)
    Classifier = ['TERMS', 'CNN', 'GRU', 'LSTM', 'RAN', 'ASPP-RAN']
    activation_function = [4, 8, 16, 32, 48, 64]
    for z in range(6):
        value = eval[z, :, 4:]
        value[:, :-1] = value[:, :-1] * 100
        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[j, :])
        print('--------------------------------------------------Batch size ', activation_function[z],
              '- Classifier Comparison --------------------------------------------------')
        print(Table)

    Eval = np.load('Eval_all2.npy', allow_pickle=True)
    Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
    for k in range(Eval.shape[0]):
        for l in range(Eval.shape[1]):
            Graph[k, l] = Eval[k, l, 0 + 4] * 100
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    X = np.arange(6)
    ax.bar(X + 0.00, Graph[:, 0], color='#fedf08', width=0.10, label="CNN")
    ax.bar(X + 0.10, Graph[:, 1], color='#cc9f3f', width=0.10, label="GRU")
    ax.bar(X + 0.20, Graph[:, 2], color='#ca0147', width=0.10, label="LSTM")
    ax.bar(X + 0.30, Graph[:, 3], color='#0d75f8', width=0.10, label="RAN")
    ax.bar(X + 0.40, Graph[:, 4], color='#00fbb0', width=0.10, label="ASPP-RAN")
    plt.xticks(X + 0.25, ('4', '8', '16', '32', '48', '64'))
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy')
    plt.legend(loc=1)
    path1 = "./Results/Accuracy_bar_2.png"
    plt.savefig(path1)
    plt.show()
    for j in range(4):
        if j == 0:
            Evs = {'Precision': Eval[5, :, 7] * 100, 'FDR': Eval[5, :, 13] * 100, }
        elif j == 1:
            Evs = {'Sensitivity': Eval[5, :, 5] * 100, 'FPR': Eval[5, :, 8] * 100, }
        elif j == 2:
            Evs = {'NPV': Eval[5, :, 12] * 100, 'FOR': Eval[5, :, 10] * 100, }
        else:
            Evs = {'Sensitivity': Eval[5, :, 5] * 100, 'FNR': Eval[5, :, 9] * 100, }
        width = 0.6  # the width of the bars: can also be len(x) sequence
        fig, ax = plt.subplots()
        bottom = np.zeros(5)
        for stck, Ev in Evs.items():
            p = ax.bar(species, Ev, width, label=stck, bottom=bottom)
            bottom += Ev
            ax.bar_label(p, label_type='center')
        ax.set_title('Batch_Size(64)')
        path1 = "./Results/_%s_bar_2.png" % (j + 1)
        plt.legend(loc=4)
        plt.savefig(path1)
        plt.show()

if __name__ == "__main__":
    Plot_Results()
    plot_gradient()
    Confusion_matrix()
    Plot_ROC()
    Plot_Fitness()
    Plot_Method()