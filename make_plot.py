import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch

def show_AUROC(true_label,prob,title):
    plt.figure(figsize=(4,4),dpi=300)
    fpr, tpr, thresholds = roc_curve(true_label+1,prob+1)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]

    sens, spec = tpr[ix], 1-fpr[ix]
    auroc = roc_auc_score(true_label, prob)

    plt.plot(fpr, tpr)

    plt.xlabel('FP Rate')
    plt.ylabel('TP Rate')
    plt.scatter(fpr[ix],tpr[ix],marker='+',color='r', label = 'Best threshold = %.3f \nSensitivity = %.3f \nSpecificity = %.3f \nAUROC = %.3f' % (best_thresh-1, sens, spec, auroc))
    plt.legend()

    plt.title(title)

    plt.show()
    return best_thresh-1

def plot_alpha(alphas,cols):
    fig, ax = plt.subplots(figsize=(100, 100))
    im = ax.imshow(alphas)
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(65))
    ax.set_xticklabels(["t-"+str(i) for i in np.arange(24, 0, -4)])
    ax.set_yticklabels(list(cols))
    for i in range(len(cols)):
        for j in range(6):
            text = ax.text(j, i, round(alphas[i, j], 3),
                        ha="center", va="center", color="w")
    ax.set_title("Importance of features and timesteps")
    plt.show()

def make_betas(betas,cols):
    betas = pd.DataFrame([cols,betas]).transpose()
    betas.columns = ['var','betas']
    betas_non_presense = betas.copy()
    betas_non_presense = betas_non_presense[betas_non_presense['var'].isin([x for x in cols if x[:10]!='s:presense'])]

    betas['betas'] = betas['betas'] - betas['betas'].mean()
    betas.sort_values(by='betas',inplace=True)

    betas_non_presense['betas'] = betas_non_presense['betas'] - betas_non_presense['betas'].mean()
    betas_non_presense.sort_values(by='betas',inplace=True)
    return betas,betas_non_presense

def plot_beta(betas):
    plt.figure(figsize=(25, 8))
    plt.title("Feature importance")
    plt.bar(range(len(betas)), betas['betas'])
    plt.xticks(ticks=range(len(betas)), labels=betas['var'], rotation=90, size=10, fontsize=10)
    plt.show()

def make_transition_test_for_Dead(df,target,rolling_size=24,batch_size=256):

    s_col = [x for x in df if x[:2]=='s:']
    a_col = [x for x in df if x[:2]=='a:']
    r_col = [x for x in df if x[:2]=='r:']

    dict = {}
    dict['traj'] = {}

    s  = []
    a  = []
    dead = []
    patients = []
    dead_24hrs = []

    for traj in tqdm(df.traj.unique()):
        df_traj = df[df['traj'] == traj]
        dict['traj'][traj] = {'s':[],'a':[]}
        dict['traj'][traj]['s'] = df_traj[s_col].values.tolist()
        dict['traj'][traj]['a'] = df_traj[a_col].values.tolist()
        dict['traj'][traj]['dead'] = df_traj[target].values.tolist()

        final_status = -1 if sum(df_traj[target].values.tolist()) < 0 else 0

        step_len = len(df_traj) - rolling_size + 1
        
        for step in range(step_len):
            s.append(dict['traj'][traj]['s'][step:step+rolling_size])
            a.append(dict['traj'][traj]['a'][step+rolling_size-1])
            dead.append(dict['traj'][traj]['dead'][step+rolling_size-1])
            patients.append(final_status)

            if (step >= step_len - 6)&(final_status==-1):
                dead_24hrs.append(-1)
            else :
                dead_24hrs.append(0)
    
    s = torch.FloatTensor(np.float32(s))
    a = torch.LongTensor(np.int64(a))
    dead = torch.FloatTensor(np.float32(dead))
    patients = torch.LongTensor(np.int64(patients))
    dead_24hrs = torch.LongTensor(np.int64(dead_24hrs))

    Dataset = TensorDataset(s,a,dead,patients,dead_24hrs)
    rt = DataLoader(Dataset,batch_size,shuffle=False)
    return rt

def make_transition_test_for_AKI(df,target,rolling_size=24,batch_size=256):

    s_col = [x for x in df.columns if x[:2]=='s:']
    a_col = [x for x in df.columns if x[:2]=='a:']
    r_col = [x for x in df.columns if x[:2]=='r:']

    dict = {}
    dict['traj'] = {}

    s, a, AKI1, AKI2, AKI3, dead, patients, dead_24hrs = ([] for _ in range(8))

    for traj in tqdm(df.traj.unique()):

        df_traj = df[df['traj'] == traj]
        dict['traj'][traj] = {'s':[],'a':[]}
        dict['traj'][traj]['s'] = df_traj[s_col].values.tolist()
        dict['traj'][traj]['a'] = df_traj[a_col].values.tolist()
        dict['traj'][traj]['AKI1'] = df_traj['AKI1'].values.tolist()
        dict['traj'][traj]['AKI2'] = df_traj['AKI2'].values.tolist()
        dict['traj'][traj]['AKI3'] = df_traj['AKI3'].values.tolist()
        dict['traj'][traj]['dead'] = df_traj[target].values.tolist()

        final_status = -1 if sum(df_traj[target].values.tolist()) < 0 else 0

        step_len = len(df_traj) - rolling_size + 1

        for step in range(step_len):
            s.append(dict['traj'][traj]['s'][step:step+rolling_size])
            a.append(dict['traj'][traj]['a'][step+rolling_size-1])            
            AKI1.append(dict['traj'][traj]['AKI1'][step+rolling_size-1])
            AKI2.append(dict['traj'][traj]['AKI2'][step+rolling_size-1])
            AKI3.append(dict['traj'][traj]['AKI3'][step+rolling_size-1])
            dead.append(dict['traj'][traj]['dead'][step+rolling_size-1])
            patients.append(final_status)

            if (step >= step_len - 6)&(final_status==-1):
                dead_24hrs.append(-1)
            else :
                dead_24hrs.append(0)
    
    s = torch.FloatTensor(np.float32(s))
    a = torch.LongTensor(np.int64(a))
    AKI1 = torch.LongTensor(np.int64(AKI1))
    AKI2 = torch.LongTensor(np.int64(AKI2))
    AKI3 = torch.LongTensor(np.int64(AKI3))
    dead = torch.LongTensor(np.int64(dead))
    patients = torch.LongTensor(np.int64(patients))
    dead_24hrs = torch.LongTensor(np.int64(dead_24hrs)) 

    Dataset = TensorDataset(s,a,AKI1,AKI2,AKI3,dead,patients,dead_24hrs)
    rt = DataLoader(Dataset,batch_size,shuffle=False)
    return rt