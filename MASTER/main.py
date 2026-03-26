from master import MASTERModel
import pickle
import numpy as np
import time

# Please install qlib first before load the data.

universe = 'csi300' # ['csi300','csi800']
prefix = 'opensource' # ['original','opensource'], which training data are you using
train_data_dir = f'./data/MASTER-data'
with open(f'{train_data_dir}/{prefix}/{universe}_dl_train.pkl', 'rb') as f:
    dl_train = pickle.load(f)

predict_data_dir = f'./data/MASTER-data/opensource'
with open(f'{predict_data_dir}/{universe}_dl_valid.pkl', 'rb') as f:
    dl_valid = pickle.load(f)
with open(f'{predict_data_dir}/{universe}_dl_test.pkl', 'rb') as f:
    dl_test = pickle.load(f)

print("Data Loaded.")


d_feat = 158
d_model = 256
t_nhead = 4
s_nhead = 2
dropout = 0.5
gate_input_start_index = 158
gate_input_end_index = 221

if universe == 'csi300':
    beta = 5
elif universe == 'csi800':
    beta = 2

n_epoch = 40
lr = 1e-5
GPU = 2
train_stop_loss_thred = 0.95

batch_size=128

ic = []
icir = []
ric = []
ricir = []

# Training
######################################################################################
for seed in [0,1]:
    print(f"\n{'='*20} Start Training Seed: {seed} {'='*20}")
    
    # 初始化模型
    model = MASTERModel(
        d_feat=d_feat, d_model=d_model, t_nhead=t_nhead, s_nhead=s_nhead, 
        T_dropout_rate=dropout, S_dropout_rate=dropout,
        beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
        n_epochs=n_epoch, lr=lr, GPU=GPU, seed=seed, train_stop_loss_thred=train_stop_loss_thred,
        save_path='model', save_prefix=f'{universe}_{prefix}', 
        batch_size=batch_size # 传入我们修改支持的 batch_size
    )

    start_time = time.time()
    
    # ==================== 开始训练 (Training) ====================
    print("Training Model...")
    # fit 方法封装了每个 epoch 的训练和验证，并会在达到 stop_loss_thred 时自动保存最优模型并早停
    model.fit(dl_train, dl_valid)
    print("Model Trained Successfully.")
    # ============================================================

    # ==================== 开始测试 (Testing) ====================
    print("Testing Model...")
    # 如果你想直接加载之前训好的模型跳过训练，可以解开下面这行的注释，并注释掉上面的 model.fit()
    # model.load_param(f'model\\{universe}_{prefix}_{seed}.pkl') 
    
    predictions, metrics = model.predict(dl_test)
    # ============================================================
    
    running_time = time.time() - start_time
    print('Seed: {:d} time cost : {:.2f} sec'.format(seed, running_time))
    print("Metrics for current seed:", metrics)

    ic.append(metrics['IC'])
    icir.append(metrics['ICIR'])
    ric.append(metrics['RIC'])
    ricir.append(metrics['RICIR'])

print(f"\n{'='*20} Final Results {'='*20}")
print("IC: {:.4f} ± {:.4f}".format(np.mean(ic), np.std(ic)))
print("ICIR: {:.4f} ± {:.4f}".format(np.mean(icir), np.std(icir)))
print("RIC: {:.4f} ± {:.4f}".format(np.mean(ric), np.std(ric)))
print("RICIR: {:.4f} ± {:.4f}".format(np.mean(ricir), np.std(ricir)))