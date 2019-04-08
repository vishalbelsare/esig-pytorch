import tosig_pytorch
import torch
import time
import pandas as pd

signal_dimensions = [4]
sig_degrees = [4, 5]
results = {}
for device in ['cpu', 'cuda']:
    print('\n \n On ' + device + '\n')
    ept = tosig_pytorch.EsigPyTorch(device=device)
    times_sig = {}
    times_logsig = {}
    for signal_dimension in signal_dimensions:
        stream = ept.brownian(100, signal_dimension)
        intra_dic_sig = {}
        intra_dic_logsig = {}
        for signature_degree in sig_degrees:
            # Signature
            print('signature-deg = {}, signal-dim = {} \n Starting Sig...'.format(signature_degree, signal_dimension))
            start = time.time()
            sig = ept.stream2sig(stream, signature_degree)
            end = time.time()
            intra_dic_sig[signature_degree] = end - start
            print('time taken for signature = {} secs \n Starting Log_sig...'.format(end-start))
            # Log-Signature
            start = time.time()
            logsig = ept.stream2logsig(stream, signature_degree)
            end = time.time()
            print('time taken for log-signature = {} secs \n __Done__ \n'.format(end-start))
            intra_dic_logsig[signature_degree] = end - start
        times_sig[signal_dimension] = intra_dic_sig
        times_logsig[signal_dimension] = intra_dic_logsig
    # Signature df
    df_sig = pd.DataFrame.from_dict(times_sig)
    df_sig.columns = ['signal_dim: {}'.format(sd) for sd in df_sig.columns]
    df_sig.index = ['sig_degree: {}'.format(sd) for sd in df_sig.index]
    results['signature_' + device + '_times'] = df_sig
    # Log-Signature df
    df_logsig = pd.DataFrame.from_dict(times_logsig)
    df_logsig.columns = ['signal_dim: {}'.format(sd) for sd in df_logsig.columns]
    df_logsig.index = ['sig_degree: {}'.format(sd) for sd in df_logsig.index]
    results['log-signature_' + device + '_times'] = df_logsig