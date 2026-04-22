import numpy as np
import pandas as pd

NON_NUMERIC = {'model', 'kernel', 'best_params', 'best_model', 'random_state', 'confusion_matrix', 'y_true', 'y_prob'}
NEW=True
def aggregate_results(all_results, n_splits=5):
    if NEW:
        rows = []
        confusion_matrices = {}
        for name, metrics in all_results.items():
            parts = name.split('_', 1)
            model  = parts[0]
            kernel = parts[1] if len(parts) > 1 else '-'
            row = {'model': model, 'kernel': kernel}
            for metric, values in metrics.items():
                if metric == 'confusion_matrix':
                    confusion_matrices[name] = np.sum(values, axis=0).astype(int)
                elif metric not in NON_NUMERIC:
                    row[f'{metric}_mean'] = np.mean(values)
                    row[f'{metric}_std']  = np.std(values)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.index = df['model'] + ' ' + df['kernel']
        df.index.name = 'model_kernel'

        return df, confusion_matrices
    else:
        aggregated = {}
        
        # all_results sekarang adalah flat list of dicts
        for r in all_results:
            # Ambil nama model dan kernel
            mod = r.get('model', 'Unknown')
            k = str(r.get('kernel', 'None'))
            best_params = r.get('best_params', None)
            best_model = r.get('best_model', None)
            
            # Buat key unik untuk pengelompokan (contoh: "SVC_rbf")
            group_key = f"{mod}_{k}"
            
            if group_key not in aggregated:
                aggregated[group_key] = {
                    'model': mod,
                    'kernel': k,
                    'best_params': best_params,
                    'best_model': best_model,
                    'metrics': {key: [] for key in r if key not in [*NON_NUMERIC]},
                    'confusion_matrices': []
                }
            
            # Masukkan nilai-nilai metrik ke dalam list
            for key in aggregated[group_key]['metrics']:
                aggregated[group_key]['metrics'][key].append(r[key])
                
            # Kumpulkan confusion matrix
            if 'confusion_matrix' in r:
                aggregated[group_key]['confusion_matrices'].append(r['confusion_matrix'])
                    
        final_results = []
        
        # Hitung Mean, Std, dan SUM untuk confusion matrix
        for group_key, data in aggregated.items():
            res = {
                'model': data['model'], 
                'kernel': data['kernel'],
                'best_params': data['best_params'],
                'best_model': data['best_model']
            }
            
            # Rata-rata dan Standar Deviasi
            for metric, values in data['metrics'].items():
                if metric in NON_NUMERIC:
                    continue

                arr = np.array(values)
                res[f"{metric}_mean"] = np.mean(arr)
                res[f"{metric}_std"] = np.std(arr)
                res[f"{metric}_raw"] = arr
                
            # Jumlahkan (Sum) Confusion Matrix
            if data['confusion_matrices']:
                res['confusion_matrix_sum'] = np.sum(data['confusion_matrices'], axis=0)
                
            final_results.append(res)
            
        return final_results