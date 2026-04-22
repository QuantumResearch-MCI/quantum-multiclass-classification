import pandas as pd
import numpy as np
from IPython.display import display, HTML

NEW=True

def report(df, results=None, data_metadata=None, is_aggregated=False):
    if NEW:
        """Print a formatted report from the aggregated results DataFrame."""

        # ── Full Table ────────────────────────────────────────────────
        print("=" * 80)
        print("MODEL COMPARISON REPORT")
        print("=" * 80)

        display_df = df.copy()
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            col = f'{metric}_mean'
            std = f'{metric}_std'
            if col in df.columns and std in df.columns:
                display_df[metric] = display_df.apply(
                    lambda r: f"{r[col]:.3f} ± {r[std]:.3f}", axis=1
                )

        print(display_df[[m for m in ['accuracy', 'precision', 'recall', 'f1'] 
                        if m in display_df.columns]].to_string())

        # ── Best / Worst ──────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("HIGHLIGHTS")
        print("=" * 80)

        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in metrics:
            col = f'{metric}_mean'
            if col not in df.columns:
                continue
            best_model  = df[col].idxmax()
            worst_model = df[col].idxmin()
            best_val    = df[col].max()
            worst_val   = df[col].min()
            print(f"\n{metric.upper()}")
            print(f"  Best  : {best_model:15s}  {best_val:.3f}")
            print(f"  Worst : {worst_model:15s}  {worst_val:.3f}")

        # ── Most / Least Stable (std) ─────────────────────────────────
        print("\n" + "=" * 80)
        print("STABILITY (lower std = more stable)")
        print("=" * 80)

        acc_std = 'accuracy_std'
        if acc_std in df.columns:
            most_stable  = df[acc_std].idxmin()
            least_stable = df[acc_std].idxmax()
            print(f"\n  Most stable  : {most_stable:15s}  std={df[acc_std].min():.3f}")
            print(f"  Least stable : {least_stable:15s}  std={df[acc_std].max():.3f}")

    else:
        metadata_count = pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'N': [len(data_metadata['X_train']), len(data_metadata['X_test'])] 
        })
        display(metadata_count.style.set_caption("Dataset Metadata"))

        distribution_df = pd.DataFrame({
            'Train': pd.Series(data_metadata['y_train']).value_counts(),
            # 'Validation': pd.Series(data_metadata['y_val']).value_counts(),
            'Test': pd.Series(data_metadata['y_test']).value_counts()
        }).fillna(0).astype(int)
        # distribution_df.index = distribution_df.index.map(dict(enumerate(data_metadata['classes'])))

        display(distribution_df.style.set_caption("Komposisi Kelas Dataset"))

        def get_display_name(r):
            model_name = r.get('model', '')
            kernel_name = str(r.get('kernel', 'None'))
            
            if model_name:
                if kernel_name == 'None':
                    return model_name
                else:
                    return f"{model_name} - {kernel_name.title()}"
            else:
                return kernel_name.title() if kernel_name != 'None' else 'None'

        if is_aggregated:
            rows = []
            for r in results:
                disp_name = get_display_name(r)
                lst_accu  = list(r.get('accuracy_raw', []))
                rows.append({
                    "Model": disp_name,
                    "List Acc":  [f"{a:.4f}" for a in lst_accu],
                    "Max Acc":   f"{max(lst_accu):.3f}",
                    "Min Acc":   f"{min(lst_accu):.3f}",
                    "Overall": f"{np.mean(lst_accu):.3f}",
                    "Std": f"{np.std(lst_accu):.3f}",
                    # "Val Acc": f"{r['val_accuracy_mean']:.3f}±{r['val_accuracy_std']:.3f}",
                    # "Acc": f"{r['accuracy_mean']:.3f}±{r['accuracy_std']:.3f}",
                    "Acc": f"{np.mean(lst_accu):.3f}±{np.std(lst_accu):.3f}",
                    # "Weighted Prec": f"{r['weighted_precision_mean']:.3f}±{r['weighted_precision_std']:.3f}",
                    "Weighted Prec": f"{r['weighted_precision_mean']:.3f}±{r['weighted_precision_std']:.3f}",
                    "Weighted Recall": f"{r['weighted_recall_mean']:.3f}±{r['weighted_recall_std']:.3f}",
                    "Weighted F1": f"{r['weighted_f1-score_mean']:.3f}±{r['weighted_f1-score_std']:.3f}",
                    "Macro Prec": f"{r['macro_precision_mean']:.3f}±{r['macro_precision_std']:.3f}",
                    "Macro Recall": f"{r['macro_recall_mean']:.3f}±{r['macro_recall_std']:.3f}",
                    "Macro F1": f"{r['macro_f1-score_mean']:.3f}±{r['macro_f1-score_std']:.3f}",
                    "Total(s)": f"{r['total_time_mean']:.2f}±{r['total_time_std']:.2f}",
                    "Best Params": f"{r['best_params']}",
                    "Best Model": f"{r['best_model']}"
                })

            df = pd.DataFrame(rows)
            df.columns = pd.MultiIndex.from_tuples([
                ("", "Model"),
                ("Accuracy", "List"),
                ("Accuracy", "Max"),
                ("Accuracy", "Min"),
                ("Accuracy", "Overall"),
                ("Accuracy", "Std"),
                # ("", "Val Acc"),
                ("Accuracy", "Acc Result"),
                ("Weighted", "Prec"),
                ("Weighted", "Recall"),
                ("Weighted", "F1"),
                ("Macro", "Prec"),
                ("Macro", "Recall"),
                ("Macro", "F1"),
                ("", "Total(s)"),
                ("HPO", "Best Params"),
                ("HPO", "Best Model")
            ])
            display(
                df
                .style
                .set_caption("Performance Summary (Mean ± Std)")
                .hide(axis="index")
                .set_properties(**{'text-align': 'center'})
                .set_table_styles([
                    {'selector': 'th', 'props': [('text-align', 'center'), ('border', '1px solid gray')]},
                    {'selector': 'td', 'props': [('text-align', 'center'), ('border', '1px solid gray'), ('padding', '6px')]}
                ])
            )
        else:
            rows = []
            for r in results:
                disp_name = get_display_name(r)
                rows.append({
                    "Model": disp_name,
                    # "Val Acc": f"{r['val_accuracy']:.3f}",
                    "Acc": f"{r['accuracy']:.3f}",
                    "Prec": f"{r['precision']:.3f}",
                    "Recall": f"{r['recall']:.3f}",
                    "F1": f"{r['f1-score']:.3f}",
                    "Train Time(s)": f"{r['train_time']:.2f}",
                    "Val Time(s)": f"{r['val_time']:.2f}",
                    "Test Time(s)": f"{r['test_time']:.2f}",
                    "Total(s)": f"{r['total_time']:.2f}"
                })
            df = pd.DataFrame(rows)
            display(df.style.set_caption("Performance Summary").hide(axis="index"))

        print("=" * 150)