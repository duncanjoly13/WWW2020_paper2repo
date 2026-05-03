import optuna
import optuna.importance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
})

sns.set_theme(style="whitegrid", rc={"font.family": "serif"})

def plot_baseline_vs_optimized(best_trial):
    metrics = ['Catalog Coverage', 'PMAP@10', 'MRR@10', 'ILD@10']
    
    baseline = [0.0066, 0.3951, 0.4083, 0.9727]
    
    optimized = [
        best_trial.user_attrs.get('coverage', 0.0),
        best_trial.value,
        best_trial.user_attrs.get('mrr', 0.0),
        best_trial.user_attrs.get('ild', 0.0)
    ]
    
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, baseline, width, label='Baseline', color='#4C72B0')
    rects2 = ax.bar(x + width/2, optimized, width, label='Optimized', color='#55A868')

    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison: Baseline vs. Optimized P2R')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper left')

    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')

    plt.savefig('fig1_baseline_comparison.pdf')
    plt.close()

def plot_parameter_importance(study):
    importance_dict = optuna.importance.get_param_importances(study)
    
    df = pd.DataFrame({
        'Hyperparameter': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Importance', y='Hyperparameter', data=df, palette='viridis', ax=ax)
    
    ax.set_title('Hyperparameter Importance for PMAP@10')
    ax.set_xlabel('Relative Importance')
    ax.set_ylabel('')
    
    plt.savefig('fig2_param_importance.pdf')
    plt.close()

def plot_optimization_history(study):
    trials = study.trials
    completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    trial_numbers = [t.number for t in completed_trials]
    values = [t.value for t in completed_trials]
    
    best_values = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(trial_numbers, values, alpha=0.5, label='Trial Score', color='#4C72B0')
    ax.plot(trial_numbers, best_values, color='#C44E52', linewidth=2, label='Best Value')
    
    ax.set_title('Optimization History (PMAP@10)')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('PMAP@10 Score')
    ax.legend()
    
    plt.savefig('fig3_optimization_history.pdf')
    plt.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate paper figures from Optuna database.")
    parser.add_argument('--db_path', type=str, default='p2r_study.db', help='Path to the SQLite database file.')
    args = parser.parse_args()
    
    storage_url = f"sqlite:///{args.db_path}"
    
    try:
        study = optuna.load_study(study_name="P2R_Hyperparameter_Sweep", storage=storage_url)
        print(f"Loaded study with {len(study.trials)} trials from {args.db_path}")
        
        plot_parameter_importance(study)
        plot_optimization_history(study)
        plot_baseline_vs_optimized(study.best_trial)
        
        print("All figures generated successfully.")
    except Exception as e:
        print(f"Error loading database at {args.db_path}: {e}")
