import os
import argparse
import pickle as pkl
import numpy as np

import torch
from pytorch_lightning import Trainer
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from p2r_module import P2rSystem

def loadData(data_root, train_div):
    """
    Load the graph dataset once to be shared across all Optuna trials
    """
    with open('%s/ind.paper-repo.data' % data_root, 'rb') as f:
        content = pkl.load(f)

    paper_edge_index = []
    for idx, edges in enumerate(content['paperGraphAdjList']):
        for edge in sorted(list(edges)):
            paper_edge_index.append([idx, edge])
            
    repo_edge_index = []
    for idx, edges in enumerate(content['repoGraphAdjList']):
        for edge in sorted(list(edges)):
            repo_edge_index.append([idx, edge])
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    p2r_data = {
        'paper_graph_adjlist': content['paperGraphAdjList'],
        'paper_edge_index': torch.LongTensor(paper_edge_index).t().contiguous().to(device),
        'paper_features': torch.LongTensor(content['paperFeatures']).to(device),
        'cofork_repo_graph_adjlist': content['coforkRepoGraphAdjList'],
        'repo_graph_adjlist': content['repoGraphAdjList'],
        'repo_edge_index': torch.LongTensor(repo_edge_index).t().contiguous().to(device),
        'repo_features': torch.LongTensor(content['repoFeatures']).to(device),
        'repo_tags': torch.LongTensor(content['repoTags']).to(device),
        'positives': content['positives'],
        'bridge_length': int(content['bridgeLength'] * train_div),
        'bridge_ids': torch.LongTensor(list(filter(lambda x: x < int(content['bridgeLength'] * train_div), content['bridgeIds']))).to(device),
        'word_embeddings': torch.FloatTensor(content['wordEmbeddings']).to(device)
    }

    print('Training div: {} BLength {} {}'.format(train_div, p2r_data['bridge_length'], len(p2r_data['bridge_ids'].tolist())))
    return p2r_data

def objective(trial, data, default_args):
    """
    Generate a new set of hyperparameters, train the model, and return the metric to be maximized
    """
    hparams = argparse.Namespace(**vars(default_args))
    
    # hyperparameter search space ---
    hparams.learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    hparams.max_nb_epochs = trial.suggest_categorical("max_nb_epochs", [4, 8, 16])
    hparams.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    hparams.gcn_mid_dim = trial.suggest_categorical("gcn_mid_dim", [128, 256, 512, 1024])
    hparams.gcn_drop_prob = trial.suggest_float("gcn_drop_prob", 0.1, 0.6)
    hparams.freeze_embeddings = trial.suggest_categorical("freeze_embeddings", [True, False])
    
    hparams.top_t = trial.suggest_int("top_t", 5, 20, step=5)
    hparams.warploss_margin = trial.suggest_float("warploss_margin", 0.1, 0.8)

    model = P2rSystem(hparams, data)

    trainer = Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        max_epochs=hparams.max_nb_epochs,
        accelerator='auto',
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="tng_loss")
        ]
    )

    trainer.fit(model)
    trainer.test(model)

    trial.set_user_attr("coverage", trainer.callback_metrics.get("test_catalog_coverage").item())
    trial.set_user_attr("mrr", trainer.callback_metrics.get("avg_MRR@10").item())
    trial.set_user_attr("ild", trainer.callback_metrics.get("avg_ILD@10").item())
    
    return trainer.callback_metrics.get("avg_PMAP@10").item()

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default=os.path.join(os.path.dirname('/content/data'), 'data'), type=str)
    parser.add_argument('--train_div', default=1.0, type=float)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--catalog_size', default=7516, type=int)
    parser.add_argument('--total_onehop', default=20, type=int)
    parser.add_argument('--total', default=50, type=int)
    parser.add_argument('--gcn_output_dim', default=256, type=int)
    
    parser.add_argument('--txtcnn_pfilter_num1', default=64, type=int)
    parser.add_argument('--txtcnn_pfilter_num2', default=64, type=int)
    parser.add_argument('--txtcnn_pfilter_num3', default=64, type=int)
    parser.add_argument('--txtcnn_pfilter_num4', default=64, type=int)
    parser.add_argument('--txtcnn_rfilter_num1', default=64, type=int)
    parser.add_argument('--txtcnn_rfilter_num2', default=32, type=int)
    parser.add_argument('--txtcnn_drop_prob', default=0.0, type=float)

    args = parser.parse_args()

    print("Loading dataset into memory...")
    p2r_data = loadData(args.data_root, args.train_div)

    study = optuna.create_study(
        direction="maximize", 
        study_name="P2R_Hyperparameter_Sweep",
        storage="sqlite:///p2r_study.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    
    print("\nStarting Hyperparameter Sweep...")
    study.optimize(lambda trial: objective(trial, p2r_data, args), n_trials=30)
    
    print("\n" + "="*30)
    print("Optimization Finished!")
    print(f"Best PMAP@10 Score: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*30 + "\n")
  
    try:
        fig1 = optuna.visualization.plot_param_importances(study)
        fig1.show()
        
        fig2 = optuna.visualization.plot_parallel_coordinate(study)
        fig2.show()
    except Exception as e:
        print("Could not display plots. Ensure 'plotly' is installed.")
