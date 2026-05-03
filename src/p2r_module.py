import os
import sys
import itertools
import random
from collections import OrderedDict

import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from test_tube import Experiment, HyperOptArgumentParser

from models import P2r
from metrics import warpLoss, metricMAP, metricMRR, metricAccuracy, metricPMAP
from dataset import P2rTrainDataset, P2rTestDataset

def calculate_ild(item_embeddings):
    """
    Calculate Intra-List Diversity for a batch of recommendation lists
    :param item_embeddings: tensor of shape (batch_size, top_k, embedding_dim)
    :return: average ILD across the batch
    """
    # ensure item_embeddings is a 3D tensor: (batch_size, top_k, embedding_dim)
    if item_embeddings.dim() == 2:
        item_embeddings = item_embeddings.unsqueeze(0)
    
    batch_size, k, embed_dim = item_embeddings.size()
    
    if k <= 1:
        return torch.tensor(0.0, device=item_embeddings.device)
        
    # L2 normalize embeddings to compute cosine similarity via dot product
    norm_emb = F.normalize(item_embeddings, p=2, dim=-1)
    
    # compute cosine similarity matrix: (batch_size, top_k, top_k)
    sim_matrix = torch.bmm(norm_emb, norm_emb.transpose(1, 2))
    
    # convert similarity to distance
    dist_matrix = 1.0 - sim_matrix
    
    # sum all distances in the matrix
    sum_dist = dist_matrix.sum(dim=(1, 2))
    
    # average over the k*(k-1) off-diagonal pairs
    ild_per_list = sum_dist / (k * (k - 1))
    
    return ild_per_list.mean()

class P2rSystem(LightningModule):
    def __init__(self, hparams, data):
        super(P2rSystem, self).__init__()
        
        self.save_hyperparameters(vars(hparams))
        self.data = data
        print(self.device)
        
        self.__build_dataset()
        self.__build_model()
        self.test_step_outputs = []
        
    def __build_model(self):
        self.p2r = P2r(self.hparams, self.data)
        self.p2r.to(self.device)
    
    def __build_dataset(self):
        random.seed(42)
        bridge_paper_indices = set(range(self.data['bridge_length'])) - set(self.data['positives'].keys())
        all_paper_indices = list(OrderedDict.fromkeys(list(bridge_paper_indices) + list(itertools.chain(*[self.data['paper_graph_adjlist'][idx] for idx in bridge_paper_indices]))).keys())
        for i in range(len(all_paper_indices) - 1, -1, -1):
            if all_paper_indices[i] in self.data['positives']:
                all_paper_indices.pop(i)
        train_paper_indices = all_paper_indices
        random.seed()
        
        self.p2r_train_dataset = P2rTrainDataset(self.hparams, self.data, train_paper_indices)
        self.p2r_test_dataset = P2rTestDataset(self.hparams, self.data, self.data['positives'])
        
        self.p2r_train_loader = DataLoader(dataset=self.p2r_train_dataset, 
                                           batch_size=self.hparams.batch_size, 
                                           shuffle=self.hparams.shuffle)
        self.p2r_test_loader = DataLoader(dataset=self.p2r_test_dataset, 
                                          batch_size=len(self.p2r_test_dataset), 
                                          shuffle=self.hparams.shuffle)

    def forward(self, paper_index, repo_index):
        return self.p2r(paper_index, repo_index)
    
    def loss(self, scores, neg_split, margin, constraint):
        return warpLoss(scores, neg_split, margin, self.device) * (1 + constraint.mean() / 2)

    def __one_step(self, batch, batch_nb):
        paper_index, repo_indices, neg_split = batch
        constraint, scores, ranks = self.forward(paper_index, repo_indices)
        loss = self.loss(scores, neg_split, self.hparams.warploss_margin, constraint)

        ild_metrics = []

        for k in [5, 10, 15, 20]:
            _, top_k_indices = torch.topk(scores, k=k, dim=-1)
            top_k_embeddings = self.p2r.repoModel(top_k_indices)
            if isinstance(top_k_embeddings, tuple):
                top_k_embeddings = top_k_embeddings[0]
            ild_val = calculate_ild(top_k_embeddings)
            ild_metrics.append(('ILD@%d' % k, ild_val))
        
        m_maps = [('MAP@%d' % k, torch.tensor(metricMAP(ranks, neg_split, k))) for k in [5, 10, 15, 20]]
        m_mrrs = [('MRR@%d' % k, torch.tensor(metricMRR(ranks, neg_split, k))) for k in [5, 10, 15, 20]]
        m_accs = [('ACC@%d' % k, torch.tensor(metricAccuracy(ranks, neg_split, k))) for k in [5, 10, 15, 20]]
        m_pmaps = [('PMAP@%d' % k, torch.tensor(metricPMAP(ranks, neg_split, k))) for k in [5, 10, 15, 20]]
        return loss, list(itertools.chain(*[m_maps, m_mrrs, m_accs, m_pmaps, ild_metrics]))
    
    def training_step(self, batch, batch_nb):
        loss_val, metrics = self.__one_step(batch, batch_nb)
        ret = OrderedDict([('loss', loss_val), ('progress', OrderedDict([('tng_loss', loss_val)] + metrics))])
        self.log('tng_loss', loss_val)
        return ret
    
    def test_step(self, batch, batch_nb):
        # run the standard metric logic from the paper
        loss, metrics = self.__one_step(batch, batch_nb)
        
        # get scores manually for Catalog Coverage
        paper_index, repo_indices, neg_split = batch
        _, scores, ranks = self.forward(paper_index, repo_indices)
        
        # get Top-10 recommendations for coverage
        _, top_k_indices = torch.topk(scores, k=10, dim=-1)
        
        output = {
            'test_loss': loss,
            'metrics': dict(metrics),
            'indices': top_k_indices.detach().cpu()
        }
        self.test_step_outputs.append(output)
        return output
    
    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return

        # aggregate Coverage
        all_indices = torch.cat([x['indices'] for x in self.test_step_outputs])
        unique_repos = torch.unique(all_indices)
        catalog_size = self.p2r.repoModel.num_embeddings
        coverage = len(unique_repos) / catalog_size
        
        avg_metrics = {}
        metric_keys = self.test_step_outputs[0]['metrics'].keys()
        
        for key in metric_keys:
            avg_val = torch.stack([x['metrics'][key] for x in self.test_step_outputs]).mean()
            avg_metrics[f'avg_{key}'] = avg_val
            self.log(f'avg_{key}', avg_val)

        self.log('test_catalog_coverage', torch.tensor(coverage))
        
        print(f"\n" + "="*30)
        print(f"BASELINE COMPARISON (WWW '20)")
        print(f"Catalog Coverage: {coverage * 100:.2f}%")
        print(f"MRR@10: {avg_metrics.get('avg_MRR@10', 0):.4f} (Target: ~0.460)")
        print(f"ILD@10: {avg_metrics.get('avg_ILD@10', 0):.4f}")
        print("="*30 + "\n")
        
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        # scheduler = scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return optimizer #], [scheduler]

    def train_dataloader(self):
        return self.p2r_train_loader
        
    def test_dataloader(self):
        return self.p2r_test_loader
    
    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])
        
        parser.set_defaults(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # network params
        parser.opt_list('--gcn_mid_dim', default=256, type=int, options=[128, 256, 512, 1024], tunable=True)
        parser.opt_list('--gcn_output_dim', default=256, type=int, options=[128, 256, 512, 1024], tunable=True)
        parser.opt_list('--txtcnn_drop_prob', default=0.0, options=[0.0, 0.1, 0.2], type=float, tunable=True)
        parser.opt_list('--gcn_drop_prob', default=0.5, options=[0.2, 0.5], type=float, tunable=True)
        parser.opt_list('--warploss_margin', default=0.4, type=float, tunable=True)
        parser.opt_list('--freeze_embeddings', default=True, options=[True, False], 
                        type=lambda x: (str(x).lower() == 'true'), tunable=True)
        
        parser.opt_list('--txtcnn_pfilter_num1', default=64, options=[16, 32, 64, 128], type=int, tunable=True)
        parser.opt_list('--txtcnn_pfilter_num2', default=64, options=[16, 32, 64, 128], type=int, tunable=True)
        parser.opt_list('--txtcnn_pfilter_num3', default=64, options=[16, 32, 64, 128], type=int, tunable=True)
        parser.opt_list('--txtcnn_pfilter_num4', default=64, options=[16, 32, 64, 128], type=int, tunable=True)
        parser.opt_list('--txtcnn_rfilter_num1', default=64, options=[16, 32, 64, 128], type=int, tunable=True)
        parser.opt_list('--txtcnn_rfilter_num2', default=32, options=[16, 32, 64, 128], type=int, tunable=True)

        # data
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'data'), type=str)
        parser.add_argument('--top_t', default=6, type=int)
        parser.add_argument('--total_onehop', default=20, type=int)
        parser.add_argument('--total', default=50, type=int)
        parser.add_argument('--shuffle', default=True, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument('--train_div', default=1.0, type=float)

        # training params (opt)
        parser.opt_list('--batch_size', default=64, options=[32, 64, 128, 256], type=int, tunable=False)
        parser.opt_list('--max_nb_epochs', default=8, options=[256, 512, 1024], type=int, tunable=False)
        parser.opt_list('--learning_rate', default=0.0005, options=[0.0001, 0.0005, 0.001], type=float, tunable=True)
        parser.opt_list('--weight_decay', default=0.001, options=[0.0001, 0.0005, 0.001], type=float, tunable=True)
        parser.add_argument('--model_save_path', default=os.path.join(root_dir, 'experiment'), type=str)
        return parser
    
