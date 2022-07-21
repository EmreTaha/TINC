from copy import deepcopy
from typing import Optional

from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np

from utils import roc_aucs,pr_aucs,f1_score, delong_auc_var_conf

class Linear_Protocoler(object):
    def __init__(self, backbone_net, num_classes: int = 10, out_dim: Optional[int] = None, device : str = 'cpu', finetune=False):
        self.device = device
        self.num_classes = num_classes
        # Copy net
        self.backbone = deepcopy(backbone_net)
        self.finetune = finetune
        if not self.finetune:
            # Turn off gradients
            # Update: this option is turned off due to some problems
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        if finetune:
            self.backbone.train()
        # get out dimension
        if out_dim:
            out_dim = out_dim
        else:
            out_dim = p.shape[0]
        # Add classification layer
        layers = []
        if self.num_classes==1 or self.num_classes==2:
            layers.append(nn.Linear(out_dim, 1))
        else:
            layers.append(nn.Linear(out_dim, self.num_classes))

        self.classifier = torch.nn.Sequential(*layers)
        self.classifier[0].weight.data.normal_(mean=0.0, std=0.01)
        self.classifier[0].bias.data.zero_()

        # Send to device
        self.backbone = self.backbone.to(self.device)
        self.classifier = self.classifier.to(self.device)
    
    def train(self, dataloader, num_epochs, lr : float = 1e-3, schedule : bool = False, class_weights = 5.0):
        # Define optimizer
        if self.finetune:
            params = list(self.classifier.parameters()) + list(self.backbone.parameters())
            self.backbone.train()
        else:
            params = list(self.classifier.parameters())
            self.backbone.eval()

        optimizer = opt.Adam(params, lr)
        # Define loss
        if self.num_classes==1 or self.num_classes==2:
            ce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(class_weights))
        else:
            ce_loss = nn.CrossEntropyLoss()
        # Define scheduler
        if schedule:
            scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        else:
            scheduler = None
        
        # Train
        self.classifier.train()          
        
        for epoch in range(num_epochs):
            for x,y in dataloader:
                x,y = x.to(self.device), y.to(self.device)
                y = y.reshape(-1,1)
                # forward
                representaions = self.backbone(x)
                loss = ce_loss(self.classifier(representaions), y.float())
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if scheduler:
                scheduler.step()

    def get_metrics(self, dataloader, volume_dataloader):

        correct = 0
        total = 0
        preds = []
        labels = []

        scan_correct = 0
        scan_total = 0
        scan_preds = []
        scan_labels = []
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            self.classifier.eval()
            self.backbone.eval()
            
            # B-scan level prediction
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y = y.reshape(-1,1)
                # calculate outputs by running images through the network
                representaions = self.backbone(x)
                outputs = self.classifier(representaions)

                labels += y.reshape(-1).cpu()
                if self.num_classes==1 or self.num_classes==2:
                    preds += torch.sigmoid(outputs).data.reshape(-1).cpu()
                else:
                    preds += torch.softmax(outputs, dim=1).data.reshape(-1).cpu()

                probs = torch.sigmoid(outputs).data
                predicted = probs.round()
                total += y.size(0)
                correct += (predicted == y).sum().item()

            # Volume level prediction
            for x, y in volume_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                x = torch.reshape(x,(x.shape[1],1,x.shape[2],x.shape[3]))
                y = y.reshape(-1,1)
                # calculate outputs by running images through the network
                representaions = self.backbone(x)
                outputs = self.classifier(representaions)

                scan_labels += y.reshape(-1).cpu()
                if self.num_classes==1 or self.num_classes==2:
                    scan_preds += torch.sigmoid(outputs).max().data.reshape(-1).cpu()
                else:
                    scan_preds += torch.softmax(outputs, dim=1).max().data.reshape(-1).cpu()

                probs = torch.sigmoid(outputs).data
                predicted = probs.round()
                scan_total += y.size(0)
                scan_correct += (predicted == y).sum().item()

            self.classifier.train()
        
        f1 = f1_score(labels,np.around(preds))
        prauc = pr_aucs(labels,preds)
        rocauc = roc_aucs(labels,preds)

        scan_f1 = f1_score(scan_labels,scan_preds)
        scan_prauc = pr_aucs(scan_labels,scan_preds)
        scan_rocauc = roc_aucs(scan_labels,scan_preds)
        _, _, scan_delong_ci = delong_auc_var_conf(np.array(scan_labels), np.array(scan_preds))


        print(  f"classification: roc-auc: {rocauc:.3f}.. "
                f"classification: pr-auc: {prauc:.3f}.. "
                f"classification: F1: {f1:.3f}.. "
                f"classification: accuracy: {correct / total:.3f}.."
                f"classification: scan roc-auc: {scan_rocauc:.3f}.. "
                f"Val delong roc-auc-ci: {['{:.3f}'.format(x) for x in scan_delong_ci]}.. "
                f"classification: scan pr-auc: {scan_prauc:.3f}.. "
                f"classification: scan F1: {scan_f1:.3f}.. "
                f"classification: scan accuracy: {scan_correct / scan_total:.3f}")
        return correct / total, f1, prauc, rocauc, scan_correct / scan_total, scan_f1, scan_prauc, scan_rocauc, scan_delong_ci