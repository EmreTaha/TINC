import time
import pandas as pd
import torch
import os

from os import path

from torch.optim import lr_scheduler
from torch import nn
import torch.nn.functional as F


from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from model import Models
    
from utils import VICReg_augmentaions
from utils import OPTIMIZERS
from utils import check_existing_model
from utils import args_parser
from utils import df_to_dataset, ScanDataset, VicRegTempDataset, ss_data_paths
from utils import TINCLoss
from utils import convert_model
from utils import initialize

initialize()

args = args_parser()

def train(args):
    # Define parameters
    ssl_data_root = args.ssl_data_dir
    os.makedirs(args.save_dir, exist_ok=True) 
    dl_kwargs = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 2} # Shuffle needs to be false, because we are using randomsampler
    min_max = (args.min_t,args.max_t)
    optim_params = {'lr':args.lr,
                    'weight_decay':args.wd,
                    'exclude_bias_and_norm':args.exclude_nb}
    train_params = {'num_epochs': args.max_iters, 'warmup_epchs': args.warmup_iters, 'eta_min':1e-7}
    eval_params  = {'lr':1e-4, 'num_epochs': args.lin_iters}


    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

    resnet = Models[args.model](zero_init_residual=True).model # encoding backbone
    repre_dim = resnet.fc.in_features

    model = Models['VICReg'](resnet, projector_hidden = (4096,4096,4096)).to(device) # contrastive model -> backbone + projector

    if torch.cuda.device_count() > 1:
        model = convert_model(model)
        model = nn.DataParallel(model)
    model.cuda()
    
    # Create datasets and dataloaders
    NORM = []

    # contrastive training augmentations
    train_transf = VICReg_augmentaions(image_size=224, normalize=NORM, temporal=True)
    # supervised training augmentations
    train_eval_transf = transforms.Compose([transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
                                        transforms.RandomAffine(0,(0.05,0.05),fill=0),
                                        transforms.RandomRotation(degrees=15),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ConvertImageDtype(torch.float32)])
                                        #transforms.Normalize(*NORM)])
    
    test_transf = transforms.Compose([transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
                                  transforms.ConvertImageDtype(torch.float32)])#, transforms.Normalize(*NORM)])
            
    ssl_paths, _ = ss_data_paths(ssl_data_root)

    ds_ssl = VicRegTempDataset(ssl_paths,train_transf,absol=True, norm_label=args.norm_label, min_max=min_max)

    df_train = pd.read_csv(args.data_dir+'/df_train.csv') #Add from labeled data as well
    ds_tr = VicRegTempDataset(df_train[df_train['Filepath'].str.contains('central_64.png')]['Filepath'].tolist(),train_transf,absol=True, norm_label=args.norm_label, min_max=min_max) #Only use central Bscans from the center

    ssl_ds = torch.utils.data.ConcatDataset([ds_ssl, ds_tr]) # concatenate unlabeled and labeled training datasets for pretraining

    sampler = torch.utils.data.RandomSampler(ssl_ds, replacement=True, num_samples=len(ssl_ds)*75) # Since the parts are created randomly on the fly, oversampling in an epoch is necessary. Each patient is oversampled 75 times

    ssl_dl = torch.utils.data.DataLoader(ssl_ds, sampler=sampler, drop_last=True, **dl_kwargs)

    df_train = df_train.drop(df_train[df_train['6m_label'] == 0].sample(frac=.5).index) # Drop half of the negative images to decrease class imbalance

    df_val = pd.read_csv(args.data_dir+'/df_val.csv')
    # For Per Volume evaluation
    val_paths = df_val['Filepath'].tolist()
    val_paths_ = [os.path.split(x)[0] for x in val_paths]
    val_labels = df_val['6m_label'].tolist()
    val_scan = list(set(list(zip(val_paths_,val_labels))))
    val_scan_paths,val_scan_labels = map(list, zip(*val_scan))
    df_val = df_val[df_val['Filepath'].str.contains("_64")]

    df_test = pd.read_csv(args.data_dir+'/df_test.csv')
    # For Per Volume evaluation
    test_paths = df_test['Filepath'].tolist()
    test_paths_ = [os.path.split(x)[0] for x in test_paths]
    test_labels = df_test['6m_label'].tolist()
    test_scan = list(set(list(zip(test_paths_,test_labels))))
    test_scan_paths,test_scan_labels = map(list, zip(*test_scan))
    df_test = df_test[df_test['Filepath'].str.contains("_64")]

    ds_train = df_to_dataset(df_train,'Filepath','6m_label', train_eval_transf)
    ds_val = df_to_dataset(df_val,'Filepath','6m_label', test_transf)
    ds_test = df_to_dataset(df_test,'Filepath','6m_label', test_transf)
    ds_val_scan = ScanDataset(val_scan_paths,val_scan_labels,test_transf,volume=False)
    ds_test_scan = ScanDataset(test_scan_paths,test_scan_labels,test_transf,volume=False)

    trainloader = torch.utils.data.DataLoader(ds_train, batch_size=dl_kwargs['batch_size'],
                                            shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(ds_val, batch_size=8,
                                            shuffle=False, num_workers=2)

    testloader = torch.utils.data.DataLoader(ds_test, batch_size=1,
                                            shuffle=False, num_workers=2)

    val_scan_loader = torch.utils.data.DataLoader(ds_val_scan, batch_size=1,
                                         shuffle=False, num_workers=2, drop_last=True)

    test_scan_loader = torch.utils.data.DataLoader(ds_test_scan, batch_size=1,
                                         shuffle=False, num_workers=2, drop_last=True)

    # Define optimizer and scheduler
    if args.exclude_nb:
        # Exclude norms and biases
        no_decay = ["bias", "LayerNorm.weight", "BatchNorm.weight"]
        model_params = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": optim_params['weight_decay'],
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
    else:
        model_params = model.parameters()

    optimizer_f = OPTIMIZERS[args.optim]
    optimizer = optimizer_f(model_params, **optim_params)
    
    # Warm-up learning rate scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer, lambda it : (it+1)/(train_params['warmup_epchs']*len(ssl_dl)))

    # Define loss
    criterion = TINCLoss(15,25,1,1,insensitive=args.insd)

    # Check for existing training
    lp_acc = []
    loss_hist = []
    lr_hist = []

    epoch_start, saved_data = check_existing_model(args.save_dir, device)

    if saved_data:
        # Extract data
        model.load_state_dict(saved_data['model'])
        optimizer.load_state_dict(saved_data['optim'])
        if epoch_start >= train_params['warmup_epchs']:
            iters_left = (train_params['num_epochs']-train_params['warmup_epchs'])*len(ssl_dl)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, iters_left,
                                                        eta_min=train_params['eta_min'],
                                                        last_epoch=epoch_start*len(ssl_dl))
        scheduler.load_state_dict(saved_data['sched'])
        lp_acc = saved_data['lp_acc']
        loss_hist = saved_data['loss_hist']
        lr_hist = saved_data['lr_hist']

    # If no training before, train a linear classifier
    if len(lp_acc)==0:
        if torch.cuda.device_count() > 1:
            linear_proto = Models['Linear'](model.module.backbone_net, num_classes=1, out_dim=repre_dim, device=device)
        else:
            linear_proto = Models['Linear'](model.backbone_net, num_classes=1, out_dim=repre_dim, device=device)
        linear_proto.train(trainloader, eval_params['num_epochs'], eval_params['lr'])
        lp_acc.append(linear_proto.get_metrics(valloader, val_scan_loader))
        del linear_proto

    # get total number of iterations
    total_iters = train_params['num_epochs'] * len(ssl_dl)
    print(total_iters)

    # Run Training
    for epoch in range(epoch_start, train_params['num_epochs']):
        epoch_loss = 0
        sim_loss = 0
        std_loss = 0
        cov_loss = 0
        model.train()
        start_time = time.time()
        for (x1,x2), diff, _ in ssl_dl:
            x1,x2,diff = x1.to(device), x2.to(device), diff.to(device)
            diff = diff.reshape(-1,1)
            
            # Forward pass
            z1, z2 = model(x1,x2)

            loss, ind_loss = criterion(z1,z2,diff) 

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            if args.grad_norm_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(),2.0, error_if_nonfinite=True)
            optimizer.step()
            
            # Scheduler every iteration for cosine deday
            scheduler.step()
            
            # Save loss and LR
            epoch_loss += loss.item()   
            sim_loss += ind_loss[0].item()
            std_loss += ind_loss[1].item()
            cov_loss += ind_loss[2].item()
            lr_hist.extend(scheduler.get_last_lr())
        
        # Switch to Cosine Decay after warmup period
        if epoch+1==train_params['warmup_epchs']:
            iters_left = (train_params['num_epochs']-train_params['warmup_epchs'])*len(ssl_dl)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                        iters_left,
                                                        eta_min=train_params['eta_min'])
        
        # Log
        loss_hist.append(epoch_loss/len(ssl_dl))
        print(f'Epoch: {epoch}, Loss: {loss_hist[-1]}, Sim loss: {sim_loss/len(ssl_dl)}, Std loss: {std_loss/len(ssl_dl)}, Cov loss: {cov_loss/len(ssl_dl)}, Time epoch: {time.time() - start_time}')
        
        # Run linear protocol and save stats
        if (epoch+1)%20==0:
            # Linear protocol
            if torch.cuda.device_count() > 1:
                linear_proto = Models['Linear'](model.module.backbone_net, num_classes=1, out_dim=repre_dim, device=device)
            else:
                linear_proto = Models['Linear'](model.backbone_net, num_classes=1, out_dim=repre_dim, device=device)
            linear_proto.train(trainloader, eval_params['num_epochs'], eval_params['lr'])

            if (epoch+1)==train_params['num_epochs']:
            # If this is the end of training, use the test dataset
                lp_acc.append(linear_proto.get_metrics(testloader, test_scan_loader))
            else:
                lp_acc.append(linear_proto.get_metrics(valloader, val_scan_loader))
            del linear_proto
            
            torch.save({'model':model.state_dict(),
                        'optim': optimizer.state_dict(),
                        'sched': scheduler.state_dict(),
                        'lp_acc': lp_acc,
                        'loss_hist': loss_hist,
                        'lr_hist': lr_hist,
                        'args': args}, 
                    path.join(args.save_dir, f'epoch_{epoch+1:03}.tar'))


def main():
    train(args)

if __name__ == "__main__":
    main()