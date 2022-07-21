
import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default="./data/supervised_data",
        help='data path to the slices')
    
    parser.add_argument('--ssl_data_dir', type=str, default="./data/unsupervised_data")

    parser.add_argument('--optim', type=str, default="Adam",
        help='data path to the slices')

    parser.add_argument('--save_dir', type=str, default="./saved_models",
        help='model and details save path')
    
    parser.add_argument('--max_iters', type=int, default=150,
        help='max. num of epochs for training')
    
    parser.add_argument('--lin_iters', type=int, default=5,
        help='max. num of epochs for linear classifier')

    parser.add_argument('--lin_lr_sch', default=False, action='store_true', required=False,
        help='Whether to use learning rate scheduler during linear evaluation')
    
    parser.add_argument('--warmup_iters', type=int, default=10,
        help='warmup epochs for training')

    parser.add_argument('--min_t', type=int, default=60,
        help='Minimun time difference between two scans pair of a patient')

    parser.add_argument('--max_t', type=int, default=60,
        help='Maximum time difference between two scans pair of a patient')

    parser.add_argument('--in_ch', type=int, default=1,
        help='number of input channels')

    parser.add_argument('--batch_size', type=int, default=32,
        help='batch size')

    parser.add_argument('--n_cl', type=int, default=4,
        help='number of classes, should be arranged accordingly with binning')

    parser.add_argument('--n_sample', type=int, default=50,
        help='number of samples for raytune')
    
    parser.add_argument('--device',  default= torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
        help="To use cuda, set to a specific GPU ID.")

    parser.add_argument('--model', type=str, default='GrayResNet',
        help='imports the given backbone model for the approach')

    parser.add_argument('--norm_label', default=False, action='store_true', required=False,
        help='Normalize temporal label for the SSL')

    parser.add_argument('--norm_embed', default=False, action='store_true', required=False,
        help='L2-Normalize embeddings from the SSL')

    parser.add_argument('--insd', type=str, default='one',
        help='degree of margin insensitive in our loss')

    parser.add_argument('--lr', type=float, default=3e-4,
        help='Main learning rate')

    parser.add_argument('--wd', type=float, default=1e-5,
        help='Weight decay')

    parser.add_argument('--grad_norm_clip', default=False, action='store_true', required=False,
        help='Enable gradient norm clipping')

    parser.add_argument('--exclude_nb', default=False, action='store_true', required=False,
        help='Exclude norm and biases')

    args = parser.parse_args()
    return args