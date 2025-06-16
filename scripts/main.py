import argparse
import ast
import torch
from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from scripts.grid_search.grid_searcher import GridSearcher
from scripts.baseline.baseline_tester import BaselineTester
from model_utils.models.siamese import SiameseCLIPModelPairs, SiameseCLIPTriplet

def main():
    parser = argparse.ArgumentParser(description='CLIP-based text similarity training and evaluation')
    parser.add_argument('--mode', type=str, choices=['train', 'grid_search', 'baseline'], required=True,
                      help='Mode to run: train, grid_search, or baseline')
    parser.add_argument('--reference_filepath', type=str, required=True,
                      help='Path to reference data')
    parser.add_argument('--test_reference_filepath', type=str, required=True,
                      help='Path to test reference data')
    parser.add_argument('--test_filepath', type=str, required=True,
                      help='Path to test data')
    parser.add_argument('--model_type', type=str, choices=['pair', 'triplet'], default='pair',
                      help='Model type: pair or triplet')
    parser.add_argument('--loss_type', type=str, choices=['cosine', 'euclidean', 'hybrid'], default='cosine',
                      help='Loss function type')
    parser.add_argument('--warmup_filepath', type=str,
                      help='Path to warmup data (optional)')
    parser.add_argument('--lrs', type=str, default='[1e-4]',
                      help='Learning rates to try (for grid search)')
    parser.add_argument('--batch_sizes', type=str, default='[32]',
                      help='Batch sizes to try (for grid search)')
    parser.add_argument('--margins', type=str, default='[0.5]',
                      help='Margins to try (for grid search)')
    parser.add_argument('--internal_layer_sizes', type=str, default='[128]',
                      help='Internal layer sizes to try (for grid search)')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                      help='Number of warmup epochs')
    parser.add_argument('--log_dir', type=str, default='results',
                      help='Directory to save results')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'baseline':
        # Test raw CLIP performance
        tester = BaselineTester()
        results_df, metrics = tester.test(args.test_reference_filepath, args.test_filepath)
        print("\nBaseline Results:")
        print(metrics)

    elif args.mode == 'train':
        # Single training run
        model_class = SiameseCLIPModelPairs if args.model_type == 'pair' else SiameseCLIPTriplet
        model = model_class(embedding_dim=512, projection_dim=128).to(device)
        
        # Get appropriate loss class
        if args.model_type == 'pair':
            if args.loss_type == 'cosine':
                from model_utils.loss.pair_losses import CosineLoss
                criterion = CosineLoss(margin=0.5)
            else:
                from model_utils.loss.pair_losses import EuclideanLoss
                criterion = EuclideanLoss(margin=1.0)
        else:  # triplet
            if args.loss_type == 'cosine':
                from model_utils.loss.triplet_losses import CosineTripletLoss
                criterion = CosineTripletLoss(margin=0.1)
            elif args.loss_type == 'euclidean':
                from model_utils.loss.triplet_losses import EuclideanTripletLoss
                criterion = EuclideanTripletLoss(margin=1.0)
            else:  # hybrid
                from model_utils.loss.triplet_losses import HybridTripletLoss
                criterion = HybridTripletLoss(margin=1.0, alpha=0.5)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        trainer = Trainer(model, criterion, optimizer, device)
        trainer.train(
            dataloader=None,  # You'll need to create appropriate dataloader
            test_reference_filepath=args.test_reference_filepath,
            test_filepath=args.test_filepath,
            mode=args.model_type,
            epochs=args.epochs,
            warmup_loader=None,  # You'll need to create appropriate dataloader
            warmup_epochs=args.warmup_epochs
        )

    elif args.mode == 'grid_search':
        # Grid search
        model_class = SiameseCLIPModelPairs if args.model_type == 'pair' else SiameseCLIPTriplet
        searcher = GridSearcher(model_class, device, log_dir=args.log_dir)
        
        lrs = ast.literal_eval(args.lrs)
        batch_sizes = ast.literal_eval(args.batch_sizes)
        margins = ast.literal_eval(args.margins)
        internal_layer_sizes = ast.literal_eval(args.internal_layer_sizes)
        
        best_config, results_df = searcher.search(
            reference_filepath=args.reference_filepath,
            test_reference_filepath=args.test_reference_filepath,
            test_filepath=args.test_filepath,
            lrs=lrs,
            batch_sizes=batch_sizes,
            margins=margins,
            internal_layer_sizes=internal_layer_sizes,
            mode=args.model_type,
            loss_type=args.loss_type,
            warmup_filepath=args.warmup_filepath,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs
        )
        
        print("\nGrid Search Results:")
        print(f"Best configuration: {best_config}")
        print("\nAll results saved to:", args.log_dir)

if __name__ == '__main__':
    main() 