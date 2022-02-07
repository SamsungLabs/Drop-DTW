import os
import torch
import argparse
import random
import torch
import numpy as np
import pytorch_lightning as pl
import torchmetrics
from copy import deepcopy, copy


from paths import  PROJECT_PATH, WEIGHTS_PATH
from models.nets import EmbeddingsMapping
from models.losses import compute_clust_loss, compute_alignment_loss
from models.visualization import visualize_drop_dtw_matching, visualize_step_strength
from data.data_module import DataModule
from data.data_utils import sample_to_device
from data.batching import unflatten_batch
from evaluate import compute_all_metrics
from utils import Namespace, load_yaml


device = "cuda" if torch.cuda.is_available() else "cpu"

# Enabling reproducibility
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help="name of the experiment")
parser.add_argument('--dataset', type=str, default='COIN', choices=['COIN', 'CrossTask', 'YouCook2'], help="name of the dataset we are encoding")

# training hyper-parameters
parser.add_argument('--batch_size', type=int, default=24, help="batch size")
parser.add_argument('--epochs', type=int, default=10, help="batch size")
parser.add_argument('--lr', type=float, default=3e-4, help="learning rate")
parser.add_argument('--wd', type=float, default=1e-4, help="weight decay")
parser.add_argument('--n_cls', type=int, default=3, help="Number of video of one class in a batch. Must divide batch_size")

# model hyper-parameters
parser.add_argument('--video_layers', type=int, default=2, help="Number of layers in nonlinear mapping for video embeddings")
parser.add_argument('--text_layers', type=int, default=0, help="Number of layers in nonlinear mapping for text embeddings")
parser.add_argument('--batchnorm', type=int, default=0, help="Wheather to use batchnorm in models")
parser.add_argument('--pretrained_drop', action='store_true', default=False, help='Start with pre-trained drop costs')

# loss hyper-parameters
parser.add_argument('--dp_algo', type=str, default='DropDTW', choices=['DropDTW', 'OTAM', 'NW', 'DTW'], help="DP algo used for matching")
parser.add_argument('--drop_cost', type=str, default='logit', choices=['logit', 'learn'], help="The way to define drop cost")
parser.add_argument('--dtw_softning', type=str, default='prob', choices=['prob', 'gamma', 'none'], help="DP algo used for matching")
parser.add_argument('--keep_percentile', type=float, default=0.3, help="If drop cost is defined as logit, computes the percentile of drops")
parser.add_argument('--contiguous_drop', type=bool, default=True, help="Wheather to do contiguous drop in Drop-DTW")
parser.add_argument('--clust_loss_mult', type=float, default=4, help="Multiplier for the step loss")
parser.add_argument('--dtw_loss_mult', type=float, default=2.5, help="Multiplier for the dtw loss")
parser.add_argument('--dtw_xz_gamma', type=float, default=10, help="Softmax temperature for xz product, in dtw")
parser.add_argument('--dtw_min_gamma', type=float, default=1, help="Softmax temperature for softmin, in dtw")
parser.add_argument('--step_xz_gamma', type=float, default=30, help="Softmax temperature for xz product, in step loss")
parser.add_argument('--bg_scope', type=str, default='global', choices=['global', 'class', 'video'], help="The scope where the background prototype is conisdered the same")
args = parser.parse_args()


class VisualizationCallback(pl.callbacks.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, flat_batch, batch_idx, dataloader_idx):
        step = trainer.global_step
        if step % 10 == 0:
            original_sample = sample_to_device(random.choice(unflatten_batch(flat_batch)), device)
            # sample = deepcopy(original_sample)
            sample = copy(original_sample)
            sample['frame_features'] = pl_module.model.map_video(sample['frame_features'].to(device)).detach()
            sample['step_features'] = pl_module.model.map_text(sample['step_features'].to(device)).detach()
            if args.drop_cost == 'learn':
                distractor = pl_module.model.compute_distractors(sample['step_features'].mean(0)).detach().cpu()
            else:
                distractor = None

            sample_gammas = (args.dtw_xz_gamma, 1)
            sample_dict = {'Ours': sample_to_device(sample, 'cpu'),
                           'HowTo100M': sample_to_device(original_sample, 'cpu')}

            dtw_image = visualize_drop_dtw_matching(
                sample_dict, distractor, gamma_f=sample_gammas,
                drop_cost=args.drop_cost, keep_percentile=args.keep_percentile, shape=(10, 2))
            steps_image = visualize_step_strength(
                sample_dict, distractor, gamma_f=sample_gammas,
                drop_cost=args.drop_cost, keep_percentile=args.keep_percentile, shape=(10, 2))
            matching_picture = np.concatenate([steps_image, dtw_image], 1)
            trainer.logger.experiment.add_image(
                'matching_picture', matching_picture.transpose((2, 0, 1)), global_step=step)


class TrainModule(pl.LightningModule):
    def __init__(self, model, data, name=None):
        super(TrainModule, self).__init__()
        self.name = name
        self.model = model
        self.data = data
        self.avg_loss_metric = torchmetrics.AverageMeter()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)
        return optimizer

    def training_step(self, flat_batch, batch_id):
        flat_batch['frame_features'] = self.model.map_video(flat_batch['frame_features'])
        flat_batch['step_features'] = self.model.map_text(flat_batch['step_features'])
        samples = unflatten_batch(flat_batch)

        if args.drop_cost == 'learn':
            mean_steps = torch.stack([s['step_features'].mean(0) for s in samples], 0)
            distractors = self.model.compute_distractors(mean_steps)
        else:
            distractors = None

        # Computing total loss
        total_loss = 0
        if args.clust_loss_mult > 0:
            clust_loss = compute_clust_loss(samples, distractors, xz_hard_ratio=1,
                                          xz_gamma=args.step_xz_gamma, frame_gamma=10,
                                          all_classes_distinct=(args.dataset == 'YouCook2'),
                                          bg_scope=args.bg_scope)
            self.log('train/clust_loss', clust_loss)
            total_loss += args.clust_loss_mult * clust_loss

        if args.dtw_loss_mult > 0:
            dtw_loss = args.dtw_loss_mult * compute_alignment_loss(
                samples, distractors, contiguous=args.contiguous_drop,
                gamma_xz=args.dtw_xz_gamma, gamma_min=args.dtw_min_gamma,
                drop_cost_type=args.drop_cost, dp_algo=args.dp_algo,
                keep_percentile=args.keep_percentile, softning=args.dtw_softning)
            self.log('train/dtw_loss', dtw_loss)
            total_loss += dtw_loss

        self.log('train/total_loss', self.avg_loss_metric(total_loss))
        return total_loss

    def training_epoch_end(self, training_step_outputs):
        self.model.eval()
        avg_total_loss = self.avg_loss_metric.compute()
        print('Train Total loss: {:.2f}'.format(avg_total_loss))
        self.avg_loss_metric.reset()

        eval_config = Namespace(dp_algo='DropDTW', drop_cost=args.drop_cost, keep_percentile=0.3,
                                use_unlabeled=True, distance='inner', dataset=args.dataset)
        _, _, accuracy_dtw, iou_dtw, recall = compute_all_metrics(
            self.data.val_dataset, self.model, gamma=30, config=eval_config)
        
        print(f"Recall is {recall:.1f}%")
        print(f"DTW Accuracy is {accuracy_dtw:.1f}%")
        print(f"DTW IoU is {iou_dtw:.1f}%")
        self.log("Metrics/Recall", recall)
        self.log("Metrics/Accuracy", accuracy_dtw)
        self.log("Metrics/IoU", iou_dtw)


def main():
    data = DataModule(args.dataset, args.n_cls, args.batch_size)
    model = EmbeddingsMapping(
        d=512, learnable_drop=(args.drop_cost == 'learn'), video_layers=args.video_layers,
        text_layers=args.text_layers, normalization_dataset=data.train_dataset,
        batchnorm=args.batchnorm)

    # load drop costs from a pre-trained model
    if args.pretrained_drop:
        # assumes that the model with the same name has been already trained
        # this retraines the model, but uses drop_mapping intialization from the previous training
        from glob import glob
        weights_path = glob(os.path.join(WEIGHTS_PATH, args.name, f"weights-epoch=*.ckpt"))[0]
        state_dict = {k[6:]: v for k, v in torch.load(weights_path, map_location=device)['state_dict'].items()
                      if k.startswith('model.drop_mapping')}
        model.load_state_dict(state_dict, strict=False)
        
    train_module = TrainModule(model, data)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='Metrics/Recall',
        dirpath=os.path.join(PROJECT_PATH, 'weights', args.name),
        filename='weights-{epoch:02d}',
        save_top_k=1,
        mode='max',
    )
    vis_callback = VisualizationCallback()
    logger = pl.loggers.TensorBoardLogger('tb_logs', args.name)

    trainer = pl.Trainer(gpus=1, callbacks=[checkpoint_callback, vis_callback],
                         max_epochs=args.epochs, logger=logger)

    trainer.fit(train_module, data)


if __name__ == '__main__':
    main()
