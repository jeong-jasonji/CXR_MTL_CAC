import copy
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, classification_report
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm


def train_epochs(opt, model, dataloaders, criterion, optimizer):
    """Runs the model for the number of epochs given and keeps track of best weights and metrics"""

    since = time.time()

    best_states = {'model_wts': copy.deepcopy(model.state_dict()), 'avg_acc': 0.0, 'avg_precision': 0.0, 'avg_recall': 0.0, 'avg_f1': 0.0, 'avg_loss': 999.0,
                   'last_improvement_epoch': 0}
    histories = {'train_acc': [], 'train_pre': [], 'train_rec': [], 'train_f1': [], 'train_loss': [], 'val_acc': [], 'val_pre': [], 'val_rec': [], 'val_f1': [],'val_loss': [], }

    for epoch in tqdm(range(opt.max_epochs), desc='Training'):
        print('Epoch {}/{}'.format(epoch + 1, opt.max_epochs), file=opt.log)

        model, epoch_metrics = train_epoch(opt, model, dataloaders, criterion, optimizer)

        time_elapsed = time.time() - since
        print('Epoch {} complete: {:.0f}m {:.0f}s'.format(epoch + 1, time_elapsed // 60, time_elapsed % 60),
              file=opt.log)

        # deep copy the model if the accuracy and auc improves
        if opt.optim_metric != 'avg_loss' and epoch_metrics['val'][opt.optim_metric] > best_states[opt.optim_metric]:
            best_states['avg_loss'] = epoch_metrics['val']['avg_loss']
            best_states['avg_acc'] = epoch_metrics['val']['avg_acc']
            best_states['avg_precision'] = epoch_metrics['val']['avg_precision']
            best_states['avg_recall'] = epoch_metrics['val']['avg_recall']
            best_states['avg_f1'] = epoch_metrics['val']['avg_f1']
            best_states['last_improvement_epoch'] = epoch
            best_states['model_wts'] = copy.deepcopy(model.state_dict())
            print('Improved {}, Updated weights \n'.format(opt.optim_metric), file=opt.log)
        elif opt.optim_metric == 'avg_loss' and epoch_metrics['val'][opt.optim_metric] < best_states[opt.optim_metric]:
            best_states['avg_loss'] = epoch_metrics['val']['avg_loss']
            best_states['avg_acc'] = epoch_metrics['val']['avg_acc']
            best_states['avg_precision'] = epoch_metrics['val']['avg_precision']
            best_states['avg_recall'] = epoch_metrics['val']['avg_recall']
            best_states['avg_f1'] = epoch_metrics['val']['avg_f1']
            best_states['last_improvement_epoch'] = epoch
            best_states['model_wts'] = copy.deepcopy(model.state_dict())
            print('Improved {}, Updated weights \n'.format(opt.optim_metric), file=opt.log)
        opt.log.flush()
        # save histories
        histories['train_acc'].append(epoch_metrics['train']['avg_acc'])
        histories['train_pre'].append(epoch_metrics['train']['avg_precision'])
        histories['train_rec'].append(epoch_metrics['train']['avg_recall'])
        histories['train_f1'].append(epoch_metrics['train']['avg_f1'])
        histories['train_loss'].append(epoch_metrics['train']['avg_loss'])
        histories['val_acc'].append(epoch_metrics['val']['avg_acc'])
        histories['val_pre'].append(epoch_metrics['train']['avg_precision'])
        histories['val_rec'].append(epoch_metrics['train']['avg_recall'])
        histories['val_f1'].append(epoch_metrics['train']['avg_f1'])
        histories['val_loss'].append(epoch_metrics['val']['avg_loss'])

        if opt.save_every_epoch:
            model.load_state_dict(best_states['model_wts'])
            torch.save(model, opt.save_dir + '{}.pth'.format(opt.model_name))
            pickle.dump(histories, open(opt.save_dir + "{}_history.pkl".format(opt.model_name), "wb"))

            # save plots for the best ones
            for k in ['acc', 'loss']:
                train_hist = [h for h in histories['train_{}'.format(k)]]
                val_hist = [h for h in histories['val_{}'.format(k)]]
                plt.xlabel("Training Epochs")
                plt.title("{} over Epochs".format('Metrics' if k == 'acc' else 'Loss'))
                plt.ylabel("{}".format('Metrics' if k == 'acc' else 'Loss'))
                if k == 'loss':
                    plt.plot(range(1, epoch + 2), train_hist)
                    plt.plot(range(1, epoch + 2), val_hist)
                    plt.legend(['Train', 'Validation'])
                else:
                    legends = []
                    for f in ['acc', 'pre', 'rec', 'f1']:
                        train_hist = [h for h in histories['train_{}'.format(f)]]
                        val_hist = [h for h in histories['val_{}'.format(f)]]
                        plt.plot(range(1, epoch + 2), train_hist)
                        plt.plot(range(1, epoch + 2), val_hist)
                        legends.append('train_{}'.format(f))
                        legends.append('val_{}'.format(f))
                    plt.legend(legends)
                plt.ylim((0, 1.)) if k == 'acc' else None
                plt.xticks(np.arange(1, epoch + 2, 1.0))
                plt.savefig(opt.save_dir + '{}_{}.png'.format(opt.model_name, k))
                # clear figure to generate new one for AUC
                plt.clf()
        # check if last improved epoch exceeds patience, stop training - early stopping
        if (epoch - best_states['last_improvement_epoch']) >= opt.patience:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=opt.log)
    print('Best Epoch: {:4f} Loss: {:.4f} Accuracy: {:.4f} F1: {:.4f}'.format(best_states['last_improvement_epoch'],
                                                                              best_states['avg_loss'],
                                                                              best_states['avg_acc'],
                                                                              best_states['avg_f1']), file=opt.log)

    model.load_state_dict(best_states['model_wts'])

    return model, histories, best_states


def train_epoch(opt, model, dataloaders, criterion, optimizer):
    """Train the model for one epoch and calculates different metrics for evaluation"""
    epoch_metrics = {}
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        # create epoch metrics dictionary
        epoch_metrics[phase] = {'loss': 0.0, 'corrects': 0.0, 'totals': 0.0, 'true': [], 'pred': []}

        for inputs, labels, img_id, img_meta in tqdm(dataloaders[phase], desc='Epoch[{}]'.format(phase), leave=False):
            inputs = inputs.cuda()
            labels = labels.cuda()
            img_meta = img_meta.cuda() if opt.joint_fusion else None

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                if opt.is_inception and phase == 'train':
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2
                else:
                    if opt.joint_fusion:
                        outputs = model(inputs, img_meta)
                    else:
                        outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # softmax used to make the range be [0-1]
                softmax = nn.Softmax(dim=1)
                # gets the prediction using the highest probability value
                _, preds = torch.max(softmax(outputs), 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            epoch_metrics[phase]['loss'] += loss.item() * inputs.size(0)
            epoch_metrics[phase]['corrects'] += torch.sum(preds == labels.data).cpu()
            epoch_metrics[phase]['totals'] += inputs.size(0)
            epoch_metrics[phase]['true'].extend(labels.data.tolist())
            epoch_metrics[phase]['pred'].extend(preds.tolist())
            # maybe add an averaged across class AUC metric here for optimizing for AUC. 

        epoch_metrics[phase]['avg_loss'] = epoch_metrics[phase]['loss'] / epoch_metrics[phase]['totals']
        epoch_metrics[phase]['avg_acc'] = epoch_metrics[phase]['corrects'] / epoch_metrics[phase]['totals']
        classification_metrics = classification_report(epoch_metrics[phase]['true'], epoch_metrics[phase]['pred'],
                                                       output_dict=True)
        epoch_metrics[phase]['avg_precision'], epoch_metrics[phase]['avg_recall'], epoch_metrics[phase]['avg_f1'] = \
        classification_metrics['weighted avg']['precision'], classification_metrics['weighted avg']['recall'], \
        classification_metrics['weighted avg']['f1-score']

        print(classification_report(epoch_metrics[phase]['true'], epoch_metrics[phase]['pred']), file=opt.log)
        print('{} Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f}  f1: {:.4f}\n'
              .format(phase, epoch_metrics[phase]['avg_loss'], epoch_metrics[phase]['avg_acc'],
                      epoch_metrics[phase]['avg_precision'], epoch_metrics[phase]['avg_recall'],
                      epoch_metrics[phase]['avg_f1']), file=opt.log)
        opt.log.flush()

    return model, epoch_metrics


def model_predict(opt, model, dataloader):
    """
    Uses the given model to predict on the dataloader data and output the true and predicted labels
    Returns a dictionary with the true labels, predicted labels, ids, and scores for each class prediction
    """
    # setting model to evaluate mode
    model.cuda()
    model.eval()

    # set up output dictionary
    out_dict = {}
    for i in 

    out_dict = {'true': [], 'pred': [], 'ids': []}
    for i in range(opt.num_classes):
        out_dict['score_{}'.format(i)] = []

    out_dict = {'dcm': [], 't1_pred': [], 't1_score_0': [], 't1_score_1': []}
    if eval_CAC:
        out_dict = {'dcm': [], 't1_pred': [], 't1_score_0': [], 't1_score_1': [], 't2_pred': [], 't2_score_0': [], 't2_score_1': [], 't2_score_2': []}
    

    for inputs, labels, pt_id, img_meta in tqdm(dataloader, desc='Predicting'):
        inputs = inputs.cuda()
        labels = labels.cuda()
        img_meta = img_meta.cuda() if img_meta is not None else img_meta

        with torch.set_grad_enabled(False):
            if opt.joint_fusion:
                outputs = model(inputs, img_meta)
            else:
                outputs = model(inputs)
            softmax = nn.Softmax(dim=1)
            score = softmax(outputs)
            _, preds = torch.max(softmax(outputs), 1)

            out_dict['true'].extend(labels.data.tolist())
            out_dict['pred'].extend(preds.tolist())
            out_dict['ids'].extend(list(pt_id))
            for i in range(opt.num_classes):
                out_dict['score_{}'.format(i)].extend(score[:, i].tolist())

    return out_dict

def model_extract(opt, model, dataloader):
    """
    Uses the given model to extract feature information with the given model
    """
    # removing the last fc layer and replacing it with an identity layer
    if opt.model_name == 'resnext101':
        model.fc = nn.Identity()
    ## - add other last layers for different models in the future
    
    # setting model to evaluate mode
    model.cuda()
    model.eval()

    # set up output dictionary
    out_dict = {'true': [], 'ids': [], 'features': []}

    for inputs, labels, pt_id, img_meta in tqdm(dataloader, desc='Predicting'):
        inputs = inputs.cuda()
        labels = labels.cuda()
        img_meta = img_meta.cuda() if img_meta is not None else img_meta

        with torch.set_grad_enabled(False):
            if opt.joint_fusion:
                outputs = model(inputs, img_meta)
            else:
                outputs = model(inputs)
            outputs = outputs.cpu().numpy()
            string_out = []
            for i in outputs:
                sample = []
                for j in i:
                    sample.append(str(j))
                string_out.append(sample)
            out_dict['true'].extend(labels.data.tolist())
            out_dict['ids'].extend(list(pt_id))
            out_dict['features'].extend(string_out)

    return out_dict