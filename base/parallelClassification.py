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


def train_epochs(opt, model, dataloaders, criterion_1, criterion_2, optimizer):
    """Runs the model for the number of epochs given and keeps track of best weights and metrics"""

    since = time.time()

    best_states = {'model_wts': copy.deepcopy(model.state_dict()), 'avg_acc': 0.0, 'avg_f1': 0.0, 'avg_loss': 999.0,
                   'last_improvement_epoch': 0}
    avg_histories = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
    t1_histories = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
    t2_histories = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}

    for epoch in tqdm(range(opt.max_epochs), desc='Training'):
        print('Epoch {}/{}'.format(epoch + 1, opt.max_epochs), file=opt.log)

        model, epoch_metrics = train_epoch(opt, model, dataloaders, criterion_1, criterion_2, optimizer)

        time_elapsed = time.time() - since
        print('Epoch {} complete: {:.0f}m {:.0f}s'.format(epoch + 1, time_elapsed // 60, time_elapsed % 60),
              file=opt.log)

        # deep copy the model if the accuracy and auc improves
        if opt.optim_metric != 'avg_loss' and epoch_metrics['val'][opt.optim_metric] > best_states[opt.optim_metric]:
            best_states['avg_loss'] = epoch_metrics['val']['avg_loss']
            best_states['avg_acc'] = epoch_metrics['val']['avg_acc']
            best_states['avg_f1'] = epoch_metrics['val']['avg_f1']
            best_states['last_improvement_epoch'] = epoch
            best_states['model_wts'] = copy.deepcopy(model.state_dict())
            print('Improved {}, Updated weights \n'.format(opt.optim_metric), file=opt.log)
        elif opt.optim_metric == 'avg_loss' and epoch_metrics['val'][opt.optim_metric] < best_states[opt.optim_metric]:
            best_states['avg_loss'] = epoch_metrics['val']['avg_loss']
            best_states['avg_acc'] = epoch_metrics['val']['avg_acc']
            best_states['avg_f1'] = epoch_metrics['val']['avg_f1']
            best_states['last_improvement_epoch'] = epoch
            best_states['model_wts'] = copy.deepcopy(model.state_dict())
            print('Improved {}, Updated weights \n'.format(opt.optim_metric), file=opt.log)
        opt.log.flush()
        # save histories
        avg_histories['train_acc'].append(epoch_metrics['train']['avg_acc'])
        avg_histories['train_loss'].append(epoch_metrics['train']['avg_loss'])
        avg_histories['val_acc'].append(epoch_metrics['val']['avg_acc'])
        avg_histories['val_loss'].append(epoch_metrics['val']['avg_loss'])
        
        t1_histories['train_acc'].append(epoch_metrics['train']['avg_acc1'])
        t1_histories['train_loss'].append(epoch_metrics['train']['avg_loss1'])
        t1_histories['val_acc'].append(epoch_metrics['val']['avg_acc1'])
        t1_histories['val_loss'].append(epoch_metrics['val']['avg_loss1'])
        
        t2_histories['train_acc'].append(epoch_metrics['train']['avg_acc2'])
        t2_histories['train_loss'].append(epoch_metrics['train']['avg_loss2'])
        t2_histories['val_acc'].append(epoch_metrics['val']['avg_acc2'])
        t2_histories['val_loss'].append(epoch_metrics['val']['avg_loss2'])

        if opt.save_every_epoch:
            model.load_state_dict(best_states['model_wts'])
            torch.save(model, opt.save_dir + '{}.pth'.format(opt.model_name))
            pickle.dump(avg_histories, open(opt.save_dir + "{}_history.pkl".format(opt.model_name), "wb"))

        # save plots for the best ones
        for k in ['acc', 'loss']:
            avg_train_hist = [h for h in avg_histories['train_{}'.format(k)]]
            avg_val_hist = [h for h in avg_histories['val_{}'.format(k)]]
            t1_train_hist = [h for h in t1_histories['train_{}'.format(k)]]
            t1_val_hist = [h for h in t1_histories['val_{}'.format(k)]]
            t2_train_hist = [h for h in t2_histories['train_{}'.format(k)]]
            t2_val_hist = [h for h in t2_histories['val_{}'.format(k)]]
            plt.xlabel("Training Epochs")
            plt.title("{} over Epochs".format('Accuracy' if k == 'acc' else 'Loss'))
            plt.ylabel("{}".format('Accuracy' if k == 'acc' else 'Loss'))
            plt.plot(range(1, epoch + 2), avg_train_hist)
            plt.plot(range(1, epoch + 2), avg_val_hist)
            plt.plot(range(1, epoch + 2), t1_train_hist)
            plt.plot(range(1, epoch + 2), t1_val_hist)
            plt.plot(range(1, epoch + 2), t2_train_hist)
            plt.plot(range(1, epoch + 2), t2_val_hist)
            plt.legend(['Avg_Train', 'Avg_Val', 't1_Train', 't1_Val', 't2_Train', 't2_Val'])
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
    histories = [avg_histories, t1_histories, t2_histories]

    return model, histories, best_states


def train_epoch(opt, model, dataloaders, criterion_1, criterion_2, optimizer):
    """Train the model for one epoch and calculates different metrics for evaluation"""
    epoch_metrics = {}
    softmax = nn.Softmax(dim=1)
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        # create epoch metrics dictionary
        epoch_metrics[phase] = {'loss': 0.0, 'loss_1': 0.0, 'loss_2': 0.0,
                'corrects': 0.0, 'corrects_1': 0.0, 'corrects_2': 0.0,
                'true': [], 'true_1': [], 'true_2': [],
                'pred': [], 'pred_1': [], 'pred_2': [],
                'totals': 0.0}

        for inputs, labels1, labels2, img_id, img_meta in tqdm(dataloaders[phase], desc='Epoch[{}]'.format(phase), leave=False):
            inputs = inputs.cuda()
            labels1 = labels1.cuda()
            labels2 = labels2.cuda()
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
                        outputs1, outputs2 = model(inputs, img_meta)
                    else:
                        outputs1, outputs2 = model(inputs)
                    loss1 = criterion_1(outputs1, labels1)
                    loss2 = criterion_2(outputs2, labels2)
                
                # get preds for task1
                _, preds1 = torch.max(softmax(outputs1), 1)
                # get preds for task2
                _, preds2 = torch.max(softmax(outputs2), 1)

                # backward + optimize only if in training phase
                loss = (loss1 + loss2) / 2
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            epoch_metrics[phase]['loss_1'] += loss1.item() * inputs.size(0)
            epoch_metrics[phase]['corrects_1'] += torch.sum(preds1 == labels1.data).cpu()
            epoch_metrics[phase]['true_1'].extend(labels1.data.tolist())
            epoch_metrics[phase]['pred_1'].extend(preds1.tolist())
            epoch_metrics[phase]['loss_2'] += loss2.item() * inputs.size(0)
            epoch_metrics[phase]['corrects_2'] += torch.sum(preds2 == labels2.data).cpu()
            epoch_metrics[phase]['true_2'].extend(labels2.data.tolist())
            epoch_metrics[phase]['pred_2'].extend(preds2.tolist())
            epoch_metrics[phase]['loss'] += loss.item() * inputs.size(0)
            epoch_metrics[phase]['totals'] += inputs.size(0)
            # maybe add an averaged across class AUC metric here for optimizing for AUC. 

        epoch_metrics[phase]['avg_loss'] = epoch_metrics[phase]['loss'] / epoch_metrics[phase]['totals']
        epoch_metrics[phase]['avg_loss1'] = epoch_metrics[phase]['loss_1'] / epoch_metrics[phase]['totals']
        epoch_metrics[phase]['avg_loss2'] = epoch_metrics[phase]['loss_2'] / epoch_metrics[phase]['totals']
        
        epoch_metrics[phase]['avg_acc1'] = epoch_metrics[phase]['corrects_1'] / epoch_metrics[phase]['totals']
        epoch_metrics[phase]['avg_acc2'] = epoch_metrics[phase]['corrects_2'] / epoch_metrics[phase]['totals']
        epoch_metrics[phase]['avg_acc'] = (epoch_metrics[phase]['avg_acc1'] + epoch_metrics[phase]['avg_acc2']) / 2
        # metrics for task1
        classification_metrics = classification_report(epoch_metrics[phase]['true_1'], epoch_metrics[phase]['pred_1'],output_dict=True)
        epoch_metrics[phase]['avg_precision1'], epoch_metrics[phase]['avg_recall1'], epoch_metrics[phase]['avg_f11'] = \
        classification_metrics['weighted avg']['precision'], classification_metrics['weighted avg']['recall'], \
        classification_metrics['weighted avg']['f1-score']
        # metrics for task2
        classification_metrics = classification_report(epoch_metrics[phase]['true_2'], epoch_metrics[phase]['pred_2'],output_dict=True)
        epoch_metrics[phase]['avg_precision2'], epoch_metrics[phase]['avg_recall2'], epoch_metrics[phase]['avg_f12'] = \
        classification_metrics['weighted avg']['precision'], classification_metrics['weighted avg']['recall'], \
        classification_metrics['weighted avg']['f1-score']
        print(classification_report(epoch_metrics[phase]['true_1'], epoch_metrics[phase]['pred_1']), file=opt.log)
        print(classification_report(epoch_metrics[phase]['true_2'], epoch_metrics[phase]['pred_2']), file=opt.log)
        
        epoch_metrics[phase]['avg_precision'] = (epoch_metrics[phase]['avg_precision1'] + epoch_metrics[phase]['avg_precision2'])/ 2
        epoch_metrics[phase]['avg_recall'] = (epoch_metrics[phase]['avg_recall1'] + epoch_metrics[phase]['avg_recall2'])/ 2
        epoch_metrics[phase]['avg_f1'] = (epoch_metrics[phase]['avg_f11'] + epoch_metrics[phase]['avg_f12'])/ 2
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
    softmax = nn.Softmax(dim=1)
    # set up output dictionary
    out_dict = {'ids': [], 't1_true': [], 't1_pred': [], 't2_true': [], 't2_pred': []}
    for i in range(opt.task1_cls):
        out_dict['t1_score_{}'.format(i)] = []
    for i in range(opt.task2_cls):
        out_dict['t2_score_{}'.format(i)] = []

    for inputs, labels1, labels2, pt_id, img_meta in tqdm(dataloader, desc='Predicting'):
        inputs = inputs.cuda()
        labels1 = labels1.cuda()
        labels2 = labels2.cuda()
        img_meta = img_meta.cuda() if img_meta is not None else img_meta

        with torch.set_grad_enabled(False):
            if opt.joint_fusion:
                outputs1, outputs2 = model(inputs, img_meta)
            else:
                outputs1, outputs2 = model(inputs)
            
            out_dict['ids'].extend(list(pt_id))
            score1 = softmax(outputs1)
            _, preds1 = torch.max(softmax(outputs1), 1)
            out_dict['t1_true'].extend(labels1.data.tolist())
            out_dict['t1_pred'].extend(preds1.tolist())
            for i in range(opt.task1_cls):
                out_dict['t1_score_{}'.format(i)].extend(score1[:, i].tolist())
            
            score2 = softmax(outputs2)
            _, preds2 = torch.max(softmax(outputs2), 1)
            out_dict['t2_true'].extend(labels2.data.tolist())
            out_dict['t2_pred'].extend(preds2.tolist())
            for i in range(opt.task2_cls):
                out_dict['t2_score_{}'.format(i)].extend(score2[:, i].tolist())

    return out_dict