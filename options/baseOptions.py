import os
import json
import torch
import argparse
from base import utilsProcessing

class BaseOptions():
    def __init__(self, json_filepath, args, is_train=True):
        self.initialized = False
        self.json_filepath = json_filepath
        self.args = args
        self.is_train = is_train

    def initialize(self):
        # load from the json file the test arguments
        self.opt = argparse.Namespace(**json.load(open(self.json_filepath, "r")))
        self.initialized = True

    def parse(self, save=True):
        # set up initialization
        if not self.initialized:
            self.initialize()
        #self.opt.isTrain = self.isTrain   # train or test
        self.opt.checkpoints_dir = self.args.checkpoints_dir
        self.opt.max_epochs = self.args.epochs
        self.opt.dataframes_dir = self.args.df_path
        self.opt.hist_norm = "./cac_mace_clf/normalized_hist_array_manual.pkl"
        self.opt.gpu_ids = self.args.device[-1]

        # set gpu ids and set cuda devices
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set class weights if not none
        if self.opt.cls_weights is not None:
            self.opt.cls_weights = [float(i) for i in self.opt.cls_weights.split(',')]
        if self.opt.t1_cls_weights is not None:
            self.opt.t1_cls_weights = [float(i) for i in self.opt.t1_cls_weights.split(',')]
        if self.opt.t2_cls_weights is not None:
            self.opt.t2_cls_weights = [float(i) for i in self.opt.t2_cls_weights.split(',')]

        if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)
        
        # set up dataframes - opt.dataframes_dir should only have dataframe files in csv or pkl
        if self.is_train:
            self.opt.dataframe_train = None
            self.opt.dataframe_val = None
            self.opt.dataframe_test = None
            self.opt.dataframe_ext = None
            for k in os.listdir(self.opt.dataframes_dir):
                if 'train.csv' in k:
                    self.opt.dataframe_train = os.path.join(self.opt.dataframes_dir, k)
                elif 'val.csv' in k:
                    self.opt.dataframe_val = os.path.join(self.opt.dataframes_dir, k)
                elif 'test.csv' in k:
                    self.opt.dataframe_test = os.path.join(self.opt.dataframes_dir, k)
                elif 'ext.csv' in k:
                    self.opt.dataframe_ext = os.path.join(self.opt.dataframes_dir, k)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # set up save dir and logs
        #if self.opt.optuna_save:
        self.opt.save_dir = os.path.join(self.opt.checkpoints_dir, str(self.opt.test_name) + '/')
        utilsProcessing.mkdirs(self.opt.save_dir)
        
        #if self.opt.optuna_save:
        # save test options
        file_name_opt = os.path.join(self.opt.save_dir, 'opt.txt')
        with open(file_name_opt, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        # make a log file
        log_file = os.path.join(self.opt.save_dir, 'log.txt')
        self.opt.log = open(log_file, 'a')
        print('------------ Log -------------\n', file=self.opt.log)
        #else:
        #    log_file = os.path.join(self.opt.save_dir, 'optuna_log.txt')
        #    self.opt.log = open(log_file, 'a')
        #    print('------------ Log -------------\n', file=self.opt.log)

        return self.opt