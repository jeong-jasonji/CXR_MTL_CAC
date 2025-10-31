import torch.optim as optim

def create_optimizer(opt):

	if opt.optimizer == 'AdamW':
		optimizer_ft = optim.AdamW(opt.params_to_update, lr=opt.learning_rate, weight_decay=opt.weight_decay)
	elif opt.optimizer == 'RAdam':
		optimizer_ft = optim.RAdam(opt.params_to_update, lr=opt.learning_rate, weight_decay=opt.weight_decay)
	elif opt.optimizer == 'NAdam':
		optimizer_ft = optim.NAdam(opt.params_to_update, lr=opt.learning_rate, weight_decay=opt.weight_decay)
	else:
		optimizer_ft = optim.Adam(opt.params_to_update, lr=opt.learning_rate, weight_decay=opt.weight_decay)

	return optimizer_ft