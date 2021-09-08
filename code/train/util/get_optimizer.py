def get_optimizer(optimizer_name, params, learning_rate, l2_weight_decay):
    if optimizer_name == 'SGD':
        from torch.optim import SGD
        optimizer = SGD(params=params, lr=learning_rate, weight_decay=l2_weight_decay)

    elif optimizer_name == 'Adam':
        from torch.optim import Adam
        optimizer = Adam(params=params, lr=learning_rate, weight_decay=l2_weight_decay)

    elif optimizer_name == 'RMS':
        from torch.optim.rmsprop import RMSprop
        optimizer = RMSprop(params=params, lr=learning_rate, weight_decay=l2_weight_decay)

    elif optimizer_name == 'Lookahead(Adam)':
        from .optimizer_plus.optimizer import Lookahead
        from torch.optim import Adam
        base_optimizer = Adam(params=params, lr=learning_rate, weight_decay=l2_weight_decay)
        optimizer = Lookahead(base_optimizer=base_optimizer)

    else:
        raise NotImplementedError

    return optimizer
