import logging
import shutil


class ValidationMonitor(object):
    def __init__(self, val_func, val_loader, metrics, host_metric_name='Acc', label_index_in_batch=-1, do_test=False):
        super(ValidationMonitor, self).__init__()
        self.val_func = val_func
        self.val_loader = val_loader
        self.metrics = metrics
        self.host_metric_name = host_metric_name
        self.best_epoch = -1
        self.label_index_in_batch = label_index_in_batch

        self.do_test = do_test
        self.mode = 'Test' if self.do_test else 'Val'
        if self.do_test:
            self.best_test = -1.0
        else:
            self.best_val = -1.0

    def state_dict(self):
        if self.do_test:
            return {'best_epoch': self.best_epoch,
                    'best_test': self.best_test}
        else:
            return {'best_epoch': self.best_epoch,
                    'best_val': self.best_val}

    def load_state_dict(self, state_dict):
        assert 'best_epoch' in state_dict, 'miss key \'best_epoch\''
        self.best_epoch = state_dict['best_epoch']
        if self.do_test:
            self.best_test = state_dict['best_test']
            assert 'best_test' in state_dict, 'miss key \'best_test\''
        else:
            self.best_val = state_dict['best_val']
            assert 'best_val' in state_dict, 'miss key \'best_val\''

    def __call__(self, epoch_num, net, optimizer, writer, clear_cache=None):
        if clear_cache is None:
            self.val_func(net, self.val_loader, self.metrics, self.label_index_in_batch)
        else:
            self.val_func(net, self.val_loader, self.metrics, self.label_index_in_batch, clear_cache)

        name, value = self.metrics.get()
        s = f'Epoch[{epoch_num}] \t{self.mode}-'
        for n, v in zip(name, value):
            if n == self.host_metric_name and v > (self.best_test if self.do_test else self.best_val):
                self.best_epoch = epoch_num
                if self.do_test:
                    self.best_test = v
                else:
                    self.best_val = v
                logging.info(f'New Best {self.mode} {self.host_metric_name}: {(self.best_test if self.do_test else self.best_val)}, Epoch: {self.best_epoch}')
                print(f'New Best {self.mode} {self.host_metric_name}: {(self.best_test if self.do_test else self.best_val)}, Epoch: {self.best_epoch}')
            s += "%s=%f,\t" % (n, v)
            if writer is not None:
                writer.add_scalar(tag=f'{self.mode}' + '-' + n,
                                  scalar_value=v,
                                  global_step=epoch_num + 1)
        logging.info(s)
        print(s)

        logging.info(f'Best {self.mode} {self.host_metric_name}: {(self.best_test if self.do_test else self.best_val)}, Epoch: {self.best_epoch}')
        print(f'Best {self.mode} {self.host_metric_name}: {(self.best_test if self.do_test else self.best_val)}, Epoch: {self.best_epoch}')





