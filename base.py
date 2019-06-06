import torch
import os
import time
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from torch import optim
from torch import nn
from torchvision.utils import save_image

class Base:
    def train(self,
              itr_train,
              itr_valid,
              epochs,
              model_dir,
              result_dir,
              save_every=1,
              scheduler_fn=None,
              scheduler_args={},
              verbose=True):
        for folder_name in [model_dir, result_dir]:
            if folder_name is not None and not os.path.exists(folder_name):
                os.makedirs(folder_name)
        f_mode = 'w' if not os.path.exists("%s/results.txt" % result_dir) else 'a'
        f = None
        if result_dir is not None:
            f = open("%s/results.txt" % result_dir, f_mode)
        for epoch in range(self.last_epoch, epochs):
            epoch_start_time = time.time()
            # Training.
            if verbose:
                pbar = tqdm(total=len(itr_train))
            train_dict = OrderedDict({'epoch': epoch+1})
            # item, pose, id
            for b, batch in enumerate(itr_train):
                batch = self.prepare_batch(batch)
                losses, outputs = self.train_on_instance(*batch,
                                                         iter=b+1)
                for key in losses:
                    this_key = 'train_%s' % key
                    if this_key not in train_dict:
                        train_dict[this_key] = []
                    train_dict[this_key].append(losses[key])
                if verbose:
                    pbar.update(1)
                    pbar.set_postfix(self._get_stats(train_dict, 'train'))
                # Process handlers.
                for handler_fn in self.handlers:
                    handler_fn(losses, batch, outputs,
                               {'epoch':epoch+1, 'iter':b+1, 'mode':'train'})
            if verbose:
                pbar.close()
            valid_dict = {}
            # TODO: enable valid
            if verbose:
                pbar = tqdm(total=len(itr_valid))
            # Validation.
            valid_dict = OrderedDict({})
            for b, valid_batch in enumerate(itr_valid):
                valid_batch = self.prepare_batch(valid_batch)
                valid_losses, valid_outputs = self.eval_on_instance(*valid_batch,
                                                                    iter=b+1)
                for key in valid_losses:
                    this_key = 'valid_%s' % key
                    if this_key not in valid_dict:
                        valid_dict[this_key] = []
                    valid_dict[this_key].append(valid_losses[key])
                if verbose:
                    pbar.update(1)
                    pbar.set_postfix(self._get_stats(valid_dict, 'valid'))
                # Process handlers.
                for handler_fn in self.handlers:
                    handler_fn(valid_losses, valid_batch, valid_outputs,
                               {'epoch':epoch+1, 'iter':b+1, 'mode':'valid'})
            if verbose:
                pbar.close()
            # Step learning rates.
            for sched in self.schedulers:
                sched.step()
            # Update dictionary of values.
            all_dict = train_dict
            all_dict.update(valid_dict)
            for key in all_dict:
                all_dict[key] = np.mean(all_dict[key])
            for key in self.optim:
                all_dict["lr_%s" % key] = \
                        self.optim[key].state_dict()['param_groups'][0]['lr']
            all_dict['time'] = time.time() - epoch_start_time
            str_ = ",".join([str(all_dict[key]) for key in all_dict])
            print(str_)
            if result_dir is not None:
                if (epoch+1) == 1:
                    f.write(",".join(all_dict.keys()) + "\n")
                f.write(str_ + "\n")
                f.flush()
            if (epoch+1) % save_every == 0 and model_dir is not None:
                self.save(filename="%s/%i.pkl" % (model_dir, epoch+1),
                          epoch=epoch+1)
        if f is not None:
            f.close()

    def vis_batch(self, batch, outputs):
        raise NotImplementedError()
