import numpy as np
import torch
import os
from torchvision.utils import save_image
import pickle
from tools import (dsprite_disentanglement,
                   dsprite_disentanglement_fv)

def dsprite_handler(gen, dataset, is_vae):
    def _dsprite_handler(losses, batch, outputs, kwargs):
        #is_vae = True if args.model == 'vae' else False
        if kwargs['iter'] == 1 and kwargs['mode'] == 'valid':
            return dsprite_disentanglement_fv(gen,
                                              dataset,
                                              is_vae=is_vae)
        return {}
    return _dsprite_handler

# Add the handler to compute predictions on test set
# at start of each epoch.
def test_set_handler(gan, dataset, save_path):
    def _test_set_handler(losses, batch, outputs, kwargs):
        # Compute at the start of the validation run.
        if kwargs['iter'] == 1 and kwargs['mode'] == 'valid':
            #dest_dir = "%s/%s/test_preds" % (save_path, args.name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            preds = []
            labels = []
            cls = gan.cls_enc
            with torch.no_grad():
                for b, (x_batch, y_batch) in enumerate(dataset):
                    x_batch = x_batch.cuda()
                    enc_batch = gan.generator.encode(x_batch)
                    if hasattr(cls, 'legacy'):
                        enc_batch = enc_batch.view(-1, cls.n_in)
                    #enc_batch = enc_batch.view(-1, np.prod(enc_batch.shape[1:]))
                    preds.append(cls(enc_batch).argmax(dim=1).cpu().numpy())
                    labels.append(y_batch.argmax(dim=1).cpu().numpy())
                    #instances.append({'pred': preds_batch, 'labels': y_labels})
                preds = np.hstack(preds).astype(np.uint8)
                labels = np.hstack(labels).astype(np.uint8)
            with open("%s/%i.pkl" % (save_path, kwargs['epoch']), 'wb') as f:
                pickle.dump({'pred': preds, 'labels': labels}, f)

        return {}
    return _test_set_handler


def image_handler_default(save_path, save_images_every):
    def _image_handler_default(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1:
            if kwargs['epoch'] % save_images_every == 0:
                mode = kwargs['mode']
                epoch = kwargs['epoch']
                recon = outputs['recon']*0.5 + 0.5
                inputs = outputs['input']*0.5 + 0.5
                inputs_permed = inputs[outputs['perm']]
                recon_permed = recon[outputs['perm']]
                mix = outputs['mix']*0.5 + 0.5
                imgs = torch.cat((inputs, recon, inputs_permed, recon_permed, mix))
                save_image( imgs,
                            nrow=inputs.size(0),
                            filename="%s/%i_%s.png" % (save_path, epoch, mode))
        return {}
    return _image_handler_default

def image_handler_vae(save_path, save_images_every):
    def _image_handler_vae(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1:
            if kwargs['epoch'] % save_images_every == 0:
                mode = kwargs['mode']
                epoch = kwargs['epoch']
                input1 = outputs['input']*0.5 + 0.5
                recon1 = outputs['recon']*0.5 + 0.5
                sample = outputs['sample']*0.5 + 0.5
                imgs = torch.cat((input1, recon1, sample))
                save_image( imgs,
                            nrow=input1.size(0),
                            filename="%s/%i_%s.png" % (save_path, epoch, mode))
        return {}
    return _image_handler_vae

def image_handler_ae(save_path, save_images_every):
    def _image_handler_ae(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1:
            if kwargs['epoch'] % save_images_every == 0:
                mode = kwargs['mode']
                epoch = kwargs['epoch']
                input1 = outputs['input']*0.5 + 0.5
                recon1 = outputs['recon']*0.5 + 0.5
                imgs = torch.cat((input1, recon1))
                save_image( imgs,
                            nrow=input1.size(0),
                            filename="%s/%i_%s.png" % (save_path, epoch, mode))
        return {}
    return _image_handler_ae

def image_handler_blank(losses, batch, outputs, kwargs):
    return {}
