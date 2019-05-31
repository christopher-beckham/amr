import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import tempfile
import torch
import numpy as np
from itertools import product
from torch import nn
from torchvision.utils import save_image
from tqdm import tqdm
from skimage.io import (imread,
                        imsave)

default_font = {'color':  'red',
                'weight': 'heavy',
                'size': 16,
                'backgroundcolor': 'white'}

def line2dict(st):
    """Convert a line of key=value pairs to a
    dictionary.

    :param st:
    :returns: a dictionary
    :rtype:
    """
    elems = st.split(',')
    dd = {}
    for elem in elems:
        elem = elem.split('=')
        key, val = elem
        try:
            int_val = int(val)
            dd[key] = int_val
        except ValueError:
            dd[key] = val
    return dd

def ndprint(x):
    # https://stackoverflow.com/questions/2891790/how-to-pretty-print-a-numpy-array-without-scientific-notation-and-with-given-pre
    print(['{:.2f}'.format(i) for i in x])

def binary_xent(p):
    return np.sum((-p*np.log(p+1e-6) - (1-p)*np.log(1-p+1e-6)))

def min_max_norm(v):
    return ( v - np.min(v) ) / (v.max() - v.min())
    
def count_params(module, trainable_only=True):
    """Count the number of parameters in a
    module.

    :param module: PyTorch module
    :param trainable_only: only count trainable
      parameters.
    :returns: number of parameters
    :rtype:

    """
    parameters = module.parameters()
    if trainable_only:
        parameters = filter(lambda p: p.requires_grad, parameters)
    num = sum([np.prod(p.size()) for p in parameters])
    return num

def compute_inception(loader,
                      gan,
                      cls,
                      save_path,
                      batch_size,
                      n_classes,
                      num_repeats=5):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = open("%s/scores.txt" % save_path, "w")
    # Run this 5 times
    for iter_ in range(num_repeats):
        # Compute the p(y|x) for 50k random mixes.
        N = 50000
        preds = np.zeros((N, n_classes)).astype(np.float32)
        for i, (x_batch, _) in enumerate(loader):
            if gan.use_cuda:
                x_batch = x_batch.cuda()
            batch_size_i = x_batch.size()[0]
            x_sample = gan.sample(x_batch)
            pred = cls(x_sample).detach().cpu().numpy()
            preds[i*batch_size:i*batch_size + batch_size_i] = pred
        # Compute binary x-ent for all prob distns.
        scores = []
        for i in range(len(preds)):
            this_xent = binary_xent(preds[i])
            scores.append(np.exp(this_xent))
        print(np.mean(scores))
        f.write("%f\n" % np.mean(scores))
    f.close()

def compute_fid(loader,
                gan,
                cls,
                save_path,
                num_repeats=5):

    from fid_score import calculate_fid_given_imgs
    
    # Collect the training set.
    train_samples = []
    gen_samples = []
    recon_samples = []
    for x_batch, _ in loader:
        train_samples.append(x_batch)
        recon_samples.append(gan.reconstruct(x_batch).cpu().numpy())
    train_samples = np.vstack(train_samples)
    recon_samples = np.vstack(recon_samples)

    train_samples = (((train_samples*0.5) + 0.5)*255.).astype(np.int32)
    recon_samples = (((recon_samples*0.5) + 0.5)*255.).astype(np.int32)

    #########################################
    # Write FID between samples and dataset #
    #########################################

    use_cuda = gan.use_cuda
    scores = []
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = open("%s/scores.txt" % save_path, "w")
    print("Writing file to: %s" % save_path)
    for iter_ in range(num_repeats):

        gen_samples = []
        for x_batch, _ in loader:
            gen_samples.append(gan.sample(x_batch).cpu().numpy())
        gen_samples = np.vstack(gen_samples)
        gen_samples = (((gen_samples*0.5) + 0.5)*255.).astype(np.int32)

        score = calculate_fid_given_imgs(train_samples,
                                         gen_samples,
                                         16,
                                         use_cuda,
                                         dims=512,
                                         model=cls)
        scores.append(score)
        f.write("%f\n" % score)
        print("Score between train and sample for mix=%s: %f" % (gan.mixer, score))
    f.close()
    print("Mean score between train and sample for mix=%s: %f" % (gan.mixer, np.mean(scores)))

    #################################################
    # Write FID between reconstructions and dataset #
    #################################################

    f = open("%s/scores_recon.txt" % save_path, "w")
    score = calculate_fid_given_imgs(train_samples,
                                     recon_samples,
                                     16,
                                     use_cuda,
                                     dims=512,
                                     model=cls)
    print("Score between train and reconstruction: %f" % score)
    f.write("%f\n" % score)
    f.close()
    
def _extract_encodings(loader,
                       gan,
                       early_stop):
    gan._eval()
    with torch.no_grad():
        buf = []
        y_buf = []
        pbar = tqdm(total=len(loader))
        for b, (x_batch, y_batch) in enumerate(loader):
            if gan.use_cuda:
                x_batch = x_batch.cuda()
            enc = gan.generator.encode(x_batch)
            buf.append(enc.data.cpu().numpy())
            y_buf.append(y_batch.numpy())
            pbar.update(1)
            if b == early_stop:
                break
    buf = np.vstack(buf)
    if len(enc.size()) == 4:
        buf = buf.reshape(-1, np.product(enc.size()[1::]))
    y_buf = np.vstack(y_buf)
    return buf, y_buf
    
def save_embedding(loader,
                   gan,
                   save_file,
                   early_stop=-1):
    """Extract the bottleneck features and save
    it, in npz format.
    """
    gan._eval()
    save_path = os.path.dirname(save_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    buf, y_buf = _extract_encodings(loader, gan, early_stop)
    print("Saving to %s..." % save_path)
    np.savez(save_file,
             X_train=buf, y_train=y_buf)

def train_logreg(loader,
                 gan,
                 save_path,
                 early_stop=-1,
                 max_iters=10000):
    """Train a logistic regression classifier on the embedding

    """
    from sklearn.linear_model import LogisticRegression
    gan._eval()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open("%s/logreg.txt" % save_path, "w") as f:
        X, y = _extract_encodings(loader, gan, early_stop)
        y = y.argmax(axis=1)
        lr = LogisticRegression(multi_class='auto',
                                solver='lbfgs',
                                max_iter=max_iters,
                                verbose=2)
        lr.fit(X, y)
        acc = (lr.predict(X) == y).mean()
        print("Accuracy: %f" % acc)
        f.write("%f\n" % acc)
    

def save_class_embedding(gan,
                         n_classes,
                         save_path):
    # NOTE: only works for binary attributes atm
    if gan.cls <= 0:
        raise Exception("cls must be > 0 in order to look at class embeddings")
    if gan.class_mixer is None:
        raise Exception("Cannot find attribute `class_mixer` in `gan`!")
    gan._eval()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    lists = []
    for _ in range(n_classes):
        lists.append((0, 1))
    ys = sorted([elem for elem in product(*lists)])
    df = None
    with torch.no_grad():
        for y in ys:
            this_y = torch.FloatTensor([y])
            if gan.use_cuda:
                this_y = this_y.cuda()
            embedding = gan.class_mixer.embed(this_y)
            embedding = embedding.cpu().numpy()
            embedding = embedding.reshape((1, embedding.shape[1]))
            if df is None:
                df = embedding
            else:
                df = np.vstack((df, embedding))
    np.savetxt(fname="%s/file.csv" % save_path,
               X=df, delimiter=",")

def generate_2d_plot(loader,
                     gan,
                     save_path,
                     early_stop=-1):
    """Generate a scatterplot of the bottleneck.
    NOTE: this only makes sense if your bottleneck
    is two-dimensional.
    """
    gan._eval()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    x_2d, y = _extract_encodings(loader, gan, early_stop=early_stop)
    n_classes = y.shape[1]
    y_int = y.argmax(axis=1)

    if x_2d.shape[1] != 2:
        raise Exception("Expected bottleneck to be of shape (N, 2)!")
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_2d[:, 0], x_2d[:, 1], c=y_int)
    k_means = []
    tot_mean = np.mean(x_2d, axis=0)
    tot_std = np.std(x_2d, axis=0)
    for k in range(n_classes):
        k_ctr = np.mean(x_2d[y_int == k], axis=0)
        k_means.append((k_ctr - tot_mean) / tot_std)
        ax.text(k_ctr[0], k_ctr[1], str(k), fontdict=default_font)
        ax.set_axis_off()
    """
    distances = []
    for i in range(len(k_means)):
        for j in range(i+1, len(k_means)):
            distances.append(np.sum((k_means[i] - k_means[j])**2))
    g.write("%f\n" % np.mean(distances))
    """
    
def generate_tsne(loader,
                  gan,
                  save_path,
                  early_stop=-1,
                  n_cores=4,
                  n_repeats=5,
                  use_labels=True):
    """Generate a t-SNE embedding in npz format and
    save it, as well as a plot.

    :param loader:
    :param gan:
    :param save_path: save fies to this path
    :param early_stop: loop through `loader` only this
      many times. (Good if you don't want to go through
      the entire dataset.)
    :param n_cores: number of CPU cores to use
    :param n_repeats: number of repeat t-SNE runs to do
    :param use_labels: if `True`, determine the clusters
      with the labels. Otherwise, use KMeans to find the
      clusters.
    :returns: 
    :rtype: 

    """
    from MulticoreTSNE import MulticoreTSNE as TSNE

    gan._eval()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    buf, y_buf = _extract_encodings(loader, gan, early_stop)
    out_name = "tsne" if use_labels else "tsne_unsup"
    with open("%s/%s.txt" % (save_path, out_name), "w") as g:
        for iter_ in range(n_repeats):
            print("iter_", iter_)
            tsne = TSNE(n_jobs=n_cores, verbose=1, random_state=iter_)
            # Fit a t-SNE and save it.
            x_2d = tsne.fit_transform(buf)
            if iter_ == 0:
                print("Saving to %s..." % save_path)
                np.savez("%s/embeddings_tsne.npz" % save_path,
                         X_train=x_2d, y_train=y_buf)
            # Also save a plot to the same dir as well.
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(1, 1, 1)

            n_classes = y_buf.shape[1]
            if not use_labels:
                from sklearn.cluster import KMeans
                kmeans_model = KMeans(n_clusters=n_classes).fit(x_2d)
                kmeans_labels = kmeans_model.labels_

            y_train_int = y_buf.argmax(axis=1)

            if use_labels:
                np.savez("%s/embeddings.npz" % save_path,
                         X_train=buf, y_train=y_train_int)

            ax.scatter(x_2d[:, 0], x_2d[:, 1],
                       c=y_train_int if use_labels else kmeans_labels)
            k_means = []
            tot_mean = np.mean(x_2d, axis=0)
            tot_std = np.std(x_2d, axis=0)

            for k in range(n_classes):
                if use_labels:
                    k_ctr = np.mean(x_2d[y_train_int == k], axis=0)
                else:
                    k_ctr = np.mean(x_2d[kmeans_labels == k], axis=0)
                k_means.append((k_ctr - tot_mean) / tot_std)
                ax.text(k_ctr[0], k_ctr[1], str(k), fontdict=default_font)
                ax.set_axis_off()
            distances = []
            for i in range(len(k_means)):
                for j in range(i+1, len(k_means)):
                    distances.append(np.sum((k_means[i] - k_means[j])**2))
            g.write("%f\n" % np.mean(distances))
            title = save_path.split("/")[1]
            #ax.set_title(title + "\n" + ("%f +/- %f" % (inter_mean, inter_std)),
            #             size=16, weight='heavy')
            ax.set_title(title)
            fig.savefig("%s/%s_%i.pdf" % (save_path, out_name, iter_))


def save_frames(gan, x_batch, out_folder, num_interps=10):
    #TODO
    gan._eval()
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for img_idx in range(x_batch.size(0)):
        this_folder = "%s/%i" % (out_folder, img_idx)
        if not os.path.exists(this_folder):
            os.makedirs(this_folder)

    with torch.no_grad():
        enc = gan.generator.encode(x_batch)
        perm = torch.randperm(x_batch.size(0))
        for interp_idx, p in enumerate(np.linspace(0, 1, num=num_interps)):
            print(interp_idx)
            alpha = gan.sampler(enc.size(0), enc.size(1), p=p)
            enc_mix = alpha*enc + (1.-alpha)*enc[perm]
            dec_enc_mix = gan.generator.decode(enc_mix)
            for img_idx in range(x_batch.size(0)):
                out_file = "%s/%i/{0:06d}.png".format(interp_idx) % (out_folder, img_idx)
                save_image(dec_enc_mix[img_idx]*0.5 + 0.5,
                           filename=out_file,
                           padding=0)            


def save_frames_continuous(gan,
                           x_batch,
                           save_path,
                           num_interps=10,
                           framerate=30,
                           crf=23,
                           resize_to=-1):
    #TODO
    gan._eval()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tmp_dir = tempfile.mkdtemp()
    tmp_dir2 = tempfile.mkdtemp()
    print("tmp_dir: %s" % tmp_dir)
    print("tmp_dir2: %s" % tmp_dir2)
    cc = 0
    fig_height, fig_width = 4.5, 15
    img_h = 50
    pbar = tqdm(total=x_batch.size(0)-1)
    interp_space = np.linspace(0, 1, num_interps)
    encs = gan.generator.encode(x_batch)
    for i in range(x_batch.size(0)-1):
        with torch.no_grad():
            enc = encs[i:i+1]
            enc_perm = encs[(i+1):(i+2)]
            # produce interpolation between image1 and image2
            for p in interp_space:
                alpha = gan.sampler(enc.size(0), enc.size(1), p=p)
                enc_mix = (1.-alpha)*enc + alpha*enc_perm
                # Create the image for the mix.
                dec_enc_mix = gan.generator.decode(enc_mix)
                out_file = "%s/{0:06d}.png".format(cc) % (tmp_dir)
                save_image(dec_enc_mix*0.5 + 0.5,
                           filename=out_file,
                           padding=0)
                """
                # Create the feature map figure.
                ev1 = min_max_norm(enc.cpu().numpy().mean(axis=(2,3)))
                ev1 = ev1.repeat(img_h, axis=0)
                ev2 = min_max_norm(enc_perm.cpu().numpy().mean(axis=(2,3)))
                ev2 = ev2.repeat(img_h, axis=0)
                av = alpha.reshape((-1, alpha.size(1))).cpu().numpy()
                ax = av.repeat(img_h, axis=0)
                fig, ax = plt.subplots(3, 1)
                fig.set_figheight(fig_height)
                fig.set_figwidth(fig_width)
                for a in ax:
                    a.set_yticklabels([])
                    a.set_yticks([])
                ax[0].imshow((1.-av) * ev1,
                             vmin=0., vmax=1., cmap='inferno')
                ax[0].set_title('m * f(x1)')
                ax[1].imshow(av * ev2,
                             vmin=0., vmax=1., cmap='inferno')
                ax[1].set_title('(1-m) * f(x2)')
                ax[2].imshow((1.-av) * ev1 + av * ev2,
                             vmin=0., vmax=1., cmap='inferno')
                ax[2].set_title('m * f(x1) + (1-m) * f(x2)')
                fig.savefig("%s/{0:06d}.png".format(cc) % tmp_dir2,
                            bbox_inches='tight')
                plt.close(fig)
                """
                cc += 1
        pbar.update(1)
    pbar.close()
    # Now run ffmpeg on this and save it as out.mp4
    from subprocess import check_output
    if os.path.exists("%s/out.mp4" % save_path):
        os.remove("%s/out.mp4" % save_path)
    resize_to = -1
    scale_str = ("-vf scale=%i:%i" % (resize_to, resize_to)) if resize_to != -1 else ""
    ffmpeg_out = check_output(
        "cd %s; ffmpeg -framerate %i -pattern_type glob -i '*.png' -c:v libx264 %s -crf %i out.mp4" % (tmp_dir, framerate, scale_str, crf),
        shell=True)
    ffmpeg_out = ffmpeg_out.decode('utf-8').rstrip()
    copy_out = check_output(
        "cp %s/out.mp4 %s/out.mp4" % (tmp_dir, save_path),
        shell=True
    )
    print(ffmpeg_out)
    print(copy_out)


def save_consistency_plot(gan, x_batch, out_folder):
    """
    """
    gan._eval()
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        
    with torch.no_grad():
        if type(x_batch) in [tuple, list]:
            x_batch_1, x_batch_2 = x_batch
        else:
            x_batch_1 = x_batch
            perm = torch.randperm(x_batch.size(0))
            x_batch_2 = x_batch[perm]
            
        enc1 = gan.generator.encode(x_batch_1)
        enc2 = gan.generator.encode(x_batch_2)
        is_2d = True if len(enc1.size()) == 2 else False
        alpha = gan.sampler(enc1.size(0), enc1.size(1), is_2d)
        enc_mix = alpha*enc1 + (1.-alpha)*enc2
        dec_enc_mix = gan.generator.decode(enc_mix)
        enc_dec_enc_mix = gan.generator.encode(dec_enc_mix)

        enc_mix = enc_mix.cpu().numpy()
        enc_dec_enc_mix = enc_dec_enc_mix.cpu().numpy()
        coords_stacked = np.hstack((enc_mix, enc_dec_enc_mix))

        fig, ax = plt.subplots(1,1)

        # Plot the actual encoded pts.
        enc_np = enc1.detach().cpu().numpy()
        ax.scatter(enc_np[:,0], enc_np[:,1], alpha=0.5)
        # Show before and after for mix.
        ax.quiver(enc_mix[:,0], enc_mix[:,1],
                  enc_dec_enc_mix[:,0], enc_dec_enc_mix[:,1],
                  width=0.002)
        fig.savefig("%s/plot.png" % out_folder)

    
def save_interp(gan, x_batch, out_folder, num=10, mix_input=False, padding=2, show_real=False):
    """Save interpolations between a batch and its permuted
    version to disk.
    :param gan: 
    :param x_batch: 
    :param out_folder: 
    :param num: number of interpolation steps to perform
    :param mix_input: if `True`, only produce input space mix
    :param padding: padding on image grid
    :returns: 
    :rtype: 
    """
    gan._eval()
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    pbuf = []
    with torch.no_grad():
        if type(x_batch) in [tuple, list]:
            x_batch_1, x_batch_2 = x_batch
        else:
            x_batch_1 = x_batch
            perm = torch.randperm(x_batch.size(0))
            x_batch_2 = x_batch[perm]
        enc1 = gan.generator.encode(x_batch_1)
        enc2 = gan.generator.encode(x_batch_2)
        is_2d = True if len(enc1.size()) == 2 else False
        for p in np.linspace(0, 1, num=num):
            if mix_input:
                alpha = gan.sampler(x_batch_1.size(0), 1, is_2d, p=p)
                dec_enc_mix = alpha*x_batch_1 + (1.-alpha)*x_batch_2
            else:
                alpha = gan.sampler(enc1.size(0), enc1.size(1), is_2d, p=p)
                enc_mix = alpha*enc1 + (1.-alpha)*enc2
                dec_enc_mix = gan.generator.decode(enc_mix)
                if show_real:
                    if p == 0:
                        dec_enc_mix = x_batch_2
                    elif p == 1:
                        dec_enc_mix = x_batch_1
            pbuf.append(dec_enc_mix.detach().cpu())
            
    for b in range(x_batch_1.size(0)):
        this_interp = torch.stack([pbuf[i][b] for i in range(len(pbuf))])
        out_file = "%s/%i.png" % (out_folder, b)
        save_image( this_interp*0.5 + 0.5,
                    nrow=this_interp.size(0),
                    filename=out_file,
                    padding=padding)
        
def save_interp_supervised(gan, x_batch, y_batch,
                           out_folder,
                           num=10,
                           padding=2,
                           overlay_attrs=True,
                           enumerate_all=True):
    gan._eval()
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    import itertools
    pbuf = []
    padding = 0 if overlay_attrs else padding
    img_sz = None
    y_combinations = []

    def _rand_sample(ys):
        arr = []
        for i in range(len(ys)):
            arr.append(np.random.choice(ys[i]))
        return arr
    
    with torch.no_grad():
        if gan.use_cuda:
            x_batch = x_batch.cuda()
        enc = gan.generator.encode(x_batch)
        if img_sz is None:
            img_sz = x_batch.size(3)
        perm = torch.randperm(x_batch.size(0))
        # Ok, get the class of x1 and y1
        #y_batch, y_batch_perm
        for i in range(y_batch.size(0)):
            this_y1 = y_batch[i]
            this_y2 = y_batch[perm][i]
            print("Iteration: %i" % i)
            print("  this_y1 = ", this_y1)
            print("  this_y2 = ", this_y2)
            print("  sum(this_y1) =", sum(this_y1))
            print("  sum(this_y2) =", sum(this_y2))            
            # Produce all possible binary combinations between
            # this_y1 and this_y2.
            this_y_stacked = torch.stack((this_y1, this_y2))
            this_y_cols = [this_y_stacked[:,j].tolist() for j in range(len(this_y1))]
            if enumerate_all:
                # Get all combinations and then run it through a set to remove duplicates.
                this_all_combinations = set([elem for elem in itertools.product(*this_y_cols)])
                # Now sort the thing.
                this_all_combinations = sorted(this_all_combinations)
            else:
                this_all_combinations = sorted(set([ tuple(_rand_sample(this_y_cols)) for _ in range(20) ]))
            print("  tot combinations found = %i" % len(this_all_combinations))
            # Produce y_mix
            y_mix = torch.FloatTensor(this_all_combinations)
            if x_batch.is_cuda:
                y_mix = y_mix.cuda()
            this_x1 = x_batch[i].repeat(y_mix.size(0), 1, 1, 1)
            this_x2 = x_batch[perm][i].repeat(y_mix.size(0), 1, 1, 1)
            this_enc_x1 = gan.generator.encode(this_x1)
            this_enc_x2 = gan.generator.encode(this_x2)
            #this_enc_x1 = this_enc_x1.repeat(y_mix.size(0), 1, 1, 1)
            #this_enc_x2 = this_enc_x2.repeat(y_mix.size(0), 1, 1, 1)
            this_enc_mix, this_mask = gan.class_mixer(this_enc_x1, this_enc_x2, y_mix)
            print("  mask mean: ", this_mask.sum(dim=1).mean().item())
            this_dec_enc_mix = gan.generator.decode(this_enc_mix)
            this_all_imgs = torch.cat((x_batch[i:i+1], this_dec_enc_mix, x_batch[perm][i:i+1]), dim=0)
            #mixes.append( (this_all_imgs, this_all_combinations) )
            y_combinations.append([this_y1.tolist()] + this_all_combinations + [this_y2.tolist()])
            out_file = "%s/%i.png" % (out_folder, i)
            save_image( this_all_imgs*0.5 + 0.5,
                        nrow=this_all_imgs.size(0),
                        filename=out_file,
                        padding=padding)

            # **DEBUG**
            '''
            this_enc_mix, _ = gan.class_mixer(this_enc_x2, this_enc_x1, y_mix)
            this_dec_enc_mix = gan.generator.decode(this_enc_mix)
            this_all_imgs = torch.cat((x_batch[perm][i:i+1], this_dec_enc_mix, x_batch[i:i+1]), dim=0)
            #mixes.append( (this_all_imgs, this_all_combinations) )
            out_file = "%s/%i_flipped.png" % (out_folder, i)
            save_image( this_all_imgs*0.5 + 0.5,
                        nrow=this_all_imgs.size(0),
                        filename=out_file,
                        padding=padding)
            '''
            # **DEBUG**

    ############################################################
    # Read in each image saved and annotate it with the labels #
    ############################################################

    # For each image.
    if overlay_attrs:
        for b in range(len(y_combinations)):
            in_file = "%s/%i.png" % (out_folder, b)
            interp_img = imread(in_file)
            new_interp_img = None
            # For each face in that image.
            this_classes = y_combinations[b]
            for i in range(len(this_classes)):
                img_cell = interp_img[0:img_sz, i*img_sz:(i+1)*img_sz, :].copy()
                for j in range(len(this_classes[i])):
                    if this_classes[i][j] == 0:
                        # If it is zero, colour it red
                        img_cell[0:4, j*4:(j+1)*4, 0] *= 0
                        img_cell[0:4, j*4:(j+1)*4, 0] += 255
                        img_cell[0:4, j*4:(j+1)*4, 1] *= 0
                        img_cell[0:4, j*4:(j+1)*4, 2] *= 0
                    else:
                        # If it is one, colour it green
                        img_cell[0:4, j*4:(j+1)*4, 0] *= 0
                        img_cell[0:4, j*4:(j+1)*4, 1] *= 0
                        img_cell[0:4, j*4:(j+1)*4, 1] += 255
                        img_cell[0:4, j*4:(j+1)*4, 2] *= 0
                if new_interp_img is None:
                    new_interp_img = img_cell
                else:
                    new_interp_img = np.hstack((new_interp_img, img_cell))
            imsave(arr=new_interp_img, fname="%s/%i_anno.png" % (out_folder, b))
