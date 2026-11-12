"""KMNIST only; train/validation/test indices never share namespaces."""
import gzip, hashlib, json, struct, urllib.request
from pathlib import Path
import numpy as np
import torch

FILES = {
 'train-images-idx3-ubyte.gz': 'bdb82020997e1d708af4cf47b453dcf7',
 'train-labels-idx1-ubyte.gz': 'e144d726b3acfaa3e44228e80efcd344',
 't10k-images-idx3-ubyte.gz': '5c965bf0a639b31b8f53240b1b52f4d7',
 't10k-labels-idx1-ubyte.gz': '7320c461ea6c1c855c0b718fb2a4b134'}

def split_tensors(split, indices, train_x, train_y, test_x, test_y):
    if split == 'test': return test_x[indices], test_y[indices]
    if split in ('train','validation'): return train_x[indices], train_y[indices]
    raise ValueError('Unknown split: '+split)

def download(root):
    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    for name, digest in FILES.items():
        path = root/name
        if not path.exists():
            temp = path.with_suffix('.part')
            errors = []
            # Use a pinned mirror while CODH is intermittently unavailable.
            # Every file MUST match torchvision's original KMNIST checksum.
            for mirror in ['https://huggingface.co/datasets/tanganke/kmnist/resolve/c069cb84fc8b936f73c01c933a54fc57345a9c87/raw/']*3+[
                           'https://codh.rois.ac.jp/kmnist/dataset/kmnist/',
                           'http://codh.rois.ac.jp/kmnist/dataset/kmnist/']:
                try:
                    url=mirror+name+('?download=true' if 'huggingface.co' in mirror else '')
                    with urllib.request.urlopen(url, timeout=60) as response:
                        temp.write_bytes(response.read())
                    if hashlib.md5(temp.read_bytes()).hexdigest() != digest:
                        raise ValueError('KMNIST checksum mismatch')
                    break
                except Exception as exc:
                    errors.append(str(exc))
            else:
                raise RuntimeError('Download failed for '+name+': '+repr(errors))
            if hashlib.md5(temp.read_bytes()).hexdigest() != digest:
                raise ValueError('Download checksum failed: '+name)
            temp.replace(path)
        if hashlib.md5(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Wrong/corrupt KMNIST file: '+str(path))
    (root/'dataset_manifest.json').write_text(json.dumps({
        'dataset':'KMNIST','source':'https://github.com/rois-codh/kmnist',
        'mirror_revision':'tanganke/kmnist@c069cb84fc8b936f73c01c933a54fc57345a9c87',
        'files':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in FILES}
    },indent=2),encoding='utf-8')

def load(root, train):
    prefix = 'train' if train else 't10k'
    with gzip.open(Path(root)/(prefix+'-images-idx3-ubyte.gz'),'rb') as f:
        magic,n,h,w = struct.unpack('>IIII', f.read(16))
        assert magic == 2051 and h == w == 28
        x = np.frombuffer(f.read(),dtype=np.uint8).copy().reshape(n,1,28,28)
    with gzip.open(Path(root)/(prefix+'-labels-idx1-ubyte.gz'),'rb') as f:
        magic,n2 = struct.unpack('>II',f.read(8)); assert magic == 2049 and n2 == n
        y = np.frombuffer(f.read(),dtype=np.uint8).copy().astype(np.int64)
    assert n == (60000 if train else 10000) and len(y) == n
    assert np.array_equal(np.bincount(y,minlength=10), np.full(10,n//10))
    # Fixed data-independent scaling; no empirical private mean/std estimation.
    return torch.from_numpy(x).float()/127.5-1, torch.from_numpy(y)

def allocate(labels, indices, priors, size, rng):
    """Capacity-constrained Dirichlet label preferences, NOT standard unequal-size split."""
    k = len(priors); capacities = np.full(k,size,dtype=int)
    result = [[] for _ in range(k)]
    indices = rng.permutation(indices)
    if len(indices) != k*size: raise ValueError('Partition capacity mismatch')
    for idx in indices:
        weights = np.maximum(priors[:,int(labels[idx])],1e-12)*(capacities>0)
        client = int(rng.choice(k,p=weights/weights.sum()))
        result[client].append(int(idx)); capacities[client] -= 1
    return result

def partition(y_train,y_test,cfg):
    rng = np.random.default_rng(cfg['seed'])
    k = cfg['clients']
    priors = rng.dirichlet(np.full(10,cfg['alpha']),size=k)
    # Exactly 10% of original training data for validation, stratified by class.
    train_idx, val_idx = [], []
    for label in range(10):
        ids = rng.permutation(np.flatnonzero(np.asarray(y_train)==label))
        nval = len(ids)//10
        val_idx.extend(ids[:nval]); train_idx.extend(ids[nval:])
    tr = allocate(y_train,train_idx,priors,cfg['train_per_client'],rng)
    va = allocate(y_train,val_idx,priors,cfg['validation_per_client'],rng)
    te = allocate(y_test,np.arange(len(y_test)),priors,cfg['test_per_client'],rng)
    assert not set(sum(tr,[])) & set(sum(va,[]))
    assert len(set(sum(tr+va,[]))) == len(y_train)
    assert len(set(sum(te,[]))) == len(y_test)
    return {'train':tr,'validation':va,'test':te,'priors':priors.tolist(),
            'definition':'capacity-constrained Dirichlet label preferences; fixed membership conditional DP'}
