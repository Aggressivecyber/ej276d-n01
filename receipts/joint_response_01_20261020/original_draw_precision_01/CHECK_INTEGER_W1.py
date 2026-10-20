from pathlib import Path
import hashlib,json,sys,time
import numpy as np
from scipy.stats import wasserstein_distance
import torch

root=Path('E:/GAN/research/N01')
here=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'src'))
from n01.joint01_integer_w1 import empirical_wasserstein_1d,original_empirical_w1
torch.set_num_threads(4)
rng=np.random.default_rng(202650280099)
records=[]
for n,m,d,tied in ((32,4,1,False),(504,63,9,False),(512,64,3,True),(32768,4096,32,False),(128,128,5,False),(129,63,3,False)):
    x=rng.normal(size=(n,d));y=rng.normal(.15,1.2,size=(m,d))
    if tied:x,y=np.round(x*2)/2,np.round(y*2)/2
    x,y=torch.from_numpy(x),torch.from_numpy(y)
    start=time.monotonic();old=original_empirical_w1(x,y);old_seconds=time.monotonic()-start
    start=time.monotonic();new=empirical_wasserstein_1d(x,y);new_seconds=time.monotonic()-start
    scipy=np.mean([wasserstein_distance(x[:,j].numpy(),y[:,j].numpy()) for j in range(d)])
    np.testing.assert_allclose(float(new),float(old),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(float(new),scipy,rtol=1e-12,atol=1e-12)
    records.append(dict(shape=[n,m,d],tied=tied,old=float(old),new=float(new),scipy=float(scipy),
        old_seconds=old_seconds,new_seconds=new_seconds,absolute_difference=abs(float(new-old))))
x,y=torch.randn(128,dtype=torch.float64),torch.randn(16,dtype=torch.float64)
np.testing.assert_allclose(float(empirical_wasserstein_1d(x,y)),wasserstein_distance(x.numpy(),y.numpy()),rtol=1e-12,atol=1e-12)
weight_checks=[]
for trainable in (False,True):
    weights=torch.linspace(.5,1.5,128,dtype=torch.float64,requires_grad=trainable)
    old,new=original_empirical_w1(x,y,weights),empirical_wasserstein_1d(x,y,weights)
    assert torch.equal(old,new)
    if trainable:
        old_grad=torch.autograd.grad(old,weights,retain_graph=True)[0]
        new_grad=torch.autograd.grad(new,weights)[0]
        assert torch.equal(old_grad,new_grad)
    weight_checks.append(dict(trainable=trainable,values_exact=True,gradient_exact=trainable))
report=dict(status='PASS_EXACT_8_TO_1_W1_LOCAL_ENGINEERING',cases=records,weight_checks=weight_checks,
    one_dimensional_value_pass=True,device='CPU',threads=4,new_generator_calls=0,scientific_updates=0,
    original_loss_source_sha256=hashlib.sha256((root/'src/n01/losses008_v2.py').read_bytes()).hexdigest(),
    optimized_operator_sha256=hashlib.sha256((root/'src/n01/joint01_integer_w1.py').read_bytes()).hexdigest(),
    note='Sorting the uniform 8:1 empirical measures pairs eight consecutive x order statistics with each y order statistic; input populations and conditional support counts are untouched. GPU native-readout equivalence remains to be checked.')
(here/'INTEGER_W1_ENGINEERING_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(report))
