import unittest
import torch
from mbrl.core import WorldModel, Actor, Critic, frozen, imagine, lambda_returns, world_loss
from mbrl.adapter import Replay


class CoreTests(unittest.TestCase):
    def test_bootstrap_and_terminal(self):
        r=torch.tensor([[1.],[2.]])
        d=torch.tensor([[0.],[1.]])
        v=torch.tensor([[0.],[10.],[100.]])
        self.assertTrue(torch.allclose(lambda_returns(r,d,v,gamma=.9,lam=1),torch.tensor([[2.8],[2.]])))

    def test_actor_gradient_through_frozen_world_and_critic(self):
        torch.manual_seed(4)
        wm=WorldModel(14,2,3,2,4,widths=(16,16))
        actor=Actor(4,2,2);critic=Critic(wm)
        s=torch.randn(3,14);s[:,3:7]=torch.tensor([1.,0,0,0])
        h=torch.randn(3,2,4);g=torch.randn(4,3,3);c=torch.randn(3,2)
        before=[p.clone() for p in wm.parameters()]
        with frozen(wm,critic):
            loss,_=imagine(wm,actor,critic,s,h,g,c,torch.zeros(4))
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in actor.parameters()))
            self.assertTrue(all(p.grad is None for p in wm.parameters()))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,wm.parameters())))

    def test_fifo_contiguity(self):
        replay=Replay(5)
        for i in range(9): replay.add({'s':torch.full((5,1),float(i))})
        b=replay.sample(3,100)
        self.assertTrue((b['s'][1:]-b['s'][:-1]==1).all())
        self.assertGreaterEqual(b['s'].min(),4)

    def test_loss_masks_reset_boundary(self):
        wm=WorldModel(14,2,3,2,4,widths=(16,))
        b={k:torch.zeros(2,3,n) for k,n in [('s',14),('nxt',14),('a',2),('g',3),('c',2),('o',4),('onext',4),('gnext',3)]}
        for k in ['r','term','reset']: b[k]=torch.zeros(2,3)
        b['reset'][0]=1
        l,_=world_loss(wm,b)
        b['nxt'][1]=1e4;b['r'][1]=1e4
        l2,_=world_loss(wm,b)
        self.assertTrue(torch.allclose(l,l2))


if __name__=='__main__': unittest.main()
