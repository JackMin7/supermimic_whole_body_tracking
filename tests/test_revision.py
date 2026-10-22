import unittest
import torch
from mbrl.core import Actor,WorldModel,proprio,rollout_support
from mbrl.sampling import ReturnSampler


class MBRLTests(unittest.TestCase):
    def test_goal_encoder_is_bounded_and_uses_future(self):
        actor=Actor(93,29,10,ng=4,goal_frames=10)
        goal=torch.randn(3,10,4,requires_grad=True)
        actor(torch.randn(3,10,93),goal).sum().backward()
        self.assertGreater(goal.grad[:, -1].abs().sum().item(),0)
        z=actor.encoder(actor.goal_normalizer(goal).flatten(1)).tanh()
        self.assertEqual(z.shape,(3,32));self.assertLessEqual(z.abs().max().item(),1.)

    def test_analytic_observation_and_previous_action(self):
        s=torch.zeros(2,100);s[:,3]=1;s[:,7:36]=.3;s[:,42:71]=.4
        s[:,39:42]=.2;s[:,-29:]=.7
        obs=proprio(s,torch.full((29,),.1),torch.full((2,29),.01))
        self.assertEqual(obs.shape,(2,93))
        self.assertTrue(torch.allclose(obs[:,:29],torch.full((2,29),.21)))
        self.assertTrue(torch.equal(obs[:,61:64],torch.tensor([[0.,0.,-1.]]).expand(2,-1)))
        self.assertTrue(torch.equal(obs[:,-29:],s[:,-29:]))

    def test_bad_sample_does_not_discard_healthy_peer(self):
        wm=WorldModel(14,2,3,2,4,widths=(8,))
        actor=Actor(4,2,1)
        s=torch.zeros(2,14);s[:,3]=1;s[1,0]=1000
        h=torch.zeros(2,1,4);g=torch.zeros(3,2,3);c=torch.zeros(2,2)
        valid=rollout_support(wm,actor,s,h,g,c,torch.zeros(2,2,4),50)
        self.assertEqual(valid.tolist(),[True,False])

    def test_sampler_distribution_and_terminal_return(self):
        sam=ReturnSampler(125,.02,'cpu')
        sam.values[:]=torch.tensor([1.,5.,9.]);p=sam.probabilities()
        self.assertAlmostEqual(float(p.sum()),1.,places=6)
        self.assertGreater(p[0],p[2]);self.assertTrue((p>=.5/3).all())
        idx=sam.sample(1000);self.assertTrue(((idx>=0)&(idx<125)).all())
        old=sam.values.clone()
        sam.add(torch.tensor([0]),torch.tensor([2.]),torch.tensor([1.]),torch.tensor([50]))
        sam.flush()
        self.assertAlmostEqual(float(sam.values[0]),float(.9*old[0]+.1*2),places=5)


if __name__=='__main__':unittest.main()
