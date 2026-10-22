"""Appendix D.1 reset mixture; running-return estimator is a documented adaptation."""
import torch


class ReturnSampler:
    def __init__(self, frames, dt, device, gamma=.99):
        self.frames, self.steps = frames, round(1/dt)
        self.gamma, self.gmax = gamma, 5*dt/(1-gamma)
        n = (frames+self.steps-1)//self.steps
        self.values = torch.full((n,), self.gmax, device=device)
        self.past = torch.ones(n, device=device)/n
        self.updates = 0
        self.pending = []

    def probabilities(self):
        hard = torch.softmax(-25*self.values/self.gmax, 0)
        return .5/len(hard) + .4*hard + .1*self.past/self.past.sum()

    def sample(self, count):
        bins = torch.multinomial(self.probabilities(), count, replacement=True)
        widths = (self.frames-bins*self.steps).clamp(max=self.steps)
        return bins*self.steps + (torch.rand(count, device=bins.device)*widths).long()

    def add(self, frame, reward, reset, next_frame):
        self.pending.append((frame.long(), reward.clone(), reset.clone(), next_frame.long()))
        if len(self.pending)>=128: self.flush()

    def flush(self):
        if not self.pending: return
        # Bootstrap truncated 128-step tails from running bin values; true episode
        # boundaries have zero bootstrap. This estimator detail is not specified
        # in the paper; it is kept explicit rather than called an exact replication.
        ret = self.values[(self.pending[-1][3]//self.steps).clamp_max(len(self.values)-1)]
        totals, counts = torch.zeros_like(self.values), torch.zeros_like(self.values)
        for frame,reward,reset,_ in reversed(self.pending):
            ret = reward+self.gamma*(1-reset)*ret
            bins = (frame//self.steps).clamp_max(len(self.values)-1)
            totals.scatter_add_(0,bins,ret)
            counts.scatter_add_(0,bins,torch.ones_like(ret))
        mask = counts>0
        self.values[mask] = .9*self.values[mask]+.1*totals[mask]/counts[mask]
        self.past += torch.softmax(-25*self.values/self.gmax,0)
        self.updates += 1
        self.pending.clear()

    def state_dict(self):
        return dict(values=self.values, past=self.past, updates=self.updates)
