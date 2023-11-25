import torch
import torch.nn as nn
import torch.optim as optim
from sac_model import Actor
import scipy.io
import numpy as np
import torch.nn.functional as F

class ALM(nn.Module):
    def __init__(self, action_dim, alm_hid):
        super(ALM, self).__init__()
        self.action_dim = action_dim
        self.alm_hid = alm_hid
        self._alm = nn.RNN(action_dim, alm_hid, batch_first=True, nonlinearity='relu')
        self._alm_out = nn.Linear(alm_hid, 3)
    
    def forward(self, x, hn):
        activity, hn = self._alm(x, hn)
        activity = F.hardtanh(self._alm_out(activity), min_val=0, max_val=1)
        return activity, hn

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_next_act(actor, alm, state, hn, alm_hn, alm_activity_arr):
    action, _, _, hn, _ = actor.sample(state, hn, sampling=True)
    alm_out, alm_hn = alm(action, alm_hn)
    next_state = torch.cat((alm_hn, alm_out, alm_activity_arr), dim=1)
    return action, hn, alm_hn, next_state

def main():

    alm_activity = scipy.io.loadmat("alm_warped_activity_3pcs_1slick.mat")
    alm_activity_arr = alm_activity["warped_activity_3pcs_1slick"]
    alm_activity_arr = NormalizeData(alm_activity_arr)
    alm_activity_arr = torch.tensor(alm_activity_arr).to(torch.float32)

    INP_DIM = 70
    HID_DIM = 256
    ACTION_DIM = 8
    ALM_HID = 64
    EPOCHS = 1000
    LR = 0.001
    TIMESTEPS = 10

    alm = ALM(ACTION_DIM, ALM_HID).cuda()

    optimizer = optim.Adam(alm.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):

        state = torch.cat((torch.zeros(size=(67,)), alm_activity_arr[0,:])).unsqueeze(0).cuda().to(torch.float32)
        hn = torch.zeros(size=(1, HID_DIM)).cuda()
        alm_hn = torch.zeros(size=(1, ALM_HID)).cuda()

        with torch.no_grad():
            all_actions = []
            rand_actor = Actor(INP_DIM, HID_DIM, ACTION_DIM).cuda()
            for t in range(1, TIMESTEPS+1):
                action, hn, alm_hn, state = get_next_act(rand_actor, alm, state, hn, alm_hn, alm_activity_arr[t,:].cuda().unsqueeze(0))
                all_actions.append(action)
            all_actions = torch.cat(all_actions)

        alm_hn = torch.zeros(size=(1, ALM_HID)).cuda()
        alm_out, alm_hn = alm(all_actions, alm_hn)
        loss = criterion(alm_out, alm_activity_arr[:TIMESTEPS,:].cuda().unsqueeze(0))
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save({
        'agent_state_dict': alm.state_dict(),
    }, 'checkpoints/alm_init.pth')

if __name__ == "__main__":
    main()