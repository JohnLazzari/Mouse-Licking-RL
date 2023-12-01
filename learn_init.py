import torch
import torch.nn as nn
import torch.optim as optim
from sac_model import Actor
import scipy.io
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ALM(nn.Module):
    def __init__(self, action_dim, alm_hid):
        super(ALM, self).__init__()
        self.action_dim = action_dim
        self.alm_hid = alm_hid
        self._alm_in = nn.Linear(action_dim, alm_hid)
        self._alm = nn.RNN(alm_hid, alm_hid, batch_first=True, nonlinearity='tanh')
        self._alm_out = nn.Linear(alm_hid, 3)

    def forward(self, x, hn):
        x = F.relu(self._alm_in(x))
        activity, hn = self._alm(x, hn)
        activity = F.relu(self._alm_out(activity))
        return activity, hn


def NormalizeData(data):
    return ((data - np.min(data)) / (np.max(data) - np.min(data)))

def get_next_act(actor, alm, state, hn, alm_hn, alm_activity_arr):
    _, _, action, hn, _ = actor.sample(state, hn, sampling=True)
    alm_out, alm_hn = alm(action, alm_hn)
    next_state = torch.cat((alm_hn, alm_out, alm_activity_arr), dim=1)
    return action, hn, alm_hn, next_state

def main():

    alm_activity = scipy.io.loadmat("alm_warped_activity_3pcs_1slick.mat")
    alm_activity_arr = alm_activity["warped_activity_3pcs_1slick"]
    alm_activity_arr = NormalizeData(alm_activity_arr)
    alm_activity_arr = torch.tensor(alm_activity_arr, dtype=torch.float32, device="cuda")

    INP_DIM = 262
    HID_DIM = 256
    ACTION_DIM = 8
    ALM_HID = 256
    EPOCHS = 10000
    LR = 0.001
    TIMESTEPS = alm_activity_arr.shape[0]

    criterion = nn.MSELoss()

    alm = ALM(ACTION_DIM, ALM_HID).cuda()
    rand_actor = Actor(INP_DIM, HID_DIM, ACTION_DIM).cuda()
    optimizer = optim.Adam(alm.parameters(), lr=LR)

    all_actions = []

    alm_hn = torch.zeros(size=(1, ALM_HID)).cuda()
    actor_hn = torch.zeros(size=(1, HID_DIM)).cuda()

    state = torch.cat((alm_hn.squeeze(), torch.zeros((3,)).cuda(), alm_activity_arr[0,:])).unsqueeze(0).to(torch.float32)

    with torch.no_grad():
        for t in range(1, TIMESTEPS):
            action, actor_hn, alm_hn, state = get_next_act(rand_actor, alm, state, actor_hn, alm_hn, alm_activity_arr[t,:].unsqueeze(0))
            all_actions.append(action)
        all_actions = torch.cat(all_actions, dim=0)

    for epoch in range(EPOCHS):

        alm_hn = torch.zeros(size=(1, ALM_HID)).cuda()
        alm_out, _ = alm(all_actions, alm_hn)
        loss = criterion(alm_out, alm_activity_arr[:-1])
        print("epoch", epoch, "loss", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    alm_out = alm_out.cpu().detach().numpy()
    alm_activity_arr = alm_activity_arr.cpu().detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(alm_activity_arr[:,0], alm_activity_arr[:,1], alm_activity_arr[:,2])
    ax.scatter(alm_out[:,0], alm_out[:,1], alm_out[:,2])
    plt.show()

    torch.save({
        'alm_state_dict': alm.state_dict()
    }, 'checkpoints/alm_init.pth')

if __name__ == "__main__":
    main()