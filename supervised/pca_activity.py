import numpy as np
import matplotlib.pyplot as plt
import torch
from models import RNN
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.decomposition import PCA

def gather_inp_data():

    x_inp = []
    len_seq = []

    # may need to potentially give the rnn some time varying input as well? (ALM Data)
    for cond in range(3):

        x_inp.append(torch.tensor([cond/3], dtype=torch.float32).repeat(300).unsqueeze(1))
        len_seq.append(300)

    x_inp_total = torch.stack(x_inp, dim=0)
    
    return x_inp_total, len_seq

def main():
    
    inp_dim = 1
    hid_dim = 533
    out_dim = 1

    check_path = "checkpoints/rnn_goal_delay.pth"
    checkpoint = torch.load(check_path)
    
    rnn = RNN(inp_dim, hid_dim, out_dim)
    rnn.load_state_dict(checkpoint)

    activity_pca = PCA(n_components=2)

    x_inp, len_seq = gather_inp_data()

    with torch.no_grad():
        
        hn = torch.zeros(size=(1, x_inp.shape[0], hid_dim))
        _, _, act = rnn(x_inp, hn, len_seq)

    for batch in act:
        reduced_act = activity_pca.fit_transform(batch.numpy())
        plt.plot(reduced_act[:, 0])
        plt.plot(reduced_act[:, 1])
    plt.show()

    
if __name__ == "__main__":
    main()
