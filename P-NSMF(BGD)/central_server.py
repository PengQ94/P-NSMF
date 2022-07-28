import numpy as np
import config as conf
from convert import int2float


class CentralServer:
    def __init__(self):

        self.V = (np.random.rand(conf.m, conf.d) - 0.5) * 0.01
        self.SV = self.cal_SV()
        self.b = None
        self.SU = None
        self.gradV = None
        self.gamma = conf.gamma

        self.Igroups = None

    # --- PSU ---

    def itemVec_aggregation(self, clients, itemVec_mask, groups):

        itemVec_masked = np.zeros([len(groups), conf.m], dtype=np.uint64)
        for g in range(len(groups)):
            for user in groups[g]:
                itemVec_masked[g] += clients[user].upload_itemVecMasked()

        itemVec = itemVec_masked - itemVec_mask
        Igroups = {}

        for g in range(len(itemVec)):
            Igroups[g] = np.nonzero(itemVec[g])[0]

        self.Igroups = Igroups

    # --- Train ---

    def cal_SV(self):
        SV = np.matmul(self.V.T, self.V) * conf.beta
        return SV

    def para_aggregation(self, clients, b_mask, SU_mask):

        b_masked = np.zeros([conf.m, conf.d], dtype=np.uint64)
        SU_masked = np.zeros([conf.d, conf.d], dtype=np.uint64)

        for user in clients:
            au_masked, SU_u_masked = user.upload_paraMasked()

            SU_masked += SU_u_masked
            b_masked[user.Igroup] += au_masked

        b_conv = b_masked - b_mask
        SU_conv = SU_masked - SU_mask

        self.b = int2float(b_conv)
        self.SU = int2float(SU_conv)

    def update(self):

        # update V
        gradV = np.matmul(self.V, conf.beta * self.SU)
        gradV = (gradV + self.b) * 2 / conf.n
        reg  = 2 * conf.lambda_ * self.V
        self.V -= self.gamma * (gradV + reg)

        self.gamma *= 0.999  # decay learning rate

        # update SV
        self.SV = self.cal_SV()
