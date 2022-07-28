import numpy as np
from numpy.linalg import pinv
import config as conf
from convert import int2float
from compress_matrix import recover_symmetric_matrix


class CentralServer:
    def __init__(self):

        self.V = (np.random.rand(conf.m, conf.d) - 0.5) * 0.01
        self.SV = self.cal_SV()
        self.bU = None
        self.AU = np.empty([conf.m, conf.d, conf.d])
        self.SU = None

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

    def para_aggregation(self, clients, bU_mask, AU_tri_mask, SU_mask):

        bU_masked = np.zeros([conf.m, conf.d], dtype=np.uint64)
        AU_tri_masked = np.zeros([conf.m, conf.dimension_tri], dtype=np.uint64)
        SU_masked = np.zeros([conf.d, conf.d], dtype=np.uint64)

        for user in clients:
            bU_u_masked, AU_u_tri_masked, SU_u_masked = user.upload_paraMasked()

            SU_masked += SU_u_masked
            bU_masked[user.Igroup] += bU_u_masked
            AU_tri_masked[user.Igroup] += AU_u_tri_masked

        bU_conv = bU_masked - bU_mask
        AU_tri_conv = AU_tri_masked - AU_tri_mask
        SU_conv = SU_masked - SU_mask

        self.bU = int2float(bU_conv)
        AU_tri = int2float(AU_tri_conv)
        for item in range(conf.m):
            self.AU[item] = recover_symmetric_matrix(AU_tri[item], conf.d)
        self.SU = int2float(SU_conv)

    def update(self):

        # update V
        self.SU *= conf.beta
        lambda_identity_matrix = conf.n * conf.lambda_identity_matrix
        for item in range(conf.m):
            self.V[item] = np.matmul(self.bU[item], pinv(self.SU + self.AU[item] + lambda_identity_matrix))

        # update SV
        self.SV = self.cal_SV()
