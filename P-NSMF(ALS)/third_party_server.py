import numpy as np
import config as conf


class ThirdPartyServer:
    def paraMask_aggregation(self, clients):
        bU_mask = np.zeros([conf.m, conf.d], dtype=np.uint64)
        AU_tri_mask = np.zeros([conf.m, conf.dimension_tri], dtype=np.uint64)
        SU_mask = np.zeros([conf.d, conf.d], dtype=np.uint64)

        for user in clients:
            bU_u_mask, AU_u_tri_mask, SU_u_mask = user.upload_paraMask()

            SU_mask += SU_u_mask
            bU_mask[user.Igroup] += bU_u_mask
            AU_tri_mask[user.Igroup] += AU_u_tri_mask

        return bU_mask, AU_tri_mask, SU_mask

    def itemVecMask_aggregation(self, clients, groups):
        itemVec_mask = np.zeros([len(groups), conf.m], dtype=np.uint64)
        for g in range(len(groups)):
            for user in groups[g]:
                itemVec_mask[g] += clients[user].upload_itemVecMask()

        return itemVec_mask
