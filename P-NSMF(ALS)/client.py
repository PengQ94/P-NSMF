import numpy as np
from numpy.linalg import pinv
import config as conf
from convert import float2int
import prng
from compress_matrix import compress_symmetric_matrix


class Client:
    def __init__(self):

        self.Iu = []  # items that interact with user u in train data
        self.Igroup = []  # items that are interacted by Gg to which u belongs
        self.Igroup_sub_Iu = []  # I'u, items in Igroup but are not interacted by user u
        self.idx_Iu = []  # index of Iu in Igroup

        self.itemVec_masked = None
        self.itemVec_mask = None

        self.U = None
        self.bU_u_masked = None
        self.AU_u_tri_masked = None
        self.SU_u_masked = None
        self.bU_u_mask = None
        self.AU_u_tri_mask = None
        self.SU_u_mask = None

        self.SV = None
        self.V = None

        self.Iu_test = []  # items that interact with user u in test data
        self.is_test = 0
        self.precision = 0
        self.recall = 0
        self.F1 = 0
        self.NDCG = 0
        self.oneCall = 0

    # --- PSU ---

    def download_Igroup(self, Igroup):
        self.Igroup = Igroup.tolist()
        self.Igroup_sub_Iu = np.setdiff1d(self.Igroup, self.Iu, assume_unique=True)

        for item in self.Iu:
            self.idx_Iu.append(self.Igroup.index(item))

    def upload_itemVecMask(self):
        return self.itemVec_mask

    def upload_itemVecMasked(self):
        return self.itemVec_masked

    def masking_itemVec(self):
        itemVec = np.zeros(conf.m, dtype=np.uint64)
        itemVec[self.Iu] = prng.generate_positiveInt8(len(self.Iu))

        itemVec_mask = prng.generate_mask(conf.m)
        itemVec_masked = itemVec + itemVec_mask

        return itemVec_masked, itemVec_mask

    def get_maskItemVec_result(self, results):
        self.itemVec_masked, self.itemVec_mask = results

    # --- Train ---

    def downlaod_para(self, SV, V_Igroup):
        self.SV = SV
        self.V = V_Igroup[self.idx_Iu]

    def upload_paraMask(self):
        return self.bU_u_mask, self.AU_u_tri_mask, self.SU_u_mask

    def upload_paraMasked(self):
        return self.bU_u_masked, self.AU_u_tri_masked, self.SU_u_masked

    def update_U(self):
        self.SV *= conf.alpha
        bV = np.sum(self.V, axis=0) * conf.omega
        AV = np.matmul(self.V.T, self.V) * (conf.omega - conf.alpha * conf.beta)
        self.U = np.matmul(bV, pinv(self.SV + AV + conf.m * conf.lambda_identity_matrix))

    def calParaForV(self):
        bU_u = np.empty([len(self.Iu), conf.d])
        AU_u_tri = np.empty([len(self.Iu), conf.dimension_tri])
        SU_u = np.outer(conf.alpha * self.U, self.U)

        U_outer_tri = compress_symmetric_matrix(np.outer(self.U, self.U), conf.d)
        for idx in range(len(self.Iu)):
            bU_u[idx] = self.U
            AU_u_tri[idx] = U_outer_tri
        bU_u *= conf.omega
        AU_u_tri *= conf.omega - conf.alpha * conf.beta

        return bU_u, AU_u_tri, SU_u

    def masking_para(self, bU_u_Iu, AU_u_tri_Iu, SU_u):
        bU_u_mask = prng.generate_mask([len(self.Igroup), conf.d])
        AU_u_tri_mask = prng.generate_mask([len(self.Igroup), conf.dimension_tri])
        SU_u_mask = prng.generate_mask(SU_u.shape)

        SU_u_conv = float2int(SU_u)
        SU_u_masked = SU_u_conv + SU_u_mask

        bU_u_Iu_conv = float2int(bU_u_Iu)
        bU_u_masked = bU_u_mask.copy()
        bU_u_masked[self.idx_Iu] += bU_u_Iu_conv

        AU_u_tri_conv = float2int(AU_u_tri_Iu)
        AU_u_tri_masked = AU_u_tri_mask.copy()
        AU_u_tri_masked[self.idx_Iu] += AU_u_tri_conv

        return (
            bU_u_masked,
            AU_u_tri_masked,
            SU_u_masked,
            bU_u_mask,
            AU_u_tri_mask,
            SU_u_mask,
        )

    def update(self):
        # update self.U
        self.update_U()
        # calculate bU_u, AU_u and SU_u for updating V in Central Server
        bU_u, AU_u_tri, SU_u = self.calParaForV()
        # mask bU_u, AU_u and SU_u
        (
            bU_u_masked,
            AU_u_tri_masked,
            SU_u_masked,
            bU_u_mask,
            AU_u_tri_mask,
            SU_u_mask,
        ) = self.masking_para(bU_u, AU_u_tri, SU_u)

        return (
            self.U,
            bU_u_masked,
            AU_u_tri_masked,
            SU_u_masked,
            bU_u_mask,
            AU_u_tri_mask,
            SU_u_mask,
        )

    def get_update_result(self, results):
        (
            self.U,
            self.bU_u_masked,
            self.AU_u_tri_masked,
            self.SU_u_masked,
            self.bU_u_mask,
            self.AU_u_tri_mask,
            self.SU_u_mask,
        ) = results

    # --- Evaluation ---

    def evaluate(self, V):

        if len(self.Iu_test) == 0:
            return

        pred = np.matmul(self.U, V.T)
        pred[self.Iu] = -(1<<20)
        item_topK = np.argpartition(pred, kth=-conf.topK)[-conf.topK:]  # unsorted topK
        pred_topK = pred[item_topK]
        idx_topK = np.argsort(-pred_topK)
        item_topK = list(item_topK[idx_topK])  # sorted topK

        hit = 0
        DCG = 0
        DCGbest = 0

        DCGbest_topK = min(len(self.Iu_test), conf.topK)
        for k in range(DCGbest_topK):
            DCGbest += 1 / np.log2(k + 2)

        for k in range(conf.topK):
            if item_topK[k] in self.Iu_test:
                hit += 1
                DCG += 1 / np.log2(k + 2)

        is_test = 1
        precision = hit / conf.topK
        recall = hit / len(self.Iu_test)
        F1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )
        NDCG = DCG / DCGbest
        oneCall = 1 if hit > 0 else 0

        return is_test, precision, recall, F1, NDCG, oneCall

    def get_evaluate_result(self, results):
        if results == None:
            return

        (
            self.is_test,
            self.precision,
            self.recall,
            self.F1,
            self.NDCG,
            self.oneCall,
        ) = results
        
    # clear cache
    def clear_cache(self):
        del self.bU_u_masked
        del self.AU_u_tri_masked
        del self.SU_u_masked
        del self.bU_u_mask
        del self.AU_u_tri_mask
        del self.SU_u_mask

    def clear_cache_psu(self):
        del self.itemVec_masked
        del self.itemVec_mask
