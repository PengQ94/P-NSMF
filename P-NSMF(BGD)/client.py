import numpy as np
import config as conf
from convert import float2int
import prng


class Client:
    def __init__(self):

        self.Iu = []  # items that interact with user u in train data
        self.Igroup = []  # items that are interacted by Gg to which u belongs
        self.Igroup_sub_Iu = []  # I'u, items in Igroup but are not interacted by user u
        self.idx_Iu = []  # index of Iu in Igroup

        self.itemVec_masked = None
        self.itemVec_mask = None

        self.U = (np.random.rand(conf.d) - 0.5) * 0.01
        self.au_masked = None
        self.SU_u_masked = None
        self.au_mask = None
        self.SU_u_mask = None

        self.SV = None
        self.V = None

        self.gamma = conf.gamma

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
        return self.au_mask, self.SU_u_mask

    def upload_paraMasked(self):
        return self.au_masked, self.SU_u_masked

    def update_U(self):
        pred = np.matmul(self.U, self.V.T)
        coe = (conf.omega - conf.alpha * conf.beta) * pred - conf.omega
        aV = np.matmul(coe, self.V)
        gradU = (aV + np.matmul(conf.alpha * self.U, self.SV)) * 2 / conf.m
        reg = 2 * conf.lambda_ * self.U
        self.U -= self.gamma * (gradU + reg)

        self.gamma *= 0.999  # decay learning rate

    def calParaForV(self):
        pred = np.matmul(self.U, self.V.T)
        au = (conf.omega - conf.alpha * conf.beta) * pred - conf.omega
        au = au.reshape(-1,1) * np.tile(self.U,(len(self.Iu),1))

        SU_u = np.outer(conf.alpha * self.U, self.U)

        return au, SU_u

    def masking_para(self, au_Iu, SU_u):
        au_mask = prng.generate_mask([len(self.Igroup), conf.d])
        SU_u_mask = prng.generate_mask(SU_u.shape)

        SU_u_conv = float2int(SU_u)
        SU_u_masked = SU_u_conv + SU_u_mask

        au_Iu_conv = float2int(au_Iu)
        au_masked = au_mask.copy()
        au_masked[self.idx_Iu] += au_Iu_conv

        return au_masked, SU_u_masked, au_mask, SU_u_mask

    def update(self):
        # update self.U
        self.update_U()
        # calculate au and SU_u for updating V in Central Server
        au, SU_u = self.calParaForV()
        # mask au and SU_u
        au_masked, SU_u_masked, au_mask, SU_u_mask = self.masking_para(au, SU_u)

        return self.U, au_masked, SU_u_masked, au_mask, SU_u_mask

    def get_update_result(self, results):
        self.U, self.au_masked, self.SU_u_masked, self.au_mask, self.SU_u_mask = results

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
        del self.au_masked
        del self.SU_u_masked
        del self.au_mask
        del self.SU_u_mask
    
    def clear_cache_psu(self):
        del self.itemVec_masked
        del self.itemVec_mask
