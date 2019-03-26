# --coding: utf-8 --
# by Haoze Sun 2019
# Stochastic EM for LDA
import numpy as np
import copy
import random
import time
import pickle as pkl


class SEM(object):
    @classmethod
    def read_data(cls, data_path, n=10000):
        W = []
        with open(data_path, 'rb') as f:
            for line in f.readlines()[:n]:
                idx = [int(x) for x in line.strip().split()]
                W.append(idx)
        return W

    def __init__(self, W, K=64, V=3000):
        self.W = W  # list of visible variables
        self.Z = copy.deepcopy(W)  # list of hidden variables
        self.M = len(W)
        self.K = K  # topic numbers
        self.V = V  # vocab
        self.alpha = np.ones([K], dtype=np.float) * 1.1
        self.beta = np.ones([V], dtype=np.float) * 1.1

        #  随机初始化参数
        self.phi = np.exp(np.random.normal(0.0, 0.01, [self.K, self.V]))
        self.phi /= np.sum(self.phi, axis=1, keepdims=True)
        self.theta = np.exp(np.random.normal(0.0, 0.01, [self.M, self.K]))
        self.theta /= np.sum(self.theta, axis=1, keepdims=True)
        # 统计量
        self.gamma = np.zeros([self.M, self.K], dtype=np.float)  # 统计每个doc中属于各主题的词数
        self.delta = np.zeros([self.K, self.V], dtype=np.float)  # 统计每个主题下对应的不同词个数
        print(self.K, self.V, self.M)

    def get_p_w(self):
        # E-step, for T(Z_ij | W_ij, \theta, \phi; \alpha, \beta) <== p(Z_ij | W_ij; \theta, \phi; \alpha, \beta)
        # return P(W_ij; \theta, \phi; \alpha, \beta) = \Sum_k p(Z_ij==k, W_ij; \theta, \phi; \alpha, \beta)
        self.p_w = np.matmul(self.theta, self.phi)  # (M, V)

    def sample_z(self):
        # S-step sample a topic for each word
        # 更新统计量，统计主题采样变化比例
        self.gamma = np.ones([self.M, self.K], dtype=np.float) * (self.alpha[None, :] - 1.0)
        self.delta = np.ones([self.K, self.V], dtype=np.float) * (self.beta[None, :] - 1.0)
        cnt_changes, total_words = 0, 0
        for i, doc in enumerate(self.W):
            for j, w_ij in enumerate(doc):
                total_words += 1
                # p(Z_ij | W_ij, \theta_i, \phi_{Z_ij})
                p_z_w = self.theta[i, :] * self.phi[:, w_ij] / self.p_w[i, w_ij]
                # sample a z_ij
                cum_p_z_w = np.cumsum(p_z_w, dtype=np.float)
                assert(abs(cum_p_z_w[-1] - 1.0) < 1e-4)
                z_ij = np.sum(cum_p_z_w <= np.random.random_sample())  # int
                if z_ij != self.Z[i][j]:
                    cnt_changes += 1
                self.Z[i][j] = z_ij
                self.gamma[i, z_ij] += 1.0
                self.delta[z_ij, w_ij] += 1.0
        return cnt_changes,  total_words

    def updata_param(self):
        # M-step 一次采样估计，最大化theta和phi, 先验分布Dir(theta; alpha), Dir(phi; beta)
        # 注意拉格朗日时s.t.\sum_k theta_ik = 1.0 and \sum_k phi_k = 1.0
        theta_new = self.gamma / np.sum(self.gamma, axis=1, keepdims=True)
        phi_new = self.delta / np.sum(self.delta, axis=1, keepdims=True)
        diff_theta = np.mean(np.abs(theta_new - self.theta) / (self.theta + 1e-6))
        diff_phi = np.mean(np.abs(phi_new - self.phi) / (self.phi + 1e-6))
        self.theta = theta_new
        self.phi = phi_new
        return diff_theta, diff_phi

    def train(self, max_epoch, queue_size, threshold=1e-4):
        assert (queue_size >= 3 and max_epoch > queue_size * 2)
        phi_queue, theta_queue, flag = [], [], 0
        for epoch in range(max_epoch):
            start = time.time()
            self.get_p_w()
            c, t = self.sample_z()
            diff_theta, diff_phi = self.updata_param()
            if len(phi_queue) >= queue_size:
                phi_queue.pop(0)
                theta_queue.pop(0)
            phi_queue.append(copy.deepcopy(self.phi))
            theta_queue.append(copy.deepcopy(self.theta))
            print("Sampling Epoch %d, changes: %d/%d, diff-theta: %f, diff-phi: %f, time: %f" %
                  (epoch, c, t, diff_theta, diff_phi, time.time() - start))
            if diff_phi < threshold and flag >= queue_size:
                break
            elif diff_phi < threshold:
                flag += 1
            else:
                pass
        self.phi = np.mean(np.asarray(phi_queue), axis=0)  # 采样-期望
        self.theta = np.mean(np.asarray(theta_queue), axis=0)

    def predict_topic(self, doc, max_epoch, queue_size, threshold=1e-4):
        # predict using exact inference using EM
        # 统计量由直接计数变为概率计数
        assert (queue_size >= 3 and max_epoch > queue_size * 2)
        theta_queue, flag = [], 0
        doc_theta = np.ones([self.K], dtype=np.float) / np.float(self.K)
        p_z_w = np.zeros([self.M, self.K], dtype=np.float)
        for epoch in range(max_epoch):
            p_w = np.matmul(doc_theta, self.phi)  # K, (K, v) -- > V
            for j, w in enumerate(doc):
                p_z_w[j, :] = doc_theta * self.phi[:, w] / p_w[w]
            doc_theta_new = np.sum(p_z_w, axis=0) + self.alpha - 1.0  # (K,)
            doc_theta_new /= np.sum(doc_theta_new)
            diff = np.mean(np.abs(doc_theta_new - doc_theta) / (doc_theta + 1e-6))
            doc_theta = doc_theta_new
            if len(theta_queue) >= queue_size:
                theta_queue.pop(0)
            theta_queue.append(copy.deepcopy(doc_theta))
            print("predict epoch: %d, diff: %f" % (epoch, diff))
            if diff < threshold and flag >= queue_size:
                break
            elif diff < threshold:
                flag += 1
            else:
                pass
        # 均值平滑
        doc_theta = np.mean(np.asarray(theta_queue), axis=0)
        return doc_theta


if __name__ == '__main__':
    sem = SEM(SEM.read_data('lda-small.txt', 2000), 50)
    sem.train(400, 10)
    pkl.dump(sem.phi, open('phi.pkl', 'wb'), protocol=2)

    train_theta = sem.theta[0, :]
    infer_theta = sem.predict_topic(sem.W[0], 100, 10)
    # check
    diff = np.max(np.abs(train_theta - infer_theta))
    print("check maximum diff: %f" % diff)
    print(train_theta)
    print(infer_theta)

