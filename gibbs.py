# --coding: utf-8 --
# by Haoze Sun 2019
# Gibbs Sampling for LDA
import numpy as np
import random
import time
import pickle as pkl
import copy

class Gibbs(object):
    @classmethod
    def read_data(cls, data_path, n=10000):
        W = []
        with open(data_path, 'rb') as f:
            for line in f.readlines()[:n]:
                idx = [int(x) for x in line.strip().split()]
                W.append(idx)
        return W

    def __init__(self, W, K=64, V=3000):
        # ref https://blog.csdn.net/yangliuy/article/details/8302599
        # ref https://www.cnblogs.com/pinard/p/6867828.html
        self.W = W  # list of visible variables
        self.Z = copy.deepcopy(W)  # list of hidden variables
        self.M = len(W)
        self.K = K  # topic numbers
        self.V = V  # vocab
        self.alpha = np.ones([K], dtype=np.float) * 0.1
        self.beta = np.ones([V], dtype=np.float) * 0.01

        self.gamma = np.zeros([self.M, self.K], dtype=np.float)  # 统计每个doc中属于各主题的词数
        self.delta = np.zeros([self.K, self.V], dtype=np.float)  # 统计每个主题下对应的不同词个数

        # initialization 随机分配topic
        print("random topic initialization....")
        self.total_words = 0
        for i, z_i in enumerate(self.Z):
            for j in range(len(z_i)):
                self.total_words += 1
                self.Z[i][j] = random.randint(0, self.K - 1)
                self.gamma[i, self.Z[i][j]] += 1.0
                self.delta[self.Z[i][j], self.W[i][j]] += 1.0

        self.gamma += self.alpha[None, :]  # 先验正则量
        self.delta += self.beta[None, :]

        self.gamma_sum = np.sum(self.gamma, axis=1)  # (M,)
        self.delta_sum = np.sum(self.delta, axis=1)  # (K, )

        print(self.K, self.V, self.M, self.total_words)

    def gibbs_sampling(self):
        cnt_change = 0
        for i, z_i in enumerate(self.Z):
            for j in range(len(z_i)):
                # 每步从统计量中先减去当前词对应的部分，计算相应的mcmc条件转义概率
                current_topic = self.Z[i][j]
                self.gamma[i, current_topic] -= 1.0
                self.delta[current_topic, self.W[i][j]] -= 1.0
                self.gamma_sum[i] -= 1.0
                self.delta_sum[current_topic] -= 1.0
                # 计算转移概率
                transfer_prob = self.gamma[i, :] * self.delta[:, self.W[i][j]] / self.delta_sum  # (K, )
                # normalize
                transfer_prob /= np.sum(transfer_prob)
                cum_prob = np.cumsum(transfer_prob)
                assert(abs(cum_prob[-1] - 1.0) < 1e-4)
                new_topic = np.sum(cum_prob <= np.random.random_sample())
                self.Z[i][j] = new_topic  # 新topic赋值

                if new_topic != current_topic:
                    cnt_change += 1

                # 恢复全部统计量
                self.gamma[i, new_topic] += 1.0
                self.delta[new_topic, self.W[i][j]] += 1.0
                self.gamma_sum[i] += 1.0
                self.delta_sum[new_topic] += 1.0
        return cnt_change

    def train(self, max_epoch, queue_size, threhold=1e-2):
        assert(queue_size >= 5 and max_epoch > queue_size * 2)
        self.delta_queue, self.gamma_queue, tmp_theta, tmp_phi = [], [], 0.0, 0.0
        for epoch in range(max_epoch):
            start = time.time()
            cnt_change = gbs.gibbs_sampling()

            if len(self.delta_queue) >= queue_size:
                self.delta_queue.pop(0)
                self.gamma_queue.pop(0)
            self.delta_queue.append(copy.deepcopy(self.delta))
            self.gamma_queue.append(copy.deepcopy(self.gamma))

            self.delta_avg = np.mean(np.asarray(self.delta_queue), axis=0)
            self.gamma_avg = np.mean(np.asarray(self.gamma_queue), axis=0)
            self.phi = self.delta_avg / np.sum(self.delta_avg, axis=1, keepdims=True)
            self.theta = self.gamma / np.sum(self.gamma, axis=1, keepdims=True)
            diff_theta = np.mean(np.abs(tmp_theta - self.theta) / (self.theta + 1e-6))
            diff_phi = np.mean(np.abs(tmp_phi - self.phi) / (self.phi + 1e-6))

            tmp_theta = copy.deepcopy(self.theta)
            tmp_phi = copy.deepcopy(self.phi)
            print("Sampling Epoch %d, changes: %d/%d, diff-theta: %f, diff-phi: %f, time: %f" %
                  (epoch, cnt_change, gbs.total_words, diff_theta, diff_phi, time.time() - start))
            if float(cnt_change) / float(self.total_words) < threhold:
                break


    def predict_topic(self, doc, max_epoch, queue_size=10):
        assert (queue_size >= 5 and max_epoch > queue_size * 2)
        doc_z = copy.deepcopy(doc)
        doc_gamma = copy.deepcopy(self.alpha)  # 直接使用训练好的phi 不改动delta
        gamma_queue, tmp_theta = [], 0.0
        for i, w in enumerate(doc):
            doc_z[i] = random.randint(0, self.K - 1)
            doc_gamma[doc_z[i]] += 1.0

        for epoch in range(max_epoch):
            start = time.time()
            for j, current_topic in enumerate(doc_z):
                doc_gamma[current_topic] -= 1.0

                # transfer prob
                transfer_prob = doc_gamma * self.phi[:, doc[j]]
                # normalize
                transfer_prob /= np.sum(transfer_prob)
                cum_prob = np.cumsum(transfer_prob)
                assert (abs(cum_prob[-1] - 1.0) < 1e-4)
                new_topic = np.sum(cum_prob <= np.random.random_sample())
                doc_z[j] = new_topic
                # recover
                doc_gamma[new_topic] += 1.0

            if len(gamma_queue) >= queue_size:
                gamma_queue.pop(0)
            gamma_queue.append(copy.deepcopy(doc_gamma))
            gamma_avg = np.mean(np.asarray(gamma_queue), axis=0)
            theta = gamma_avg / np.sum(gamma_avg)
            diff = np.mean(np.abs(tmp_theta - theta) / theta)
            tmp_theta = copy.deepcopy(theta)
            print("Inference Epoch %d, diff: %f, time: %f" % (epoch, diff, time.time() - start))

        return theta

if __name__ == '__main__':
    gbs = Gibbs(Gibbs.read_data('lda-small.txt', 2000), 20)
    gbs.train(400, 20)
    pkl.dump(gbs.phi, open('phi.pkl', 'wb'), protocol=2)
    train_theta = gbs.theta[0, :]
    infer_theta = gbs.predict_topic(gbs.W[0], 100, 20)
    # check
    diff = np.max(np.abs(train_theta - infer_theta))
    print("check maximum diff: %f" % diff)
    print(train_theta)
    print(infer_theta)










