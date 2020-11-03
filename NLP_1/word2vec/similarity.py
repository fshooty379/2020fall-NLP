import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

def get_word2vec(seg_list, aim_word):
    for i in seg_list:
        if i[0] == aim_word:
            i.pop(0)
            i = list(map(float, i))
            return np.array(i)

if __name__ == '__main__':
    file = open('output.txt', "r", encoding='UTF-8')
    text = file.readlines()
    file.close()
    seg_list = []  # 特征列表
    for i in text:
        i = i.strip('\n')  # 去除每一行的首尾空格
        b = i.split(' ')
        seg_list.append(b)
    seg_list.pop(0)  # 去掉第一行
    word_dic = {}
    for i in seg_list:
        name = i[0]
        i.pop(0)
        i = list(map(float, i))
        word_dic[name] = i

    # 计算中国与中华的相似度：
    vec1 = np.array(word_dic['中国'])
    vec2 = np.array(word_dic['中华'])
    print('计算中国与中华的相似度')
    print(cosine_similarity(vec1, vec2))

    # 计算与中国最相似的前五个词：
    aim_word = '中国'
    aim_vec = np.array(word_dic[aim_word])
    result = {}
    for i in word_dic:
        if i != aim_word:
            vec = np.array(word_dic[i])
            now_result = cosine_similarity(aim_vec, vec)
            result[i] = now_result
    result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    result = result[:5]
    print('计算与中国最相似的前五个词：')
    print(result)

    # 湖北+武汉-成都的相似度
    word1 = '湖北'
    word2 = '武汉'
    word3 = '成都'
    vec1 = np.array(word_dic[word1])
    vec2 = np.array(word_dic[word2])
    vec3 = np.negative(np.array(word_dic[word3]))
    result = {}
    for i in word_dic:
        if (i != word1) & (i != word2) & (i != word3):
            vec = np.array(word_dic[i])
            now_result = cosine_similarity(vec1+vec2+vec3, vec)
            result[i] = now_result
    result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    result = result[:5]
    print('湖北+武汉-成都的相似度')
    print(result)

    # 画图
    pca = PCA(n_components=2)
    cities = ['江苏', '南京', '成都', '四川', '湖北', '武汉', '河南', '郑州',
              '甘肃', '兰州', '湖南', '长沙', '陕西', '西安', '吉林', '长春',
              '广东', '广州', '浙江', '杭州']
    embeddings = []  # 词向量
    for i in cities:
        embeddings.append(word_dic[i])
    results = pca.fit_transform(embeddings)
    for i, j in zip(cities, results):  # 绘制散点图的名称标签
        plt.annotate(i, j, family="SimHei")
    sns.scatterplot(x=results[:, 0], y=results[:, 1])
    plt.show()