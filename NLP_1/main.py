from gensim.models import Word2Vec
import jieba
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    try:
        model = Word2Vec.load('word2vec.model')
    except Exception:
        file = open('exp1_corpus.txt', "r", encoding='UTF-8')
        text = file.readlines()
        file.close()
        seg_list = [] #特征列表
        for i in text:
            i = i.strip() #去除每一行的首尾空格
            word = jieba.cut(i, cut_all=False) #分词
            seg = [j for j in word] #形成该行的特征列表
            seg_list.append(seg) #得到整个语料库的特征列表
        model = Word2Vec(seg_list, size=100, window=5, min_count=1, workers=4)
        model.save("word2vec.model")

    print('鱼和水的相关性:'+str(model.wv.similarity('鱼', '水')))
    print('湖南和长沙的相关性比较:' + str(model.wv.similarity('湖南', '长沙')))
    print('和武汉最相似的5个词:\n'+str(model.wv.most_similar(positive=['武汉'], topn=5)))
    print('和沙漠最相似的5个词:\n' + str(model.wv.most_similar(positive=['沙漠'], topn=5)))
    print('使用训练好的词向量选出与指定词类比相似的5个词')
    print('旅游 + 机票 - 游客 :\n'+str(model.wv.most_similar(positive=['旅游', '机票'], negative=['游客'], topn=5)))
    print('广西 + 济南 - 南宁 :\n'+str(model.wv.most_similar(positive=['广西', '济南'], negative=['南宁'], topn=5)))

    pca = PCA(n_components=2)
    cities = ['江苏', '南京', '成都', '四川', '湖北', '武汉', '河南', '郑州',
              '甘肃', '兰州', '湖南', '长沙', '陕西', '西安', '吉林', '长春',
              '广东', '广州', '浙江', '杭州']
    embeddings = [] #词向量
    for i in cities:
        embeddings.append(model.wv[i])
    results = pca.fit_transform(embeddings)
    for i, j in zip(cities, results): #绘制散点图的名称标签
        plt.annotate(i, j , family = "SimHei")
    sns.scatterplot(x = results[:, 0], y = results[:, 1])
    plt.show()
