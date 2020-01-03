#!/usr/bin/env python
# coding: utf-8

# ### 一、数据分析

# In[10]:
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from sklearn.cluster import MeanShift


def preprocesstitle(title):
    temp = ''
    for i in title:
        if i >= 'a' and i <= 'z':
            temp += i
        elif i >= 'A' and i <= 'Z':
            temp += i
        elif i == ' ':
            temp += i
        else:
            continue
    return temp


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# plt.show()

# print(len(authors))
# for author in train_data:
#     author_ids = train_data[author].keys()
#     print(author)
#     print(len(author_ids))

# In[16]:

test_row_data_path = 'sna_test_data/sna_test_data/sna_test_author_raw.json'
test_pub_data_path = 'sna_test_data/sna_test_data/test_pub_sna.json'

# 合并数据
test_pub_data = json.load(open(test_pub_data_path, 'r', encoding='utf-8'))
test_data = json.load(open(test_row_data_path, 'r', encoding='utf-8'))
merge_data = {}
for author in test_data:
    test_data[author] = [test_pub_data[paper_id] for paper_id in test_data[author]]

for paper in papers:
    authors = paper['authors']
    venues = paper['venue']
    years = paper['year']
    keywords = paper['keywords']

    venue_dict[venues] += 1
    year_dict[years] += 1
    for keyword in keywords:
        keywords_dict[keyword] += 1

    for paper_author in authors:
        name = paper_author['name']
        org = paper_author['org'] if 'org' in paper_author else ""
        org_dict[org] += 1

# 绘制该名称下论文数据情况
fig = plt.figure(figsize=(20, 20), dpi=80)

ax1 = fig.add_subplot(2, 2, 1)
x = range(5)
y = [len(papers), len(venue_dict), len(year_dict), len(keywords_dict), len(org_dict)]
s = ['涉及论文数量', '涉及期刊数量', '涉及年份数量', '涉及关键字数量', '涉及机构数量']

plt.bar(x, y, width=0.5)
plt.xticks(x, s, rotation=270)
plt.xlabel('%s论文数据情况' % author)
plt.ylabel('数量（个）')
for xl, yl in zip(x, y):
    plt.text(xl, yl + 0.3, str(yl), ha='center', va='bottom', fontsize=10.5)

ax2 = fig.add_subplot(2, 2, 2)
plt.bar(range(len(venue_dict)), venue_dict.values(), width=0.3)
plt.xlabel('%s期刊数据情况' % author)
plt.ylabel('数量（个）')

ax3 = fig.add_subplot(2, 2, 3)
plt.bar(range(len(year_dict)), year_dict.values(), width=0.5)
plt.xticks(range(len(year_dict)), year_dict.keys(), rotation=270)
plt.xlabel('%s年份数据情况' % author)
plt.ylabel('数量（个）')

ax4 = fig.add_subplot(2, 2, 4)
plt.bar(range(len(org_dict)), org_dict.values(), width=0.5)
plt.xlabel('%s机构数据情况' % author)
plt.ylabel('数量（个）')
# plt.show()

# In[29]:


# 查看论文作者名中是否包含消歧作者名
print(authors)
for author in validate_data:
    print('disambiguation name: ', author)
    papers = validate_data[author]
    for paper in papers[:10]:
        print('\npaper id: ' + paper['id'])
        authors = paper['authors']
        for paper_author in authors:
            name = paper_author['name']
            org = paper_author['org']
            print('paper author name: ', name)
            # print(org)

    break
'''
作者名存在不一致的情况：
1、大小写
2、姓、名顺序不一致
3、下划线、横线
4、简写与不简写
5、姓名有三个字的表达: 名字是否分开

同理：机构的表达也存在不一致的情况
因此：需要对数据做相应的预处理统一表达
'''

# In[12]:


import re


# 数据预处理

# 预处理名字
def precessname(name):
    name = name.lower().replace(' ', '_')
    name = name.replace('.', '_')
    name = name.replace('-', '')
    name = re.sub(r"_{2,}", "_", name)
    return name


# 预处理机构,简写替换，
def preprocessorg(org):
    if org != "":
        org = org.replace('Sch.', 'School')
        org = org.replace('Dept.', 'Department')
        org = org.replace('Coll.', 'College')
        org = org.replace('Inst.', 'Institute')
        org = org.replace('Univ.', 'University')
        org = org.replace('Lab ', 'Laboratory ')
        org = org.replace('Lab.', 'Laboratory')
        org = org.replace('Natl.', 'National')
        org = org.replace('Comp.', 'Computer')
        org = org.replace('Sci.', 'Science')
        org = org.replace('Tech.', 'Technology')
        org = org.replace('Technol.', 'Technology')
        org = org.replace('Elec.', 'Electronic')
        org = org.replace('Engr.', 'Engineering')
        org = org.replace('Aca.', 'Academy')
        org = org.replace('Syst.', 'Systems')
        org = org.replace('Eng.', 'Engineering')
        org = org.replace('Res.', 'Research')
        org = org.replace('Appl.', 'Applied')
        org = org.replace('Chem.', 'Chemistry')
        org = org.replace('Prep.', 'Petrochemical')
        org = org.replace('Phys.', 'Physics')
        org = org.replace('Phys.', 'Physics')
        org = org.replace('Mech.', 'Mechanics')
        org = org.replace('Mat.', 'Material')
        org = org.replace('Cent.', 'Center')
        org = org.replace('Ctr.', 'Center')
        org = org.replace('Behav.', 'Behavior')
        org = org.replace('Atom.', 'Atomic')
        org = org.split(';')[0]  # 多个机构只取第一个
    return org


# 正则去标点
def etl(content):
    content = re.sub("[\s+\.\!\/,;$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", " ", content)
    content = re.sub(r" {2,}", " ", content)
    return content


from sklearn.cluster import SpectralClustering


def get_org(co_authors, author_name):
    for au in co_authors:
        name = precessname(au['name'])
        name = name.split('_')
        if ('_'.join(name) == author_name or '_'.join(name[::-1]) == author_name) and 'org' in au:
            return au['org']
    return ''


# ### 二、解决方案：
#
# 1. 基于规则：利用文献之间和作者关系、机构关系，通过人为设定一些规则将待消歧文献归类到相应已有类簇中。
#
# 2. 无监督聚类：按照设定的相似度度量方法，计算待消歧数据集中所有样本彼此之间的相似度，得到样本间相似度矩阵，利用计算出的相似度矩阵进行聚类。
#
# 3. 半监督聚类：利用已标注数据数据集，构建二分类训练样本，即标签为两个文献是否属于同一个作者或者两者之间的距离。通过训练样本训练模型，得到样本之间的距离函数模型。根据已训练的模型在待消歧数据集的预测结果，即样本之间的距离矩阵，运用聚类算法得到最终的聚类类簇，也就是消歧后的结果。


# In[16]:

from sklearn.cluster import Birch
# 3. 无监督聚类（根据合作者和机构TFIDF进行相似度聚类） 线上得分：0.2637
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

words = 'a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other,our,own,rather,said,say,says,she,should,since,so,some,than,that,the,their,them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,with,would,yet,you,your'
stop = str.split(words, ',')


def disambiguate_by_cluster():
    res_dict = {}
    for author in validate_data:
        print(author)
        coauther_orgs = {}
        papers = validate_data[author]
        if len(papers) == 0:
            res_dict[author] = []
            continue
        print(len(papers))
        paper_dict = {}
        for paper in papers:
            d = {}
            authors = paper['authors']
            names = [precessname(paper_author['name']) for paper_author in authors]
            orgs = [preprocessorg(paper_author['org']) for paper_author in authors if 'org' in paper_author]
            abstract = paper["abstract"] if 'abstract' in paper else ''
            venue = paper["venue"] if 'venue' in paper else ''
            title = etl(paper["title"]) if 'title' in paper else ''
            year = paper["year"] if 'year' in paper else ''
            d["name"] = names
            d["org"] = org
            d['keywords'] = paper['keywords'] if 'keywords' in paper else ""
            d['venue'] = venue
            st = ''
            if 'keywords' in paper:
                for key in paper['keywords']:
                    st += key
            coauther_orgs[paper['id']] = d  # etl(' '.join(names + orgs)) + ' ' + d

        tfidf = TfidfVectorizer().fit_transform(coauther_orgs.values())

        clf = DBSCAN(metric='cosine')
        s = clf.fit(tfidf)

        # 每个样本所属的簇
        for label, paper in zip(clf.labels_, papers):
            if str(label) not in paper_dict:
                paper_dict[str(label)] = [paper['id']]
            else:
                paper_dict[str(label)].append(paper['id'])
        res_dict[author] = list(paper_dict.values())
    json.dump(res_dict, open('result/result.json', 'w', encoding='utf-8'), indent=4)


# disambiguate_by_cluster()
def xiugai():
    test = 'result/0.36.json'
    test_data = json.load(open(test, 'r', encoding='utf-8'))
    # print (test_data.keys())

    res_dict = {}
    for key, value in test_data.items():
        paper_dict = {}
        i = 0
        paper_dict[i] = []

        for papers in test_data[key]:
            if len(papers) >= 8 or i == 0:
                paper_dict[i] = papers
                i += 1
            else:
                if len(paper_dict[i - 1]) > 8:
                    paper_dict[i] = []
                    paper_dict[i] = papers
                    i += 1
                else:
                    for paper in papers:
                        paper_dict[i - 1].append(paper)
        res_dict[key] = list(paper_dict.values())
    json.dump(res_dict, open('result/xiugai36.json', 'w', encoding='utf-8'), indent=4)


xiugai()


def disambiguate_by_coauthor():
    res_dict = {}

    for author in test_data:
        print(author)
        res = []
        papers = test_data[author]
        print(len(papers))
        paper_dict = {}
        for paper in papers:
            d = {}
            authors = [precessname(paper_author['name']) for paper_author in paper['authors']]
            if author in authors:
                authors.remove(author)
            org = preprocessorg(get_org(paper['authors'], author))
            venue = paper['venue']
            d["authors"] = authors
            d["org"] = org
            d['keywords'] = paper['keywords'] if 'keywords' in paper else ""
            d['venue'] = venue

            if len(res) == 0:
                res.append([paper['id']])
            else:
                max_inter = 0
                indx = 0
                for i, clusters in enumerate(res):
                    score = 0
                    for pid in clusters:
                        insection = set(paper_dict[pid]['authors']) & set(authors)
                        score += len(insection)

                    if score > max_inter:
                        max_inter = score
                        indx = i

                if max_inter > 0:
                    res[indx].append(paper['id'])  # 若最高分大于0，将id添加到得分最高的簇中
                else:
                    res.append([paper['id']])  # 否则，另起一簇

            paper_dict[paper['id']] = d

        res_dict[author] = res
    json.dump(res_dict, open('result/zuihouyici.json', 'w', encoding='utf-8'), indent=4)

# disambiguate_by_coauthor()
