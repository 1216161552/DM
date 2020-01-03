#!/usr/bin/env python
# coding: utf-8

# ### 一、数据分析

# In[10]:


import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# In[2]:


# 训练集分析
train_row_data_path = 'train/train/train_author.json'
train_pub_data_path = 'train/train/train_pub.json'

train_pub_data = json.load(open(train_pub_data_path, 'r', encoding='utf-8'))
train_data = json.load(open(train_row_data_path, 'r', encoding='utf-8'))
authors = [author for author in train_data]
authors_num_person = [len(train_data[author].keys()) for author in train_data]

print('训练集同名数量：', len(authors))
print('消歧后实际作者数量：', sum(authors_num_person))

# 绘制训练集同名作者个体数量
plt.figure(figsize=(40, 8), dpi=80)
x = range(len(authors))

plt.bar(x, authors_num_person, width=0.5)
plt.xticks(x, authors)
plt.xticks(rotation=270)
plt.xlabel('训练集同名作者')
plt.ylabel('该名字同名作者数量（个）')
for xl, yl in zip(x, authors_num_person):
    plt.text(xl, yl + 0.3, str(yl), ha='center', va='bottom', fontsize=10.5)

mean_person = int(np.mean(authors_num_person))
plt.gca().hlines(mean_person, -1, 225, linestyles='--', colors='red', label='平均值')
plt.annotate(u"平均值:" + str(mean_person), xy=(225, mean_person), xytext=(225, mean_person + 40),
             arrowprops=dict(facecolor='red', shrink=0.1, width=2))

# plt.show()

# print(len(authors))
# for author in train_data:
#     author_ids = train_data[author].keys()
#     print(author)
#     print(len(author_ids))

# In[16]:


# 绘制训练集同名作者论文总数
authors_num_papers = []
for author in train_data:
    num = 0
    for author_id in train_data[author]:
        papers = train_data[author][author_id]
        num += len(papers)
    authors_num_papers.append(num)

plt.figure(figsize=(40, 8), dpi=80)
x = range(len(authors))

plt.bar(x, authors_num_papers, width=0.5)
plt.xticks(x, authors)
plt.xticks(rotation=270)
plt.xlabel('训练集同名作者')
plt.ylabel('该名字论文总数（篇）')
for xl, yl in zip(x, authors_num_papers):
    plt.text(xl, yl + 0.3, str(yl), ha='center', va='bottom', fontsize=10.5)

mean_person = int(np.mean(authors_num_papers))
plt.gca().hlines(mean_person, -1, 225, linestyles='--', colors='red', label='平均值')
plt.annotate(u"平均值:" + str(mean_person), xy=(225, mean_person), xytext=(225, mean_person + 40),
             arrowprops=dict(facecolor='red', shrink=0.1, width=2))

# plt.show()

# In[11]:


valid_row_data_path = 'sna_test_data/sna_test_data/sna_test_author_raw.json'
valid_pub_data_path = 'sna_test_data/sna_test_data/test_pub_sna.json'

# 合并数据
validate_pub_data = json.load(open(valid_pub_data_path, 'r', encoding='utf-8'))
validate_data = json.load(open(valid_row_data_path, 'r', encoding='utf-8'))
merge_data = {}
for author in validate_data:
    validate_data[author] = [validate_pub_data[paper_id] for paper_id in validate_data[author]]

# 验证集数据分析
authors = validate_data.keys()
papers_perauthor = [len(validate_data[author]) for author in validate_data]
print('同名作者数量：', len(authors))
print('涉及的论文数：', np.sum(papers_perauthor))
print('平均论文数量：', np.mean(papers_perauthor))
print('提供的论文数：', len(validate_pub_data))

# 绘制同名作者论文数量
plt.figure(figsize=(20, 8), dpi=80)
x = range(len(authors))

plt.bar(x, papers_perauthor, width=0.8)
plt.xticks(x, authors)
plt.xticks(rotation=270)
plt.xlabel('测试集同名作者')
plt.ylabel('测试集论文数量（篇）')
for xl, yl in zip(x, papers_perauthor):
    plt.text(xl, yl + 0.3, str(yl), ha='center', va='bottom', fontsize=10.5)

plt.gca().hlines(np.mean(papers_perauthor), -1, 50, linestyles='--', colors='red', label='平均值')
plt.annotate(u"平均值", xy=(0, np.mean(papers_perauthor)), xytext=(0, 1400),
             arrowprops=dict(facecolor='red', shrink=0.1, width=2))

# plt.show()

# In[112]:


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


def get_org(co_authors, author_name):
    for au in co_authors:
        name = precessname(au['name'])
        name = name.split('_')
        if ('_'.join(name) == author_name or '_'.join(name[::-1]) == author_name) and 'org' in au:
            return au['org']
    return ''


# 3. 无监督聚类（根据合作者和机构TFIDF进行相似度聚类） 线上得分：0.2637
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer


def disambiguate_by_cluster():
    res_dict = {}
    for author in validate_data:
        print(author)
        coauther_orgs = []
        papers = validate_data[author]
        if len(papers) == 0:
            res_dict[author] = []
            continue
        print(len(papers))
        paper_dict = {}
        for paper in papers:
            authors = paper['authors']
            names = [precessname(paper_author['name']) for paper_author in authors]
            orgs = [preprocessorg(paper_author['org']) for paper_author in authors if 'org' in paper_author]
            abstract = paper["abstract"] if 'abstract' in paper else ''
            coauther_orgs.append(etl(' '.join(names + orgs) + ' ' + abstract))
        tfidf = TfidfVectorizer().fit_transform(coauther_orgs)
        # sim_mertric = pairwise_distances(tfidf, metric='cosine')

        clf = DBSCAN(metric='cosine')
        s = clf.fit_predict(tfidf)
        # 每个样本所属的簇
        for label, paper in zip(clf.labels_, papers):
            print (label)
            if str(label) not in paper_dict:
                paper_dict[str(label)] = [paper['id']]
            else:
                paper_dict[str(label)].append(paper['id'])
        res_dict[author] = list(paper_dict.values())
    json.dump(res_dict, open('result/result.json', 'w', encoding='utf-8'), indent=4)


# disambiguate_by_cluster()

words = 'a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other,our,own,rather,said,say,says,she,should,since,so,some,than,that,the,their,them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,with,would,yet,you,your'
stop = str.split(words, ',')


def Calculate_Similarity():
    res_dict = {}
    iii = 0
    for author in validate_data:
        print(author)

        author_list = {}
        papers = validate_data[author]
        print(len(papers))
        index = 0
        temp = []
        for paper in papers:
            authors = paper['authors']
            names = [precessname(paper_author['name']) for paper_author in authors]
            orgs = [preprocessorg(paper_author['org']) for paper_author in authors if 'org' in paper_author]
            abstract = paper["abstract"] if 'abstract' in paper else ''
            title = paper["title"] if 'title' in paper else ''
            if not author_list:
                author_list[index] = [paper['id']]
                temp.append(paper['id'])
                index += 1
                continue
            flag = 0
            for key, value in author_list.items():
                score = 0
                if value == paper['id']:
                    continue
                for id in value:
                    wenzhang = validate_pub_data[id]
                    zuozhe = wenzhang['authors']
                    xingmings = [precessname(paper_author['name']) for paper_author in zuozhe]
                    danweis = [preprocessorg(paper_author['org']) for paper_author in zuozhe if 'org' in paper_author]
                    fenshu = 0
                    cnt = 0
                    for name in names:
                        for xingming in xingmings:
                            if name == xingming:
                                cnt += 1
                                break
                    if cnt >= 3:
                        flag = 1
                        if paper['id'] in temp:
                            break
                        author_list[key].append(paper['id'])
                        temp.append(paper['id'])
                        break
                    for org in orgs:
                        for danwei in danweis:
                            if org == danwei:
                                fenshu += 3
                                break
                    if fenshu > 10:
                        fenshu = 10
                    score += fenshu

                    if 'keywords' in paper and 'keywords' in wenzhang:
                        for k in paper['keywords']:
                            for guanjianzi in wenzhang['keywords']:
                                if k == guanjianzi:
                                    score += 5
                    zy = wenzhang["abstract"] if 'abstract' in wenzhang else ''
                    zhaiyao = str.split(zy, ' ')
                    for zhai in zhaiyao:
                        if zhai in stop:
                            zhaiyao.remove(zhai)
                    abstract1 = str.split(abstract, ' ')
                    for zhai in abstract1:
                        if zhai in stop:
                            abstract1.remove(zhai)
                    for zhai in zhaiyao:
                        for yao in abstract1:
                            if zhai == yao:
                                score += 1
                if score > 35:
                    flag = 1
                    if paper['id'] in temp:
                        break
                    temp.append(paper['id'])
                    author_list[key].append(paper['id'])
                    flag = 1
                    break

            if flag == 0:
                author_list[index] = [paper['id']]
                index += 1
        res_dict[author] = list(author_list.values())
    json.dump(res_dict, open('result/zhegeshixinde.json', 'w', encoding='utf-8'), indent=4)


#Calculate_Similarity()


def Calculate_DBSCAN():
    res_dict = {}
    for author in validate_data:
        print(author)
        coauther_orgs = []
        papers = validate_data[author]
        if len(papers) == 0:
            res_dict[author] = []
            continue
        print(len(papers))
        paper_dict = {}
        baocun = []

        for p1 in papers:
            flag = 0
            authors = p1['authors']
            names1 = [precessname(paper_author['name']) for paper_author in authors]
            if p1['id'] in baocun:
                continue
            for p2 in papers:
                authors1 = p2['authors']
                names2 = [precessname(paper_author['name']) for paper_author in authors1]
                if p2['id'] in baocun:
                    continue
                if p1['id'] == p2['id']:
                    continue
                cnt = 0
                for n1 in names1:
                    if cnt >= 3:
                        break
                    for n2 in names2:
                        if n1 == n2:
                            cnt += 1
                            break
                        if cnt >= 3:
                            break
                if cnt >= 3:
                    if p1['id'] not in paper_dict.keys():
                        paper_dict[p1['id']] = []
                    paper_dict[p1['id']].append(p2['id'])
                    baocun.append(p2['id'])
                    flag = 1
            if flag == 1:
                paper_dict[p1['id']].append(p1['id'])
                baocun.append(p1['id'])

        for paper in papers:
            if paper['id'] in baocun:
                continue
            authors = paper['authors']
            names = [precessname(paper_author['name']) for paper_author in authors]
            orgs = [preprocessorg(paper_author['org']) for paper_author in authors if 'org' in paper_author]
            abstract = paper["abstract"] if 'abstract' in paper else ''
            st = ''
            if 'keywords' in paper:
                for key in paper['keywords']:
                    st += key
            coauther_orgs.append(etl(' '.join(names + orgs) + ' ' + st))
        if len(coauther_orgs) == 0:
            res_dict[author] = list(paper_dict.values())
            continue
        tfidf = TfidfVectorizer().fit_transform(coauther_orgs)
        # sim_mertric = pairwise_distances(tfidf, metric='cosine')

        clf = DBSCAN(metric='cosine')
        s = clf.fit_predict(tfidf)
        # 每个样本所属的簇
        for label, paper in zip(clf.labels_, papers):
            if str(label) not in paper_dict:
                paper_dict[str(label)] = [paper['id']]
            else:
                paper_dict[str(label)].append(paper['id'])
        res_dict[author] = list(paper_dict.values())
    json.dump(res_dict, open('result/new.json', 'w', encoding='utf-8'), indent=4)

Calculate_DBSCAN()