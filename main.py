import argparse
import os

import numpy as np
import torch.cuda
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from configs import load_yaml
from configs import merge_arg
import torch
import torch.nn.functional as F
from tqdm import tqdm


class MyDataSet(Dataset):
    """数据集对象"""
    def __init__(self, datas: list, labels: list, device: torch.device, domains: list):
        self.datas = datas
        self.labels = labels
        self.domains = domains
        self.device = device

    def __getitem__(self, index):
        data = torch.cuda.FloatTensor(np.array([self.datas[index]]))
        label = torch.cuda.LongTensor(np.array([self.labels[index]]))
        domain = torch.cuda.LongTensor(np.array([self.domains[index]]))
        data = data.cuda()
        label = label.cuda()
        domain = domain.cuda()
        return data.squeeze().t(), label.squeeze(), domain.squeeze()

    def __len__(self):
        return len(self.datas)


class FeatureExtractor(torch.nn.Module):
    """特征提取器"""
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(5, 256, 7, 1, 3),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(256, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.linear = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(960, 128)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class Classifier(torch.nn.Module):
    """分类器"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.linear(x))
        return x


class Discriminator(torch.nn.Module):
    """域鉴别器"""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 15)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.softmax(x)
        return x


def parse_args():
    """添加参数"""
    parser = argparse.ArgumentParser(description="Add training args")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path of training args")
    args = parser.parse_args()
    return args


def data_to_people(people_num, datas, labels):
    """拼接数据，获取每个人对应的数据，用于被试独立"""
    peoples_data_list_version = []
    peoples_label_list_version = []
    for i in range(len(datas)):
        if i < people_num:
            peoples_data_list_version.append([datas[i]])
            peoples_label_list_version.append([labels[i]])
        else:
            peoples_data_list_version[i % 15].append(datas[i])
            peoples_label_list_version[i % 15].append(labels[i])
    peoples_data = []
    peoples_label = []
    for i in range(people_num):
        people_data = []
        people_label = []
        for j in range(len(peoples_data_list_version[i])):
            for k in range(len(peoples_data_list_version[i][j])):
                people_data.append(peoples_data_list_version[i][j][k])
                people_label.append(peoples_label_list_version[i][j][k])
        peoples_data.append(people_data)
        peoples_label.append(people_label)
    return peoples_data, peoples_label


def get_datas(test_num, peoples_datas, peoples_labels):
    """根据轮数，获取对应轮次的训练集、验证集和测试集，用于被试独立"""
    train_domains = []
    train_datas = []
    train_labels = []
    validate_domains = []
    validate_datas = []
    validate_labels = []
    flag = True
    for people in range(len(peoples_datas)):
        if people == test_num:
            test_datas = peoples_datas[people]
            test_labels = peoples_labels[people]
            test_domains = [people] * len(peoples_labels[people])
        else:
            if flag:
                validate_datas = peoples_datas[people]
                validate_labels = peoples_labels[people]
                validate_domains = [people] * len(peoples_labels[people])
                flag=False
            else:
                for i in range(len(peoples_datas[people])):
                    train_datas.append(peoples_datas[people][i])
                    train_labels.append(peoples_labels[people][i])
                    train_domains.append(people)
    return train_datas, train_labels, test_datas, test_labels, train_domains, test_domains, validate_datas, validate_labels, validate_domains


def train(feature_extractor: FeatureExtractor, classifier: Classifier,
          discriminator: Discriminator, train_dataloader: DataLoader, validate_dataloader: DataLoader):
    """训练网络"""
    print("Train:")
    feature_extractor.train()
    classifier.train()
    discriminator.train()
    feature_extractor_optimizer = optim.SGD(feature_extractor.parameters(), lr=0.001, momentum=0.9)
    classifier_optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
    discriminator_optimizer = optim.SGD(discriminator.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    loop_num = 100
    acc_best = 0
    best_feature_extractor = FeatureExtractor()
    best_classifier = Classifier()
    progress_bar = tqdm(range(loop_num), position=0)
    progress_bar.set_description("Loops")
    for epoch in range(loop_num):
        loss_tol = 0
        dloss_tol = 0
        num = 0
        for i, data in enumerate(train_dataloader, 0):
            #训练分类器和特征抽取器
            input, label, domain = data
            classifier_optimizer.zero_grad()
            feature_extractor_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()
            feature = feature_extractor(input)
            out_label = classifier(feature)
            loss_label = criterion(out_label, label)
            loss_tol += loss_label.item()
            loss_label.backward()
            classifier_optimizer.step()
            feature_extractor_optimizer.step()
            #训练域分类器
            classifier_optimizer.zero_grad()
            feature_extractor_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()
            feature = feature_extractor(input)
            out_domain = discriminator(feature)
            loss_domain = criterion(out_domain, domain)
            dloss_tol += loss_domain.item()
            loss_domain.backward()
            discriminator_optimizer.step()
            #训练特征抽取器(域对抗部分)
            feature_extractor_optimizer.zero_grad()
            feature = feature_extractor(input)
            out_domain = discriminator(feature)
            loss_domain = criterion(out_domain, domain)
            loss_domain.backward(torch.tensor(-1.0))
            feature_extractor_optimizer.step()
            num += 1
        avg_loss = loss_tol / num
        dloss = dloss_tol / num
        #用验证集验证，保存训练效果好的网络参数
        acc = test(feature_extractor, classifier, validate_dataloader)
        if acc > acc_best:
            acc_best = acc
            best_feature_extractor.load_state_dict(feature_extractor.state_dict())
            best_classifier.load_state_dict(classifier.state_dict())
        logs = {"loss": avg_loss, "dloss": dloss}
        progress_bar.set_postfix(**logs)
        progress_bar.update(1)
    progress_bar.close()
    return best_feature_extractor, best_classifier


def test(feature_extractor: FeatureExtractor, classifier: Classifier, test_dataloader: DataLoader):
    """测试模型的准确率"""
    sum = 0
    correct = 0
    feature_extractor.eval()
    classifier.eval()
    for i, data in enumerate(test_dataloader, 0):
        inputs, labels, domains = data
        features = feature_extractor(inputs)
        outputs = classifier(features)
        _, predicts = torch.max(outputs, dim=1)
        for j in range(predicts.size(0)):
            sum += 1
            if predicts[j] == labels[j]:
                correct += 1
    return (correct / sum) * 100


def train_and_test(train_datas: list, train_labels: list, test_datas: list, test_labels: list, train_domains: list,
                   test_domains: list, validate_datas: list, validate_labels: list, validate_domains: list):
    """训练并测试模型，返回准确率"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(torch.cuda.get_device_name(device))
    train_dataset = MyDataSet(train_datas, train_labels, device, train_domains)
    test_dataset = MyDataSet(test_datas, test_labels, device, test_domains)
    validate_dataset = MyDataSet(validate_datas, validate_labels,device, validate_domains)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=0)
    validate_dataloader = DataLoader(dataset=validate_dataset, batch_size=16, shuffle=False, num_workers=0)
    feature_extractor = FeatureExtractor()
    classifier = Classifier()
    discriminator = Discriminator()
    feature_extractor.to(device)
    classifier.to(device)
    discriminator.to(device)
    best_feature_extractor, best_classifier = train(feature_extractor, classifier, discriminator, train_dataloader,
                                                    validate_dataloader)
    print("Test:")
    best_feature_extractor.to(device)
    best_classifier.to(device)
    acc = test(best_feature_extractor, best_classifier, test_dataloader)
    return acc


def normalize(peoples_data: list):
    """正则化数据集中的数据，使用min_max方法"""
    data_min = peoples_data[0][0].min()
    data_max = peoples_data[0][0].max()
    for i in range(len(peoples_data)):
        for j in range(len(peoples_data[i])):
            now_min = peoples_data[i][j].min()
            now_max = peoples_data[i][j].max()
            if now_max > data_max:
                data_max = now_max
            if now_min < data_min:
                data_min = now_min
    for i in range(len(peoples_data)):
        for j in range(len(peoples_data[i])):
            peoples_data[i][j] = (peoples_data[i][j] - data_min) / (data_max - data_min)
    return peoples_data


if __name__ == '__main__':
    #设置参数
    args = parse_args()
    config = load_yaml(args.config)
    args = merge_arg(args, config)
    #读取数据
    dir_name = './SEED-IV'
    sessions = os.listdir(dir_name)
    train_datas = []
    train_labels = []
    test_datas = []
    test_labels = []
    for session in sessions:
        session_dir = dir_name + '/' + session
        peoples = os.listdir(session_dir)
        for people in peoples:
            people_dir = session_dir + '/' + people + '/'
            train_data = np.load(people_dir + 'train_data.npy')
            train_label = np.load(people_dir + 'train_label.npy')
            test_data = np.load(people_dir + 'test_data.npy')
            test_label = np.load(people_dir + 'test_label.npy')
            train_datas.append(train_data)
            train_labels.append(train_label)
            test_datas.append(test_data)
            test_labels.append(test_label)
    #获取每个人对应的数据
    peoples_data, peoples_label = data_to_people(15, train_datas + test_datas, train_labels + test_labels)
    #正则化数据
    peoples_data = normalize((peoples_data))
    accs = []
    for i in range(15):
        print("Experiment " + str(i) + ":")
        #获取训练集、验证集和测试集
        train_datas, train_labels, test_datas, test_labels, train_domains, test_domains, validate_datas, validate_labels, validate_domains = get_datas(
            i, peoples_data,
            peoples_label)
        #训练并测试
        acc = train_and_test(train_datas, train_labels, test_datas, test_labels, train_domains, test_domains,
                             validate_datas, validate_labels, validate_domains)
        accs.append(acc)
    print(accs)
    print("average_acc:" + str(np.array(accs).sum() / len(accs)) + "%")
