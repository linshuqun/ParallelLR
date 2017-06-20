#include "stdafx.h"
#include <windows.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

#define THREADSIZE 5        // 线程数
#define TIMES 5000          // 迭代次数
#define RATE 0.01           // 学习率
#define BATCH 2500          // batch size

double train[25000][385];   // 25000个训练样本,train[i][384]存放label值
double test[25000][385];    // 25000个测试样本,test[i][384]存放要计算的label值
double weight[THREADSIZE][385];    // 每个线程上的子集的相应权重
int sample[THREADSIZE][BATCH];     // 存储每次迭代所选样本的索引,sample[i][]为线程i所分到的子集的索引
int allSamples[25000];     // 所有样本的索引
int h[THREADSIZE][BATCH / THREADSIZE];  // 每个进程上样本计算出的h值

void initial();       // 初始化
void randomSample();  // 取随机样本
void loadTrainSet();  // 载入训练集
void loadTestSet();   // 载入测试集
void workOut();       // 预测测试集结果同时写入文件
void updateWeight();  // 更新权重
void getH(int mark);  // 计算h(x)
double getJ(int mark, int index);   // 计算J(θ)


DWORD WINAPI ThreadRun(LPVOID lpParameter) {
	int mark = (int)lpParameter;
	getH(mark);    // 计算h(x)
	double tempWeight[385];
	for (int i = 0; i < 385; i++) {
		tempWeight[i] = weight[mark][i] - RATE * getJ(mark, i) / (BATCH / THREADSIZE);   // 计算J(θ)并更新
	}
	for (int i = 0; i < 385; i++) {
		weight[mark][i] = tempWeight[i];
	}
	return 0;
}

int main() {
	srand((unsigned)time(NULL));
	initial();       // 初始化
	loadTrainSet();  // 载入训练集
	HANDLE threads[THREADSIZE];
	// 开始迭代
	int times = TIMES;
	cout << "start iteration..." << endl;
	time_t start = time(NULL);
	while (times--)
	{
		randomSample();   // 取随机样本
		// 开启多线程
		for (int i = 0; i < THREADSIZE; i++) {
			threads[i] = CreateThread(NULL, 3000, ThreadRun, (LPVOID)i, 0, NULL);
		}
		WaitForMultipleObjects(THREADSIZE, threads, TRUE, INFINITE);
		updateWeight();
	}
	time_t end = time(NULL);
	cout << "finish iteration...It takes " << difftime(end, start) << "s" << endl;
	loadTestSet();    // 载入测试集
	workOut();        // 预测测试集结果同时写入文件
	system("pause");
	return 0;
}

void initial() {
	for (int i = 0; i < 25000; i++) {
		allSamples[i] = i;
	}
	for (int i = 0; i < THREADSIZE; i++)
		for (int j = 0; j < 385; j++)
			weight[i][j] = (double)rand() / ((double)RAND_MAX / 3);
}

void loadTrainSet() {
	cout << "start loading the train set" << endl;
	// 读取训练集文件
	FILE* fp;
	errno_t err = fopen_s(&fp, "save_train.txt", "r");

	if (err != 0) {
		cout << "failed to read the train set..." << endl;
		exit(1);
	}

	char firstline[3500];    // 第一行
	fgets(firstline, sizeof(firstline), fp);    // 读取第一行不保存
	for (int i = 0; i < 25000; i++) {
		int id;  // id值，不保存
		fscanf_s(fp, "%d", &id);
		for (int j = 0; j < 385; j++) {
			fscanf_s(fp, "%lf", &train[i][j]);
		}
		if (rand() % 2 == 0)
			train[i][384] = 1;
		else
			train[i][384] = 0;
	}
	fclose(fp);
	cout << "finish reading train set" << endl;
}

void loadTestSet() {
	cout << "start load the test set" << endl;
	// 读取测试集文件
	FILE* fp;
	errno_t err = fopen_s(&fp, "save_test.txt", "r");
	if (err != 0) {
		cout << "faile to read the test set..." << endl;
		exit(1);
	}
	char firstline[3500];    // 第一行
	fgets(firstline, sizeof(firstline), fp);
	for (int i = 0; i < 25000; i++) {
		int id;   // id，不保存
		fscanf_s(fp, "%d", &id);
		for (int j = 0; j < 384; j++) {
			fscanf_s(fp, "%lf", &test[i][j]);
		}
	}
	fclose(fp);
	cout << "finish reading test set" << endl;
}


void workOut() {
	// 预测测试样本的reference值
	for (int i = 0; i < 25000; i++) {
		test[i][384] = 0;
		for (int j = 0; j < 385; j++) {
			if (j != 384)
				test[i][384] += weight[0][j] * test[i][j];
			else
				test[i][384] += weight[0][j];
		}
		test[i][384] = 1 / (1 + 1 / exp(test[i][384]));
	}
	// 将预测结果写入文件中
	ofstream outfile;
	outfile.open("result.csv", ios::out | ios::trunc);
	outfile << "id,label" << endl;
	for (int i = 0; i < 25000; i++)
		outfile << i << "," << test[i][384] << endl;
	outfile.close();
	cout << "finish writing result.csv" << endl;
}

void randomSample() {
	for (int i = 0; i < 25000; i++) {
		int index = rand() % 25000;
		int temp = allSamples[i];
		allSamples[i] = allSamples[index];
		allSamples[index] = temp;
	}
	// 每个进程随机分配到的样本
	for (int i = 0; i < THREADSIZE; i++) {
		int index = i * 5000;
		for (int j = 0; j < 5000; j ++) {
			sample[i][j] = allSamples[index + j];
		}
	}
}

void getH(int mark) {
	int size = BATCH / THREADSIZE;
	for (int i = 0; i < size; i++) {
		h[mark][i] = 0;
		int index = sample[mark][i];
		for (int j = 0; j < 385; j++) {
			if (j == 384) {
				h[mark][i] += weight[mark][384];
			}
			else {
				h[mark][i] += weight[mark][j] * train[index][j];
			}
		}
		h[mark][i] = 1 / (1 + 1 / exp(h[mark][i])) - train[index][384];
	}
}

double getJ(int mark, int index) {
	double J = 0;
	for (int i = 0; i < BATCH / THREADSIZE; i++) {
		int realindex = sample[mark][i];
		double x = train[realindex][index];
		if (index == 384) {
			J += h[mark][i];
		}
		else {
			J += (h[mark][i] * x);
		}
	}
	return J;
}

void updateWeight() {
	for (int i = 0; i < 385; i++) {
		double temp = 0;
		for (int j = 0; j < THREADSIZE; j++)
			temp += weight[j][i];
		temp /= THREADSIZE;
		for (int k = 0; k < THREADSIZE; k++)
			weight[k][i] = temp;
	}
}
