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

#define THREADSIZE 5        // �߳���
#define TIMES 5000          // ��������
#define RATE 0.01           // ѧϰ��
#define BATCH 2500          // batch size

double train[25000][385];   // 25000��ѵ������,train[i][384]���labelֵ
double test[25000][385];    // 25000����������,test[i][384]���Ҫ�����labelֵ
double weight[THREADSIZE][385];    // ÿ���߳��ϵ��Ӽ�����ӦȨ��
int sample[THREADSIZE][BATCH];     // �洢ÿ�ε�����ѡ����������,sample[i][]Ϊ�߳�i���ֵ����Ӽ�������
int allSamples[25000];     // ��������������
int h[THREADSIZE][BATCH / THREADSIZE];  // ÿ�������������������hֵ

void initial();       // ��ʼ��
void randomSample();  // ȡ�������
void loadTrainSet();  // ����ѵ����
void loadTestSet();   // ������Լ�
void workOut();       // Ԥ����Լ����ͬʱд���ļ�
void updateWeight();  // ����Ȩ��
void getH(int mark);  // ����h(x)
double getJ(int mark, int index);   // ����J(��)


DWORD WINAPI ThreadRun(LPVOID lpParameter) {
	int mark = (int)lpParameter;
	getH(mark);    // ����h(x)
	double tempWeight[385];
	for (int i = 0; i < 385; i++) {
		tempWeight[i] = weight[mark][i] - RATE * getJ(mark, i) / (BATCH / THREADSIZE);   // ����J(��)������
	}
	for (int i = 0; i < 385; i++) {
		weight[mark][i] = tempWeight[i];
	}
	return 0;
}

int main() {
	srand((unsigned)time(NULL));
	initial();       // ��ʼ��
	loadTrainSet();  // ����ѵ����
	HANDLE threads[THREADSIZE];
	// ��ʼ����
	int times = TIMES;
	cout << "start iteration..." << endl;
	time_t start = time(NULL);
	while (times--)
	{
		randomSample();   // ȡ�������
		// �������߳�
		for (int i = 0; i < THREADSIZE; i++) {
			threads[i] = CreateThread(NULL, 3000, ThreadRun, (LPVOID)i, 0, NULL);
		}
		WaitForMultipleObjects(THREADSIZE, threads, TRUE, INFINITE);
		updateWeight();
	}
	time_t end = time(NULL);
	cout << "finish iteration...It takes " << difftime(end, start) << "s" << endl;
	loadTestSet();    // ������Լ�
	workOut();        // Ԥ����Լ����ͬʱд���ļ�
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
	// ��ȡѵ�����ļ�
	FILE* fp;
	errno_t err = fopen_s(&fp, "save_train.txt", "r");

	if (err != 0) {
		cout << "failed to read the train set..." << endl;
		exit(1);
	}

	char firstline[3500];    // ��һ��
	fgets(firstline, sizeof(firstline), fp);    // ��ȡ��һ�в�����
	for (int i = 0; i < 25000; i++) {
		int id;  // idֵ��������
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
	// ��ȡ���Լ��ļ�
	FILE* fp;
	errno_t err = fopen_s(&fp, "save_test.txt", "r");
	if (err != 0) {
		cout << "faile to read the test set..." << endl;
		exit(1);
	}
	char firstline[3500];    // ��һ��
	fgets(firstline, sizeof(firstline), fp);
	for (int i = 0; i < 25000; i++) {
		int id;   // id��������
		fscanf_s(fp, "%d", &id);
		for (int j = 0; j < 384; j++) {
			fscanf_s(fp, "%lf", &test[i][j]);
		}
	}
	fclose(fp);
	cout << "finish reading test set" << endl;
}


void workOut() {
	// Ԥ�����������referenceֵ
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
	// ��Ԥ����д���ļ���
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
	// ÿ������������䵽������
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
