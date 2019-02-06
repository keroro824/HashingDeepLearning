#pragma once
#define BUCKETSIZE 128
class Bucket
{
private:
	int *arr;
	int isInit = -1;
	int index = 0;
	int _counts = 0;
	
public:
	Bucket();
//	static int _size;
//	void setSize(int size);
	int add(int id);
	int retrieve(int index);
	int * getAll();
	int getTotalCounts();
	int getSize();
	~Bucket();
};


