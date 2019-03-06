#pragma once
#include "Config.h"

class Bucket
{
private:
	int *arr;
	int isInit = -1;
	int index = 0;
	int _counts = 0;
	
public:
	Bucket();
	int add(int id);
	int retrieve(int index);
	int * getAll();
	int getTotalCounts();
	int getSize();
	~Bucket();
};


