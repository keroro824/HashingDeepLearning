#pragma once
#include "Config.h"
#include <vector>

class Bucket
{
private:
  std::vector<int> arr;
	int isInit = -1;
	int index = 0;
	int _counts = 0;
	
public:
	Bucket();
	int add(int id);
	int * getAll();
	int getTotalCounts();
	int getSize();
	~Bucket();
};


