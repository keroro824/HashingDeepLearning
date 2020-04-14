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
	void add(int id);
	const int * getAll();
	int getSize() const { return _counts; }
	~Bucket();
};


