#pragma once
#include "IReader.h"

class Readers {
public:
	static IReaderPtr makeKittiReader(const std::string& dir);
	static IReaderPtr makeCintelReader(const std::string& dir);
};