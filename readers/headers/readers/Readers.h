#pragma once
#include "IReader.h"

#include "RenderingType.h"

class Readers {
public:
	static IReaderPtr makeKittiReader(const std::string& dir);
	static IReaderPtr makeSintelReader(const std::string& dir, const RenderingType& type);
};