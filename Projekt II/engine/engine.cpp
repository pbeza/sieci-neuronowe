#include <iostream>

#include "stdafx.h"
#include "engine.h"
#include "cudnn.h"

namespace engine {
	BSTR test()
	{
		std::cout << "hello cuda" << std::endl;
		cudnnStatus_t status;
		cudnnHandle_t handle;
		status = cudnnCreate(&handle);
		std::cout << "status " << status << std::endl;
		cudnnDestroy(handle);
		return ::SysAllocString(L"Greetings from the native world!");
	}
}
