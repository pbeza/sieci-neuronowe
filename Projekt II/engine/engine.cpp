#include "stdafx.h"

#include "engine.h"

namespace engine {
	BSTR test()
	{
		return ::SysAllocString(L"Greetings from the native world!");
	}
}
