#include "stems.h"

extern "C" uint32_t stems_api_version(void) { return STEMS_API_VERSION; }

extern "C" const char* stems_version_string(void) { return "stems " STEMS_VERSION; }
