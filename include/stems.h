/* stems — in-process music source separation library, public C API.
 *
 * Phase 0 surface: version query only. The full engine/job API is added in
 * later phases.
 */
#ifndef STEMS_STEMS_H
#define STEMS_STEMS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version of the C ABI declared by this header. Incremented on breaking changes. */
#define STEMS_API_VERSION 1u

/* Returns STEMS_API_VERSION of the linked library. */
uint32_t stems_api_version(void);

/* Returns a static, NUL-terminated version string (library + build info).
 * The pointer stays valid for the lifetime of the process. */
const char* stems_version_string(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* STEMS_STEMS_H */
