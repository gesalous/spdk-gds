#ifndef GDR_GPUMAP_HELPER_H
#define GDR_GPUMAP_HELPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Translate a CPU virtual address returned by gdr_map() into the GPU BAR
 * physical address using /proc/driver/gdrdrv/gpumap.
 *
 * @returns BAR physical address on success, UINT64_MAX on failure.
 */
uint64_t gdr_gpumap_translate_cpu_va(uint64_t cpu_va);

/* Optional helper if you ever need to repopulate the cache. */
void gdr_gpumap_invalidate_cache(void);

#ifdef __cplusplus
}
#endif

#endif /* GDR_GPUMAP_HELPER_H */