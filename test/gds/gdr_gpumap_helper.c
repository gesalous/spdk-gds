#include "gdr_gpumap_helper.h"
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>

#define GPUMAP_PATH "/proc/driver/gdrdrv/gpumap"
#define MAX_GDR_MAPPINGS 64

typedef struct {
    uint64_t cpu_va_start;
    uint64_t gpu_va_start;
    uint64_t length;
} gdr_mapping_t;

static gdr_mapping_t g_cache[MAX_GDR_MAPPINGS];
static int g_cache_entries;
static pthread_once_t g_cache_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_cache_mutex = PTHREAD_MUTEX_INITIALIZER;
static bool g_cache_populated;

static void
gdr_gpumap_populate_locked(void)
{
    FILE *fp = fopen(GPUMAP_PATH, "r");
    char line[256];
    pid_t pid = getpid();

    g_cache_entries = 0;
    if (!fp) {
        g_cache_populated = true;
        return;
    }

    while (fgets(line, sizeof(line), fp) && g_cache_entries < MAX_GDR_MAPPINGS) {
        int tgid;
        uint64_t cpu_va, gpu_va, len;

        if (sscanf(line, "tgid=%d cpu_va=0x%lx gpu_va=0x%lx len=%lu",
                   &tgid, &cpu_va, &gpu_va, &len) == 4 &&
            tgid == pid) {
            g_cache[g_cache_entries].cpu_va_start = cpu_va;
            g_cache[g_cache_entries].gpu_va_start = gpu_va;
            g_cache[g_cache_entries].length = len;
            g_cache_entries++;
        }
    }

    fclose(fp);
    g_cache_populated = true;
}

static void
gdr_gpumap_populate_once(void)
{
    pthread_mutex_lock(&g_cache_mutex);
    gdr_gpumap_populate_locked();
    pthread_mutex_unlock(&g_cache_mutex);
}

uint64_t
gdr_gpumap_translate_cpu_va(uint64_t cpu_va)
{
    pthread_once(&g_cache_once, gdr_gpumap_populate_once);

    pthread_mutex_lock(&g_cache_mutex);
    if (!g_cache_populated) {
        gdr_gpumap_populate_locked();
    }

    for (int i = 0; i < g_cache_entries; ++i) {
        uint64_t start = g_cache[i].cpu_va_start;
        uint64_t end = start + g_cache[i].length;
        if (cpu_va >= start && cpu_va < end) {
            uint64_t pa = g_cache[i].gpu_va_start + (cpu_va - start);
            pthread_mutex_unlock(&g_cache_mutex);
            return pa;
        }
    }

    /* One more attempt after refreshing. */
    gdr_gpumap_populate_locked();
    for (int i = 0; i < g_cache_entries; ++i) {
        uint64_t start = g_cache[i].cpu_va_start;
        uint64_t end = start + g_cache[i].length;
        if (cpu_va >= start && cpu_va < end) {
            uint64_t pa = g_cache[i].gpu_va_start + (cpu_va - start);
            pthread_mutex_unlock(&g_cache_mutex);
            return pa;
        }
    }

    pthread_mutex_unlock(&g_cache_mutex);
    return UINT64_MAX;
}

void
gdr_gpumap_invalidate_cache(void)
{
    pthread_mutex_lock(&g_cache_mutex);
    g_cache_entries = 0;
    g_cache_populated = false;
    pthread_mutex_unlock(&g_cache_mutex);
}